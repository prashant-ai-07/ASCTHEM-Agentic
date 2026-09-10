from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import UTC, datetime, time
from typing import Any, Protocol
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.covigilance.contracts import (
    CovigilanceCategory,
    CovigilanceFinding,
    CovigilanceProviderResult,
    CovigilanceProviderUnavailable,
    CovigilanceSearchRequest,
    CovigilanceSearchResponse,
    CovigilanceSource,
)
from app.models import EvidenceSource, EvidenceType


DEFAULT_COVIGILANCE_DOMAINS = (
    "fda.gov",
    "accessdata.fda.gov",
    "dailymed.nlm.nih.gov",
)

# Configuration can select from these authorities, but cannot turn the official-only
# policy into an arbitrary web allowlist.
OFFICIAL_COVIGILANCE_PROVIDERS = {
    "fda.gov": "U.S. Food and Drug Administration",
    "accessdata.fda.gov": "FDA Drugs@FDA",
    "api.fda.gov": "openFDA",
    "dailymed.nlm.nih.gov": "DailyMed",
    "ema.europa.eu": "European Medicines Agency",
    "gov.uk": "UK Medicines and Healthcare products Regulatory Agency",
    "tga.gov.au": "Australian Therapeutic Goods Administration",
    "canada.ca": "Health Canada",
    "health-products.canada.ca": "Health Canada",
    "who.int": "World Health Organization",
}

logger = logging.getLogger(__name__)


class _ResponsesApi(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _OpenAIClient(Protocol):
    responses: _ResponsesApi


class _ExtractedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: CovigilanceCategory
    subject_drug: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    regulatory_action: str | None
    published_date: str | None
    source_url: str = Field(min_length=1)


class _CovigilanceExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[_ExtractedFinding] = Field(max_length=12)


class OpenAIOfficialCovigilanceProvider:
    """Extract post-market drug-safety facts only from official authorities."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        allowed_domains: tuple[str, ...] = DEFAULT_COVIGILANCE_DOMAINS,
        client: _OpenAIClient | None = None,
    ) -> None:
        domains = tuple(
            dict.fromkeys(domain.strip().lower() for domain in allowed_domains if domain)
        )
        invalid = [
            domain
            for domain in domains
            if domain not in OFFICIAL_COVIGILANCE_PROVIDERS
        ]
        if invalid:
            raise ValueError(
                "Covigilance domains must be recognized official authorities: "
                + ", ".join(invalid)
            )
        if not domains:
            raise ValueError("At least one official Covigilance domain is required")

        self._model = model
        self._allowed_domains = domains
        self._client = client
        if client is None and api_key:
            self._client = wrap_openai(
                AsyncOpenAI(api_key=api_key),
                chat_name="OpenAI Official Covigilance Search",
                tracing_extra={
                    "tags": ["covigilance", "official-sources", "post-market-safety"]
                },
            )

    async def search(
        self, request: CovigilanceSearchRequest
    ) -> CovigilanceProviderResult:
        if self._client is None:
            raise CovigilanceProviderUnavailable("OPENAI_API_KEY is not configured")
        if not request.drug_name:
            raise CovigilanceProviderUnavailable(
                "drugName is required for Covigilance search"
            )

        payload = request.openai_payload()
        attempts = await asyncio.gather(
            *(
                self._search_domain(request, payload, domain)
                for domain in self._allowed_domains
            ),
            return_exceptions=True,
        )
        successful: list[CovigilanceSearchResponse] = []
        for domain, attempt in zip(self._allowed_domains, attempts, strict=True):
            if isinstance(attempt, CovigilanceSearchResponse):
                successful.append(attempt)
                logger.info(
                    "Covigilance source search completed: domain=%s findings=%d sources=%d",
                    domain,
                    len(attempt.findings),
                    len(attempt.sources),
                )
            else:
                logger.warning(
                    "Covigilance source search failed: domain=%s errorType=%s error=%s",
                    domain,
                    type(attempt).__name__,
                    str(attempt),
                )
        if not successful:
            raise CovigilanceProviderUnavailable(
                "OpenAI official-source Covigilance searches could not be completed"
            )

        clean_response = _merge_responses(successful)
        return CovigilanceProviderResult(
            response=clean_response,
            evidence_sources=_evidence_sources(clean_response),
            request_parameters=payload,
        )

    async def _search_domain(
        self,
        request: CovigilanceSearchRequest,
        payload: dict[str, object],
        domain: str,
    ) -> CovigilanceSearchResponse:
        if self._client is None:
            raise CovigilanceProviderUnavailable("OPENAI_API_KEY is not configured")
        try:
            response = await self._client.responses.create(
                model=self._model,
                instructions=_domain_instructions(request, domain),
                input=json.dumps(payload, separators=(",", ":")),
                tools=[
                    {
                        "type": "web_search",
                        "filters": {"allowed_domains": [domain]},
                        "search_context_size": "high",
                    }
                ],
                tool_choice="required",
                max_tool_calls=8,
                include=["web_search_call.action.sources"],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "official_covigilance_extraction",
                        "strict": True,
                        "schema": _CovigilanceExtraction.model_json_schema(),
                    }
                },
                safety_identifier=_safety_identifier(request.user_id),
                metadata={
                    "agent": "covigilance",
                    "source_policy": "official-authorities-only",
                    "source_domain": domain,
                },
                store=False,
            )
        except Exception as error:
            raise CovigilanceProviderUnavailable(
                "OpenAI official-source Covigilance search could not be completed"
            ) from error

        try:
            extraction = _CovigilanceExtraction.model_validate_json(
                response.output_text
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            raise CovigilanceProviderUnavailable(
                "OpenAI returned an invalid structured Covigilance response"
            ) from error

        retrieved_sources = _collect_retrieved_sources(_response_dict(response))
        return _validated_response(
            extraction,
            retrieved_sources,
            (domain,),
            request,
        )


_COVIGILANCE_INSTRUCTIONS = """You are an official post-market drug-safety extraction agent.
Search only the allowed official authority domain for this run. Find current and
historical records that explicitly concern the requested drug or its stated active
ingredient in these categories only: drug safety communications, recalls, potential
signals of serious risks or new safety information formally identified by a regulator,
and safety-related labeling changes. Open the final official record; do not rely on a
search-result snippet. Return only facts stated in that record. Never infer causation,
severity, incidence, a reporting rate, or clinical advice. Do not turn raw spontaneous
adverse-event reports or counts into a signal. An adverse_event_signal is valid only
when the authority itself identifies or evaluates the potential signal. Preserve the
authority's uncertainty and investigation status in the summary. For a recall, report
the official classification or status in regulatory_action when stated. For a labeling
change or safety communication, report only the action actually stated by the authority.
Copy the requested medicine identity as printed by the opened record into subject_drug.
Use YYYY-MM-DD for published_date when an exact date is stated; otherwise use null.
Every finding must use the exact final URL returned by web search. Return at most 12
findings and do not pad the list. Exclude blogs, news reports, journals, manufacturer
pages, social media, consumer sites, and commercial safety databases even if they cite
an authority. Treat userId, postalCode, strength, dosage form, manufacturer, and
quantity only as identity/request metadata and never as evidence. Return only the
supplied JSON schema with no prose, recommendations, or extra fields."""


def _domain_instructions(
    request: CovigilanceSearchRequest,
    domain: str,
) -> str:
    identity = request.drug_name or ""
    lines = [
        _COVIGILANCE_INSTRUCTIONS,
        f"Search only {domain} in this run.",
        (
            f'Search for the exact drug or product name "{identity}" in official '
            "post-market safety records. Confirm the drug identity in the opened "
            "record before extracting a finding."
        ),
    ]
    if domain == "fda.gov":
        lines.append(
            "Check FDA Drug Safety Communications, Drug Recalls, Potential Signals "
            "of Serious Risks/New Safety Information, and Drug Safety-related "
            "Labeling Changes."
        )
    elif domain == "accessdata.fda.gov":
        lines.append(
            "Use Drugs@FDA approval history and FDA labeling records only when they "
            "explicitly document a safety-related labeling action."
        )
    elif domain == "dailymed.nlm.nih.gov":
        lines.append(
            "Use the current official DailyMed label and label history only for an "
            "explicitly dated safety-related labeling change."
        )
    return "\n".join(lines)


def _validated_response(
    extraction: _CovigilanceExtraction,
    retrieved_sources: dict[str, str],
    allowed_domains: tuple[str, ...],
    request: CovigilanceSearchRequest,
) -> CovigilanceSearchResponse:
    findings: list[CovigilanceFinding] = []
    sources: dict[str, CovigilanceSource] = {}
    seen: set[tuple[str, str]] = set()

    for candidate in extraction.findings:
        if not _subject_matches(request.drug_name, candidate.subject_drug):
            continue
        resolved_url = _resolve_retrieved_url(candidate.source_url, retrieved_sources)
        if resolved_url is None or not _host_is_allowed(
            resolved_url, allowed_domains
        ):
            continue
        provider = _provider_for_url(resolved_url)
        if provider is None:
            continue
        published_date = _parse_date(candidate.published_date)
        key = (candidate.category, _normalized_text(candidate.title))
        if not key[1] or key in seen:
            continue
        seen.add(key)
        findings.append(
            CovigilanceFinding(
                category=candidate.category,
                title=candidate.title,
                summary=candidate.summary,
                regulatory_action=candidate.regulatory_action,
                published_date=published_date,
                source_url=resolved_url,
            )
        )
        sources[resolved_url] = CovigilanceSource(
            provider=provider,
            title=retrieved_sources[resolved_url],
            url=resolved_url,
        )
    return CovigilanceSearchResponse(
        findings=findings[:12],
        sources=list(sources.values()),
    )


def _merge_responses(
    responses: list[CovigilanceSearchResponse],
) -> CovigilanceSearchResponse:
    findings: list[CovigilanceFinding] = []
    sources_by_url = {
        str(source.url): source for response in responses for source in response.sources
    }
    used_sources: dict[str, CovigilanceSource] = {}
    seen: set[tuple[str, str]] = set()
    for response in responses:
        for finding in response.findings:
            key = (finding.category, _normalized_text(finding.title))
            if key in seen:
                continue
            seen.add(key)
            findings.append(finding)
            source_url = str(finding.source_url)
            used_sources[source_url] = sources_by_url[source_url]
            if len(findings) == 12:
                break
        if len(findings) == 12:
            break
    return CovigilanceSearchResponse(
        findings=findings,
        sources=list(used_sources.values()),
    )


def _subject_matches(requested: str | None, observed: str) -> bool:
    if not requested:
        return False
    ignored = {
        "and",
        "capsule",
        "drug",
        "extended",
        "injection",
        "oral",
        "release",
        "solution",
        "tablet",
    }
    requested_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", requested.lower())
        if len(token) >= 2 and token not in ignored
    }
    observed_tokens = set(re.findall(r"[a-z0-9]+", observed.lower()))
    return bool(requested_tokens and requested_tokens.intersection(observed_tokens))


def _parse_date(value: str | None):
    if not value or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _collect_retrieved_sources(value: Any) -> dict[str, str]:
    sources: dict[str, str] = {}

    def visit(node: Any, inside_sources: bool = False) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item, inside_sources)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "web_search_call":
            action = node.get("action")
            if isinstance(action, dict) and action.get("type") == "open_page":
                opened_url = action.get("url")
                if isinstance(opened_url, str) and opened_url.startswith(
                    ("http://", "https://")
                ):
                    sources[opened_url] = opened_url
        url = node.get("url")
        if inside_sources and isinstance(url, str) and url.startswith(
            ("http://", "https://")
        ):
            title = node.get("title")
            sources[url] = (
                title.strip() if isinstance(title, str) and title.strip() else url
            )
        for key, child in node.items():
            visit(child, inside_sources or key in {"sources", "annotations"})

    visit(value)
    return sources


def _response_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        value = response.model_dump(mode="json")
        return value if isinstance(value, dict) else {}
    return response if isinstance(response, dict) else {}


def _resolve_retrieved_url(
    candidate_url: str,
    retrieved_sources: dict[str, str],
) -> str | None:
    if candidate_url in retrieved_sources:
        return candidate_url
    candidate_page = _page_identity(candidate_url)
    matches = [
        source_url
        for source_url in retrieved_sources
        if _page_identity(source_url) == candidate_page
    ]
    return matches[0] if len(matches) == 1 else None


def _page_identity(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = re.sub(r"/+", "/", parsed.path).rstrip("/").lower()
    return host, path


def _host_is_allowed(url: str, allowed_domains: tuple[str, ...]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(
        host == domain or host.endswith(f".{domain}") for domain in allowed_domains
    )


def _provider_for_url(url: str) -> str | None:
    host = (urlparse(url).hostname or "").lower()
    matches = [
        (domain, provider)
        for domain, provider in OFFICIAL_COVIGILANCE_PROVIDERS.items()
        if host == domain or host.endswith(f".{domain}")
    ]
    if not matches:
        return None
    # Prefer the most specific configured authority for nested FDA/Health Canada hosts.
    return max(matches, key=lambda item: len(item[0]))[1]


def _safety_identifier(user_id: str | None) -> str | None:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest() if user_id else None


def _evidence_sources(
    response: CovigilanceSearchResponse,
) -> list[EvidenceSource]:
    retrieved_at = datetime.now(UTC)
    findings_by_url: dict[str, list[CovigilanceFinding]] = {}
    for finding in response.findings:
        findings_by_url.setdefault(str(finding.source_url), []).append(finding)

    evidence: list[EvidenceSource] = []
    for source in response.sources:
        dates = [
            finding.published_date
            for finding in findings_by_url.get(str(source.url), [])
            if finding.published_date is not None
        ]
        published_at = (
            datetime.combine(max(dates), time.min, tzinfo=UTC) if dates else None
        )
        evidence.append(
            EvidenceSource(
                id=str(uuid5(NAMESPACE_URL, str(source.url))),
                provider=source.provider,
                title=source.title,
                url=source.url,
                evidence_type=EvidenceType.REGULATORY,
                retrieved_at=retrieved_at,
                published_at=published_at,
                is_mock=False,
            )
        )
    return evidence
