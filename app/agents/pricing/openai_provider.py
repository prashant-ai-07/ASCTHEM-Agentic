from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from urllib.parse import parse_qs, urlencode, urlparse
from uuid import NAMESPACE_URL, uuid5

from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.pricing.contracts import (
    PriceOffer,
    PricingProviderResult,
    PricingProviderUnavailable,
    PricingSearchRequest,
    PricingSearchResponse,
    PricingSource,
)
from app.models import EvidenceSource, EvidenceType


logger = logging.getLogger(__name__)


class _ResponsesApi(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _OpenAIClient(Protocol):
    responses: _ResponsesApi


class _ExtractedOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    pharmacy: str = Field(min_length=1)
    price: float = Field(ge=0)
    currency: Literal["USD"]
    savings_type: Literal["cash", "coupon", "membership", "retail", "unknown"]
    coupon_code: str | None
    source_url: str = Field(min_length=1)
    matched_strength: str | None


class _PricingExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offers: list[_ExtractedOffer]


class OpenAIWebPricingProvider:
    """Extract source-backed pricing with Responses API web search."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        allowed_domains: tuple[str, ...],
        client: _OpenAIClient | None = None,
    ) -> None:
        self._model = model
        self._allowed_domains = tuple(dict.fromkeys(allowed_domains))
        self._client = client
        if client is None and api_key:
            self._client = wrap_openai(
                AsyncOpenAI(api_key=api_key),
                chat_name="OpenAI Pricing Web Search",
                tracing_extra={"tags": ["pricing", "web-search"]},
            )

    async def search(self, request: PricingSearchRequest) -> PricingProviderResult:
        if self._client is None:
            raise PricingProviderUnavailable("OPENAI_API_KEY is not configured")
        if not self._allowed_domains:
            raise PricingProviderUnavailable(
                "PRICING_SOURCE_DOMAINS does not contain any providers"
            )

        payload = request.openai_payload()
        attempts = await asyncio.gather(
            *(
                self._search_domain(request, payload, domain, exhaustive=False)
                for domain in self._allowed_domains
            ),
            return_exceptions=True,
        )

        successful: list[PricingSearchResponse] = []
        retry_domains: list[str] = []
        for domain, attempt in zip(self._allowed_domains, attempts, strict=True):
            if isinstance(attempt, PricingSearchResponse):
                successful.append(attempt)
                logger.info(
                    "Pricing provider search completed: domain=%s offers=%d sources=%d",
                    domain,
                    len(attempt.offers),
                    len(attempt.sources),
                )
                if len(attempt.offers) <= 1:
                    retry_domains.append(domain)
            else:
                retry_domains.append(domain)
                logger.warning(
                    "Pricing provider search failed: domain=%s errorType=%s error=%s",
                    domain,
                    type(attempt).__name__,
                    str(attempt),
                )

        if retry_domains:
            retries = await asyncio.gather(
                *(
                    self._search_domain(request, payload, domain, exhaustive=True)
                    for domain in retry_domains
                ),
                return_exceptions=True,
            )
            for domain, retry in zip(retry_domains, retries, strict=True):
                if isinstance(retry, PricingSearchResponse):
                    successful.append(retry)
                    logger.info(
                        "Sparse pricing search retried: domain=%s offers=%d sources=%d",
                        domain,
                        len(retry.offers),
                        len(retry.sources),
                    )
                else:
                    logger.warning(
                        "Sparse pricing retry failed: domain=%s errorType=%s error=%s",
                        domain,
                        type(retry).__name__,
                        str(retry),
                    )

        if not successful:
            raise PricingProviderUnavailable(
                "OpenAI pricing searches could not be completed"
            )

        clean_response = _merge_responses(successful)
        evidence_sources = _evidence_sources(clean_response.sources)
        return PricingProviderResult(
            response=clean_response,
            evidence_sources=evidence_sources,
            request_parameters=payload,
        )

    async def _search_domain(
        self,
        request: PricingSearchRequest,
        payload: dict[str, object],
        domain: str,
        exhaustive: bool,
    ) -> PricingSearchResponse:
        if self._client is None:
            raise PricingProviderUnavailable("OPENAI_API_KEY is not configured")

        try:
            response = await self._client.responses.create(
                model=self._model,
                instructions=_domain_instructions(request, domain, exhaustive),
                input=json.dumps(payload, separators=(",", ":")),
                tools=[
                    {
                        "type": "web_search",
                        "filters": {"allowed_domains": [domain]},
                        "search_context_size": "high",
                    }
                ],
                tool_choice="required",
                max_tool_calls=5 if exhaustive else 3,
                include=["web_search_call.action.sources"],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "pricing_web_extraction",
                        "strict": True,
                        "schema": _PricingExtraction.model_json_schema(),
                    }
                },
                safety_identifier=_safety_identifier(request.user_id),
                metadata={"pricing_source_domain": domain},
                store=False,
            )
        except Exception as error:
            raise PricingProviderUnavailable(
                "OpenAI pricing search could not be completed"
            ) from error

        try:
            extraction = _PricingExtraction.model_validate_json(response.output_text)
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            raise PricingProviderUnavailable(
                "OpenAI returned an invalid structured pricing response"
            ) from error

        response_data = _response_dict(response)
        retrieved_sources = _collect_retrieved_sources(response_data)
        clean_response = _validated_response(
            extraction,
            retrieved_sources,
            (domain,),
            request,
        )
        return clean_response


_PRICING_INSTRUCTIONS = """You are a precise medication-price extraction agent. Use web search for every request.

Exact Constraints: Medication strength and quantity are absolute constraints, never preferences. Return an offer only when the browsed page explicitly displays both the exact requested strength and the exact requested quantity. Never substitute a nearest strength or normalize a quantity (e.g., do not extrapolate a 90-day price from a 30-day supply). Copy the exact strength and quantity printed beside the price into matched_strength and matched_quantity. If the page does not explicitly expose both, discard the result.

Location Verification: Postal code is a mandatory search constraint. You must verify the price is localized. Return a price only if the browsed page or exact URL explicitly reflects the requested postal code, city, or state. Discard results that lack geographic confirmation or default to a national average.

Price & Pharmacy: Report only a price explicitly visible in a browsed source. Never estimate, calculate, combine, or infer a price. The pharmacy name must be clearly stated in the searchable text; never return an offer with an "Unknown" pharmacy or from an unverified source.

Source URL & Metadata: For each offer, copy the exact final source URL returned by web search. Never return a search results URL, a generic provider home page, or an invented URL. Treat userId strictly as opaque request metadata and never use it as a search term.

Output Formatting: Return exactly and only the supplied JSON schema. Output the raw JSON string directly. Do NOT wrap the JSON in Markdown code blocks (e.g., do not use ```json). Do not include any conversational prose, medical advice, recommendations, or additional fields outside the schema."""


def _domain_instructions(
    request: PricingSearchRequest,
    domain: str,
    exhaustive: bool,
) -> str:
    medication_slug = "-".join(
        re.findall(r"[a-z0-9]+", request.drug_name.lower())
    )
    host = domain if domain.startswith("www.") else f"www.{domain}"
    candidate_url = f"https://{host}/{medication_slug}"
    url_strength = re.sub(r"\s+", "", request.strength.lower())
    if re.fullmatch(r"\d+(?:\.\d+)?", url_strength):
        url_strength = f"{url_strength}mg"
    candidate_parameters = {
        "label_override": medication_slug,
        "dosage": url_strength,
        "location": request.postal_code,
    }
    if request.dosage_form:
        candidate_parameters["form"] = request.dosage_form.strip().lower()
    if request.quantity is not None and request.quantity > 1:
        candidate_parameters["quantity"] = f"{request.quantity:g}"
    parameterized_candidate_url = (
        f"{candidate_url}?{urlencode(candidate_parameters)}"
    )
    lines = [
            _PRICING_INSTRUCTIONS,
            f"Search only {domain} in this run.",
            (
                f'Search for: site:{domain} "{request.drug_name}" '
                f'"{request.strength}" "{request.postal_code}" price coupon.'
            ),
            (
                "Try opening these generated medication-page candidates directly: "
                f"{parameterized_candidate_url} and {candidate_url}. Prefer the "
                "parameterized URL because it carries the requested location. If "
                "that provider uses a different URL pattern, "
                "find and open its exact medication page instead."
            ),
            f"Use the mandatory postal code {request.postal_code} for this search.",
            (
                "Open the provider's exact medication pricing page; "
                "do not rely only on a search snippet."
            ),
            (
                "Return one offer for every visible pharmacy price; do not select "
                "only one pharmacy or only the page's headline price."
            ),
    ]
    if exhaustive:
        lines.append(
            "This is an exhaustive retry because the first pass returned at most "
            "one offer. Reopen the exact medication page, apply the requested "
            "strength and postal code, expand the pharmacy list, and return every "
            "visible pharmacy-price row. Do not repeat only the initially selected "
            "pharmacy."
        )
    return "\n".join(lines)


def _merge_responses(
    responses: list[PricingSearchResponse],
) -> PricingSearchResponse:
    offers: dict[tuple[str, str, float, str], PriceOffer] = {}
    sources: dict[str, PricingSource] = {}
    for response in responses:
        for source in response.sources:
            sources[str(source.url)] = source
        for offer in response.offers:
            key = (
                offer.provider,
                offer.pharmacy,
                offer.price,
                str(offer.source_url),
            )
            offers[key] = offer

    sorted_offers = sorted(offers.values(), key=lambda offer: offer.price)
    return PricingSearchResponse(
        offers=sorted_offers,
        sources=list(sources.values()),
    )


def _safety_identifier(user_id: str | None) -> str | None:
    if not user_id:
        return None
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def _response_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        value = response.model_dump(mode="json")
        return value if isinstance(value, dict) else {}
    return response if isinstance(response, dict) else {}


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
        if (
            inside_sources
            and isinstance(url, str)
            and url.startswith(("http://", "https://"))
        ):
            title = node.get("title")
            sources[url] = (
                title.strip()
                if isinstance(title, str) and title.strip()
                else url
            )

        for key, child in node.items():
            visit(child, inside_sources or key in {"sources", "annotations"})

    visit(value)
    return sources


def _validated_response(
    extraction: _PricingExtraction,
    retrieved_sources: dict[str, str],
    allowed_domains: tuple[str, ...],
    request: PricingSearchRequest,
) -> PricingSearchResponse:
    offers: list[PriceOffer] = []
    sources: dict[str, PricingSource] = {}

    for candidate in extraction.offers:
        if not _strength_matches(request.strength, candidate.matched_strength):
            continue
        resolved_url = _resolve_retrieved_url(
            candidate.source_url,
            retrieved_sources,
        )
        if resolved_url is None:
            continue
        if not _host_is_allowed(resolved_url, allowed_domains):
            continue
        if not _location_matches_request(resolved_url, request.postal_code):
            continue
        if not _source_matches_request(resolved_url, request):
            continue
        provider = candidate.provider.strip()
        offer = PriceOffer(
            provider=provider,
            pharmacy=candidate.pharmacy,
            price=candidate.price,
            currency=candidate.currency,
            savings_type=candidate.savings_type,
            coupon_code=candidate.coupon_code,
            source_url=resolved_url,
        )
        offers.append(offer)
        sources[resolved_url] = PricingSource(
            provider=provider,
            title=retrieved_sources[resolved_url],
            url=resolved_url,
        )

    offers.sort(key=lambda offer: offer.price)
    return PricingSearchResponse(offers=offers, sources=list(sources.values()))


def _strength_matches(requested: str | None, observed: str | None) -> bool:
    """Conservatively compare the requested strength with source-observed text."""
    if not requested:
        return True
    if not observed:
        return False

    requested_value = _strength_parts(requested)
    observed_value = _strength_parts(observed)
    if requested_value is None or observed_value is None:
        return False

    requested_numbers, requested_units = requested_value
    observed_numbers, observed_units = observed_value
    if requested_numbers != observed_numbers:
        return False
    # A bare API value such as "40" may match an explicitly observed "40 mg".
    # When the caller supplied units, require those units to match too.
    return not requested_units or requested_units == observed_units


def _strength_parts(value: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    normalized = (
        value.strip()
        .lower()
        .replace("μg", "mcg")
        .replace("µg", "mcg")
        .replace("micrograms", "mcg")
        .replace("milligrams", "mg")
        .replace("grams", "g")
        .replace("milliliters", "ml")
    )
    numbers = tuple(
        token.rstrip("0").rstrip(".") if "." in token else token.lstrip("0") or "0"
        for token in re.findall(r"\d+(?:\.\d+)?", normalized)
    )
    if not numbers:
        return None
    units = tuple(re.findall(r"[a-z]+", normalized))
    return numbers, units


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


def _source_matches_request(url: str, request: PricingSearchRequest) -> bool:
    if not request.drug_name:
        return True
    path_tokens = set(re.findall(r"[a-z0-9]+", urlparse(url).path.lower()))
    drug_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", request.drug_name.lower())
        if len(token) >= 4
    ]
    return not drug_tokens or any(token in path_tokens for token in drug_tokens)


def _location_matches_request(url: str, postal_code: str) -> bool:
    """Reject a source URL when it explicitly identifies another location."""
    query = parse_qs(urlparse(url).query)
    location_keys = ("location", "zip", "zipcode", "postal_code", "postalCode")
    observed_locations = [
        value.strip()
        for key in location_keys
        for value in query.get(key, [])
        if value.strip()
    ]
    return not observed_locations or all(
        location == postal_code for location in observed_locations
    )


def _host_is_allowed(url: str, allowed_domains: tuple[str, ...]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(
        host == domain or host.endswith(f".{domain}")
        for domain in allowed_domains
    )


def _evidence_sources(sources: list[PricingSource]) -> list[EvidenceSource]:
    retrieved_at = datetime.now(UTC)
    return [
        EvidenceSource(
            id=str(uuid5(NAMESPACE_URL, str(source.url))),
            provider=source.provider,
            title=source.title,
            url=source.url,
            evidence_type=EvidenceType.COMMERCIAL_PRICE,
            retrieved_at=retrieved_at,
            is_mock=False,
        )
        for source in sources
    ]
