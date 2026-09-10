from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.covigilance.contracts import (
    CovigilanceProvider,
    CovigilanceProviderResult,
    CovigilanceProviderUnavailable,
    CovigilanceSearchRequest,
)
from app.models import (
    AgentName,
    AgentResult,
    Confidence,
    NormalizedMedication,
    ResultStatus,
)


class _CovigilanceState(TypedDict, total=False):
    request: CovigilanceSearchRequest
    provider_result: CovigilanceProviderResult


class CovigilanceAgent:
    def __init__(self, provider: CovigilanceProvider) -> None:
        self._provider = provider
        builder = StateGraph(_CovigilanceState)
        builder.add_node("search_official_safety_sources", self._search_sources)
        builder.add_edge(START, "search_official_safety_sources")
        builder.add_edge("search_official_safety_sources", END)
        self._graph = builder.compile()

    async def _search_sources(self, state: _CovigilanceState) -> dict:
        return {"provider_result": await self._provider.search(state["request"])}

    async def run(
        self,
        medication: NormalizedMedication,
        user_id: str | None = None,
        postal_code: str | None = None,
    ) -> AgentResult:
        checked_at = datetime.now(UTC)
        request = CovigilanceSearchRequest(
            ndc=medication.ndc,
            gtin=medication.gtin,
            drug_name=medication.drug_name,
            strength=medication.strength,
            dosage_form=medication.dosage_form,
            manufacturer=medication.manufacturer,
            quantity=medication.quantity,
            user_id=user_id,
            postal_code=postal_code,
        )
        empty_data = {
            "requestParameters": request.openai_payload(),
            "findings": [],
            "countsByCategory": {},
        }
        if not request.drug_name:
            return AgentResult(
                agent=AgentName.PHARMACOVIGILANCE,
                status=ResultStatus.UNAVAILABLE,
                summary="A drugName is required for official Covigilance search.",
                data=empty_data,
                sources=[],
                last_checked=checked_at,
                confidence=Confidence.NOT_ASSESSED,
                limitations=["drugName was not supplied"],
            )

        try:
            final_state = await self._graph.ainvoke(
                {"request": request},
                config={
                    "run_name": "covigilance_agent",
                    "tags": ["covigilance", "official-sources", "post-market-safety"],
                },
            )
            provider_result = final_state["provider_result"]
        except CovigilanceProviderUnavailable as error:
            return AgentResult(
                agent=AgentName.PHARMACOVIGILANCE,
                status=ResultStatus.UNAVAILABLE,
                summary="Official-source Covigilance search is unavailable.",
                data=empty_data,
                sources=[],
                last_checked=checked_at,
                confidence=Confidence.NOT_ASSESSED,
                limitations=[str(error)],
            )

        findings = [
            finding.model_dump(mode="json", by_alias=True)
            for finding in provider_result.response.findings
        ]
        counts = Counter(
            finding.category for finding in provider_result.response.findings
        )
        limitations = [
            "Absence of a returned finding does not establish absence of risk.",
            (
                "Regulator-identified adverse-event signals do not by themselves "
                "establish that a medicine caused the reported event."
            ),
        ]
        found = bool(findings)
        return AgentResult(
            agent=AgentName.PHARMACOVIGILANCE,
            status=(
                ResultStatus.INFORMATION_FOUND
                if found
                else ResultStatus.NO_INFORMATION
            ),
            summary=(
                "Official post-market drug-safety findings were found."
                if found
                else "No matching post-market findings were found in the searched official sources."
            ),
            data={
                "requestParameters": provider_result.request_parameters,
                "findings": findings,
                "countsByCategory": dict(counts),
            },
            sources=provider_result.evidence_sources,
            last_checked=checked_at,
            confidence=Confidence.MEDIUM if found else Confidence.NOT_ASSESSED,
            limitations=limitations,
        )
