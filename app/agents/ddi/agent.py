from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.ddi.contracts import (
    DdiProvider,
    DdiProviderResult,
    DdiProviderUnavailable,
    DdiSearchRequest,
)
from app.models import (
    AgentName,
    AgentResult,
    Confidence,
    NormalizedMedication,
    ResultStatus,
)


class _DdiState(TypedDict, total=False):
    request: DdiSearchRequest
    provider_result: DdiProviderResult


class DdiAgent:
    def __init__(self, provider: DdiProvider) -> None:
        self._provider = provider
        builder = StateGraph(_DdiState)
        builder.add_node("search_official_ddi_sources", self._search_official_sources)
        builder.add_edge(START, "search_official_ddi_sources")
        builder.add_edge("search_official_ddi_sources", END)
        self._graph = builder.compile()

    async def _search_official_sources(self, state: _DdiState) -> dict:
        return {"provider_result": await self._provider.search(state["request"])}

    async def run(
        self,
        medication: NormalizedMedication,
        user_id: str | None = None,
        postal_code: str | None = None,
    ) -> AgentResult:
        checked_at = datetime.now(UTC)
        request = DdiSearchRequest(
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

        if not request.drug_name:
            return AgentResult(
                agent=AgentName.DDI,
                status=ResultStatus.UNAVAILABLE,
                summary="A drugName is required for official-label DDI search.",
                data={
                    "boxedWarnings": [],
                    "interactions": [],
                    "requestParameters": request.openai_payload(),
                },
                sources=[],
                last_checked=checked_at,
                confidence=Confidence.NOT_ASSESSED,
                limitations=["drugName was not supplied"],
            )

        try:
            final_state = await self._graph.ainvoke(
                {"request": request},
                config={
                    "run_name": "ddi_agent",
                    "tags": ["ddi", "official-sources", "fda"],
                },
            )
            provider_result = final_state["provider_result"]
        except DdiProviderUnavailable as error:
            return AgentResult(
                agent=AgentName.DDI,
                status=ResultStatus.UNAVAILABLE,
                summary="Official-source DDI search is unavailable.",
                data={
                    "boxedWarnings": [],
                    "interactions": [],
                    "requestParameters": request.openai_payload(),
                },
                sources=[],
                last_checked=checked_at,
                confidence=Confidence.NOT_ASSESSED,
                limitations=[str(error)],
            )

        boxed_warnings = [
            warning.model_dump(mode="json", by_alias=True)
            for warning in provider_result.response.boxed_warnings
        ]
        interactions = [
            interaction.model_dump(mode="json", by_alias=True)
            for interaction in provider_result.response.interactions
        ]
        status = (
            ResultStatus.INFORMATION_FOUND
            if boxed_warnings or interactions
            else ResultStatus.NO_INFORMATION
        )
        summary = (
            "Official-source boxed warning or drug interaction information was found."
            if boxed_warnings or interactions
            else "No boxed warnings or explicitly supported drug interactions were found in searched official sources."
        )
        return AgentResult(
            agent=AgentName.DDI,
            status=status,
            summary=summary,
            data={
                "requestParameters": provider_result.request_parameters,
                "boxedWarnings": boxed_warnings,
                "interactions": interactions,
            },
            sources=provider_result.evidence_sources,
            last_checked=checked_at,
            confidence=(
                Confidence.MEDIUM
                if boxed_warnings or interactions
                else Confidence.NOT_ASSESSED
            ),
            limitations=[],
        )
