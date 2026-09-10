from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.pricing.contracts import (
    PricingProvider,
    PricingProviderResult,
    PricingProviderUnavailable,
    PricingSearchRequest,
    PricingSearchResponse,
)
from app.models import (
    AgentName,
    AgentResult,
    Confidence,
    NormalizedMedication,
    ResultStatus,
)


class _PricingState(TypedDict, total=False):
    request: PricingSearchRequest
    provider_result: PricingProviderResult


class PricingAgent:
    def __init__(self, provider: PricingProvider) -> None:
        self._provider = provider
        builder = StateGraph(_PricingState)
        builder.add_node("search_pricing_web", self._search_pricing_web)
        builder.add_edge(START, "search_pricing_web")
        builder.add_edge("search_pricing_web", END)
        self._graph = builder.compile()

    async def _search_pricing_web(self, state: _PricingState) -> dict:
        return {"provider_result": await self._provider.search(state["request"])}

    async def search(self, request: PricingSearchRequest) -> PricingSearchResponse:
        final_state = await self._graph.ainvoke(
            {"request": request},
            config={"run_name": "pricing_agent", "tags": ["pricing", "web-search"]},
        )
        return final_state["provider_result"].response

    async def run(
        self,
        medication: NormalizedMedication,
        user_id: str | None = None,
        postal_code: str | None = None,
    ) -> AgentResult:
        last_checked = datetime.now(UTC)
        request = PricingSearchRequest(
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

        try:
            final_state = await self._graph.ainvoke(
                {"request": request},
                config={"run_name": "pricing_agent", "tags": ["pricing", "web-search"]},
            )
            provider_result = final_state["provider_result"]
        except PricingProviderUnavailable as error:
            return AgentResult(
                agent=AgentName.PRICING,
                status=ResultStatus.UNAVAILABLE,
                summary="Live web pricing is unavailable.",
                data={"offers": [], "requestParameters": request.openai_payload()},
                sources=[],
                last_checked=last_checked,
                confidence=Confidence.NOT_ASSESSED,
                limitations=[str(error)],
            )

        response = provider_result.response
        if not response.offers:
            return AgentResult(
                agent=AgentName.PRICING,
                status=ResultStatus.NO_INFORMATION,
                summary="No source-backed pricing information was found.",
                data={
                    "offers": [],
                    "requestParameters": provider_result.request_parameters,
                },
                sources=provider_result.evidence_sources,
                last_checked=last_checked,
                confidence=Confidence.NOT_ASSESSED,
                limitations=[],
            )

        source_ids = {
            str(source.url): source.id for source in provider_result.evidence_sources
        }
        serialized_offers = []
        for offer in response.offers:
            value = offer.model_dump(mode="json", by_alias=True)
            value["sourceId"] = source_ids[str(offer.source_url)]
            serialized_offers.append(value)

        return AgentResult(
            agent=AgentName.PRICING,
            status=ResultStatus.INFORMATION_FOUND,
            summary="Source-backed web pricing information was found.",
            data={
                "requestParameters": provider_result.request_parameters,
                "offers": serialized_offers,
                "lowestOffer": serialized_offers[0],
            },
            sources=provider_result.evidence_sources,
            last_checked=last_checked,
            confidence=Confidence.MEDIUM,
            limitations=[],
        )
