from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.covigilance.agent import CovigilanceAgent
from app.agents.covigilance.contracts import (
    CovigilanceFinding,
    CovigilanceProviderResult,
    CovigilanceSearchRequest,
    CovigilanceSearchResponse,
    CovigilanceSource,
)
from app.agents.ddi.agent import DdiAgent
from app.agents.ddi.contracts import (
    BoxedWarning,
    DdiProviderResult,
    DdiSearchRequest,
    DdiSearchResponse,
    DdiSource,
    DrugInteraction,
)
from app.agents.pricing.agent import PricingAgent
from app.agents.pricing.contracts import (
    PriceOffer,
    PricingProviderResult,
    PricingSearchRequest,
    PricingSearchResponse,
    PricingSource,
)
from app.models import EvidenceSource, EvidenceType, OrchestratorRequest
from app.observability import RunLogEvent
from app.orchestrator.service import OrchestratorService


class MemoryRunLogger:
    def __init__(self) -> None:
        self.events: list[RunLogEvent] = []

    async def log(self, event: RunLogEvent) -> None:
        self.events.append(event)


class SourceBackedStubPricingProvider:
    async def search(self, request: PricingSearchRequest) -> PricingProviderResult:
        from datetime import UTC, datetime

        source_id = "pricing-source"
        source_url = "https://www.goodrx.com/example-drug"
        return PricingProviderResult(
            response=PricingSearchResponse(
                offers=[
                    PriceOffer(
                        provider="GoodRx",
                        pharmacy="Example Pharmacy",
                        price=18.50,
                        currency="USD",
                        savings_type="coupon",
                        coupon_code=None,
                        source_url=source_url,
                    )
                ],
                sources=[
                    PricingSource(
                        provider="GoodRx",
                        title="Example Drug Prices",
                        url=source_url,
                    )
                ],
            ),
            evidence_sources=[
                EvidenceSource(
                    id=source_id,
                    provider="GoodRx",
                    title="Example Drug Prices",
                    url=source_url,
                    evidence_type=EvidenceType.COMMERCIAL_PRICE,
                    retrieved_at=datetime.now(UTC),
                    is_mock=False,
                )
            ],
            request_parameters=request.openai_payload(),
        )


class SourceBackedStubDdiProvider:
    async def search(self, request: DdiSearchRequest) -> DdiProviderResult:
        from datetime import UTC, datetime

        source_url = "https://dailymed.nlm.nih.gov/dailymed/example-label"
        return DdiProviderResult(
            response=DdiSearchResponse(
                boxed_warnings=[
                    BoxedWarning(
                        title="WARNING: RISK OF THYROID C-CELL TUMORS",
                        risk_summary="The official label states the boxed risk.",
                        contraindications=["Example contraindication"],
                        patient_counseling="Counsel according to the label.",
                        label_section="BOXED WARNING",
                        source_url=source_url,
                    )
                ],
                interactions=[
                    DrugInteraction(
                        interacting_drug_or_class="Example Drug Class",
                        severity="not_stated",
                        clinical_effect="The label states an increased effect.",
                        management="Monitor according to the prescribing information.",
                        label_section="7 DRUG INTERACTIONS",
                        source_url=source_url,
                    )
                ],
                sources=[
                    DdiSource(
                        provider="DailyMed",
                        title="Example Drug Label",
                        url=source_url,
                    )
                ],
            ),
            evidence_sources=[
                EvidenceSource(
                    id="ddi-source",
                    provider="DailyMed",
                    title="Example Drug Label",
                    url=source_url,
                    evidence_type=EvidenceType.INTERACTION,
                    retrieved_at=datetime.now(UTC),
                    is_mock=False,
                )
            ],
            request_parameters=request.openai_payload(),
        )


class SourceBackedStubCovigilanceProvider:
    async def search(
        self, request: CovigilanceSearchRequest
    ) -> CovigilanceProviderResult:
        from datetime import UTC, date, datetime

        source_url = "https://www.fda.gov/example-drug-safety-communication"
        return CovigilanceProviderResult(
            response=CovigilanceSearchResponse(
                findings=[
                    CovigilanceFinding(
                        category="safety_communication",
                        title="FDA safety communication for Example Drug",
                        summary="FDA published a post-market safety update.",
                        regulatory_action="FDA required a labeling update.",
                        published_date=date(2026, 3, 20),
                        source_url=source_url,
                    )
                ],
                sources=[
                    CovigilanceSource(
                        provider="U.S. Food and Drug Administration",
                        title="FDA safety communication",
                        url=source_url,
                    )
                ],
            ),
            evidence_sources=[
                EvidenceSource(
                    id="covigilance-source",
                    provider="U.S. Food and Drug Administration",
                    title="FDA safety communication",
                    url=source_url,
                    evidence_type=EvidenceType.REGULATORY,
                    retrieved_at=datetime.now(UTC),
                    is_mock=False,
                )
            ],
            request_parameters=request.openai_payload(),
        )


def create_service() -> tuple[OrchestratorService, MemoryRunLogger]:
    logger = MemoryRunLogger()
    service = OrchestratorService(
        PricingAgent(SourceBackedStubPricingProvider()),
        DdiAgent(SourceBackedStubDdiProvider()),
        CovigilanceAgent(SourceBackedStubCovigilanceProvider()),
        logger,
    )
    return service, logger


@pytest.mark.asyncio
async def test_routes_pricing_request_to_pricing_agent() -> None:
    service, logger = create_service()
    request = OrchestratorRequest.model_validate(
        {
            "ndc": "0002-8215-01",
            "drugName": "Example Drug",
            "strength": "10 mg",
            "quantity": 30,
            "postalCode": "10001",
        }
    )

    response = await service.run(request)

    assert response.status.value == "complete"
    assert response.medication.ndc == "00002821501"
    assert {agent.value for agent in response.executed_agents} == {
        "pricing",
        "ddi",
        "pharmacovigilance",
    }
    assert response.skipped_agents == []
    pricing_section = next(
        section for section in response.sections if section.agent.value == "pricing"
    )
    assert all(not source.is_mock for source in pricing_section.sources)
    ddi_section = next(
        section for section in response.sections if section.agent.value == "ddi"
    )
    assert ddi_section.data["boxedWarnings"][0]["priority"] == "highest"
    assert ddi_section.data["boxedWarnings"][0]["labelSection"] == "BOXED WARNING"
    parameters = pricing_section.data["requestParameters"]
    assert parameters["ndc"] == "00002821501"
    assert parameters["drugName"] == "Example Drug"
    assert parameters["quantity"] == 30
    assert parameters["postalCode"] == "10001"
    assert len(response.insights[0].evidence_source_ids) == 1
    assert "run_completed" in [event.event for event in logger.events]


@pytest.mark.asyncio
async def test_single_request_runs_pricing_and_ddi_langgraph_agents() -> None:
    service, _ = create_service()
    request = OrchestratorRequest.model_validate(
        {
            "drugName": "Example Drug",
            "strength": "10 mg",
            "postalCode": "10001",
        }
    )

    response = await service.run(request)

    assert response.status.value == "complete"
    assert {agent.value for agent in response.executed_agents} == {
        "pricing",
        "ddi",
        "pharmacovigilance",
    }
    assert response.skipped_agents == []


def test_request_requires_medication_identifier() -> None:
    with pytest.raises(ValidationError):
        OrchestratorRequest.model_validate(
            {}
        )


@pytest.mark.asyncio
async def test_maps_normalized_input_to_openai_pricing_parameters() -> None:
    service, _ = create_service()
    request = OrchestratorRequest.model_validate(
        {
            "drugName": "Lipitor",
            "strength": "20 mg",
            "dosageForm": "tablet",
            "manufacturer": "Example Manufacturer",
            "quantity": 30,
            "userId": "user-1",
            "postalCode": "90401",
        }
    )

    response = await service.run(request)
    pricing_section = next(
        section for section in response.sections if section.agent.value == "pricing"
    )
    parameters = pricing_section.data["requestParameters"]

    assert parameters == {
        "ndc": None,
        "gtin": None,
        "drugName": "Lipitor",
        "strength": "20 mg",
        "dosageForm": "tablet",
        "manufacturer": "Example Manufacturer",
        "quantity": 30.0,
        "userId": "user-1",
        "postalCode": "90401",
    }
