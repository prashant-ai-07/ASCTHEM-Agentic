from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.dependencies import get_orchestrator_service
from app.main import app
from app.models import (
    AgentName,
    NormalizationStatus,
    NormalizedMedication,
    OrchestratorRequest,
    OrchestratorResponse,
    ResponseStatus,
)


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class StubOrchestratorService:
    request: OrchestratorRequest | None = None

    async def run(self, request: OrchestratorRequest) -> OrchestratorResponse:
        self.request = request
        return OrchestratorResponse(
            request_id="request-1",
            run_id="run-1",
            status=ResponseStatus.COMPLETE,
            medication=NormalizedMedication(
                drug_name=request.drug_name,
                strength=request.strength,
                quantity=request.quantity,
                normalization_status=NormalizationStatus.PARTIAL,
            ),
            executed_agents=[
                AgentName.PRICING,
                AgentName.DDI,
                AgentName.PHARMACOVIGILANCE,
            ],
            skipped_agents=[],
            sections=[],
            insights=[],
            disclaimer="Informational only.",
            generated_at=datetime.now(UTC),
        )


def test_single_flat_orchestrator_endpoint_accepts_pricing_input() -> None:
    service = StubOrchestratorService()
    app.dependency_overrides[get_orchestrator_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/orchestrator/run",
                json={
                    "ndc": "string",
                    "gtin": "string",
                    "drugName": "soaanz",
                    "strength": "60",
                    "dosageForm": "string",
                    "manufacturer": "string",
                    "quantity": 1,
                    "userId": "string",
                    "postalCode": "22182",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert service.request is not None
    assert service.request.drug_name == "soaanz"
    assert service.request.strength == "60"
    assert service.request.ndc is None
    assert response.json()["executedAgents"] == [
        "pricing",
        "ddi",
        "pharmacovigilance",
    ]
    assert response.json()["skippedAgents"] == []


def test_old_standalone_pricing_endpoint_is_removed() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/pricing/search",
            json={"drugName": "soaanz", "strength": "40"},
        )

    assert response.status_code == 404


def test_orchestrator_rejects_indian_postal_code() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/orchestrator/run",
            json={
                "drugName": "soaanz",
                "strength": "40 mg",
                "postalCode": "110001",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "postalCode"]
