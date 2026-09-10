from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_orchestrator_service
from app.models import OrchestratorRequest, OrchestratorResponse
from app.orchestrator.service import OrchestratorService


router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/api/v1/orchestrator/run",
    response_model=OrchestratorResponse,
    response_model_by_alias=True,
)
async def run_orchestrator(
    request: OrchestratorRequest,
    service: OrchestratorService = Depends(get_orchestrator_service),
) -> OrchestratorResponse:
    return await service.run(request)
