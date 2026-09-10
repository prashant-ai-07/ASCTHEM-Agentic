from __future__ import annotations

import time
from datetime import UTC, datetime
from uuid import uuid4

from app.agents.covigilance.agent import CovigilanceAgent
from app.agents.ddi.agent import DdiAgent
from app.agents.pricing.agent import PricingAgent
from app.models import OrchestratorRequest, OrchestratorResponse
from app.observability import RunLogEvent, RunLogger
from app.orchestrator.graph import build_orchestrator_graph
from app.orchestrator.state import OrchestratorState


class OrchestratorService:
    def __init__(
        self,
        pricing_agent: PricingAgent,
        ddi_agent: DdiAgent,
        covigilance_agent: CovigilanceAgent,
        run_logger: RunLogger,
    ) -> None:
        self._run_logger = run_logger
        self._graph = build_orchestrator_graph(
            pricing_agent,
            ddi_agent,
            covigilance_agent,
            run_logger,
        )

    async def run(self, request: OrchestratorRequest) -> OrchestratorResponse:
        request_id = str(uuid4())
        run_id = str(uuid4())
        started = time.perf_counter()
        await self._run_logger.log(
            RunLogEvent.create(
                run_id=run_id,
                request_id=request_id,
                event="run_started",
                details={
                    "requestedAgents": [
                        "pricing",
                        "ddi",
                        "pharmacovigilance",
                    ]
                },
            )
        )

        initial_state: OrchestratorState = {
            "request": request,
            "request_id": request_id,
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "requested_agents": [],
            "selected_agents": [],
            "skipped_agents": [],
            "results": [],
        }
        try:
            final_state = await self._graph.ainvoke(
                initial_state,
                config={
                    "run_name": "ascthem_orchestrator",
                    "tags": ["orchestrator", "all-agents"],
                    "metadata": {"request_id": request_id, "run_id": run_id},
                },
            )
            response = final_state["response"]
            await self._run_logger.log(
                RunLogEvent.create(
                    run_id=run_id,
                    request_id=request_id,
                    event="run_completed",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    details={"status": response.status.value},
                )
            )
            return response
        except Exception as error:
            await self._run_logger.log(
                RunLogEvent.create(
                    run_id=run_id,
                    request_id=request_id,
                    event="run_failed",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    details={"error": str(error)},
                )
            )
            raise
