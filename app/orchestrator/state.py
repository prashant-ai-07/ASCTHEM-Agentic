from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from app.models import (
    AgentName,
    AgentResult,
    NormalizedMedication,
    OrchestratorRequest,
    OrchestratorResponse,
    SkippedAgent,
)


class OrchestratorState(TypedDict, total=False):
    request: OrchestratorRequest
    request_id: str
    run_id: str
    started_at: str
    medication: NormalizedMedication
    requested_agents: list[AgentName]
    selected_agents: list[AgentName]
    skipped_agents: list[SkippedAgent]
    results: Annotated[list[AgentResult], operator.add]
    response: OrchestratorResponse
