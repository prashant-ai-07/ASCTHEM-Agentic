from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import cast

from langgraph.graph import END, START, StateGraph

from app.agents.covigilance.agent import CovigilanceAgent
from app.agents.ddi.agent import DdiAgent
from app.agents.pricing.agent import PricingAgent
from app.models import (
    AgentName,
    Insight,
    NormalizationStatus,
    NormalizedMedication,
    OrchestratorResponse,
    ResponseStatus,
    ResultStatus,
)
from app.observability import RunLogEvent, RunLogger
from app.orchestrator.state import OrchestratorState


def _clean(value: str | None) -> str | None:
    return " ".join(value.split()) if value and value.strip() else None


def _normalize_ndc(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.strip().split("-")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        lengths = tuple(len(part) for part in parts)
        if lengths == (4, 4, 2):
            return f"0{parts[0]}{parts[1]}{parts[2]}"
        if lengths == (5, 3, 2):
            return f"{parts[0]}0{parts[1]}{parts[2]}"
        if lengths == (5, 4, 1):
            return f"{parts[0]}{parts[1]}0{parts[2]}"
    normalized = "".join(character for character in value if character.isdigit())
    return normalized or None


def _digits(value: str | None) -> str | None:
    normalized = "".join(character for character in value or "" if character.isdigit())
    return normalized or None


def build_orchestrator_graph(
    pricing_agent: PricingAgent,
    ddi_agent: DdiAgent,
    covigilance_agent: CovigilanceAgent,
    run_logger: RunLogger,
):
    async def normalize_medication(state: OrchestratorState) -> dict:
        started = time.perf_counter()
        medication_input = state["request"]
        status = (
            NormalizationStatus.NORMALIZED
            if all(
                (
                    medication_input.drug_name,
                    medication_input.strength,
                    medication_input.dosage_form,
                )
            )
            else NormalizationStatus.PARTIAL
        )
        medication = NormalizedMedication(
            ndc=_normalize_ndc(medication_input.ndc),
            gtin=_digits(medication_input.gtin),
            drug_name=_clean(medication_input.drug_name),
            ingredient=None,
            strength=_clean(medication_input.strength),
            dosage_form=_clean(medication_input.dosage_form),
            manufacturer=_clean(medication_input.manufacturer),
            quantity=medication_input.quantity,
            normalization_status=status,
        )
        await run_logger.log(
            RunLogEvent.create(
                run_id=state["run_id"],
                request_id=state["request_id"],
                event="node_completed",
                node="normalize_medication",
                duration_ms=int((time.perf_counter() - started) * 1000),
                details={"normalizationStatus": status.value},
            )
        )
        return {"medication": medication}

    async def route_agents(state: OrchestratorState) -> dict:
        started = time.perf_counter()
        requested = [
            AgentName.PRICING,
            AgentName.DDI,
            AgentName.PHARMACOVIGILANCE,
        ]
        selected = requested
        skipped = []
        await run_logger.log(
            RunLogEvent.create(
                run_id=state["run_id"],
                request_id=state["request_id"],
                event="node_completed",
                node="route_agents",
                duration_ms=int((time.perf_counter() - started) * 1000),
                details={
                    "requestedAgents": [agent.value for agent in requested],
                    "selectedAgents": [agent.value for agent in selected],
                    "skippedAgents": [agent.agent.value for agent in skipped],
                },
            )
        )
        return {
            "requested_agents": requested,
            "selected_agents": selected,
            "skipped_agents": skipped,
        }

    async def run_pricing_agent(state: OrchestratorState) -> dict:
        started = time.perf_counter()
        result = await pricing_agent.run(
            medication=state["medication"],
            user_id=state["request"].user_id,
            postal_code=state["request"].postal_code,
        )
        await run_logger.log(
            RunLogEvent.create(
                run_id=state["run_id"],
                request_id=state["request_id"],
                event="node_completed",
                node="pricing_agent",
                duration_ms=int((time.perf_counter() - started) * 1000),
                details={
                    "status": result.status.value,
                    "sourceCount": len(result.sources),
                },
            )
        )
        return {"results": [result]}

    async def run_ddi_agent(state: OrchestratorState) -> dict:
        started = time.perf_counter()
        result = await ddi_agent.run(
            medication=state["medication"],
            user_id=state["request"].user_id,
            postal_code=state["request"].postal_code,
        )
        await run_logger.log(
            RunLogEvent.create(
                run_id=state["run_id"],
                request_id=state["request_id"],
                event="node_completed",
                node="ddi_agent",
                duration_ms=int((time.perf_counter() - started) * 1000),
                details={
                    "status": result.status.value,
                    "sourceCount": len(result.sources),
                },
            )
        )
        return {"results": [result]}

    async def run_covigilance_agent(state: OrchestratorState) -> dict:
        started = time.perf_counter()
        result = await covigilance_agent.run(
            medication=state["medication"],
            user_id=state["request"].user_id,
            postal_code=state["request"].postal_code,
        )
        await run_logger.log(
            RunLogEvent.create(
                run_id=state["run_id"],
                request_id=state["request_id"],
                event="node_completed",
                node="covigilance_agent",
                duration_ms=int((time.perf_counter() - started) * 1000),
                details={
                    "status": result.status.value,
                    "sourceCount": len(result.sources),
                },
            )
        )
        return {"results": [result]}

    async def finalize_response(state: OrchestratorState) -> dict:
        started = time.perf_counter()
        results = state.get("results", [])
        skipped = state.get("skipped_agents", [])
        insights: list[Insight] = []

        pricing_result = next(
            (result for result in results if result.agent == AgentName.PRICING), None
        )
        if pricing_result:
            lowest_offer = pricing_result.data.get("lowestOffer")
            if isinstance(lowest_offer, dict):
                insights.append(
                    Insight(
                        text=(
                            "The lowest currently returned source-backed price is "
                            f"{lowest_offer['currency']} {lowest_offer['price']:.2f}."
                        ),
                        evidence_source_ids=[cast(str, lowest_offer["sourceId"])],
                    )
                )

        available_results = [
            result
            for result in results
            if result.status not in (ResultStatus.UNAVAILABLE, ResultStatus.ERROR)
        ]
        unavailable_results = [
            result
            for result in results
            if result.status in (ResultStatus.UNAVAILABLE, ResultStatus.ERROR)
        ]
        if available_results and (skipped or unavailable_results):
            response_status = ResponseStatus.PARTIAL
        elif available_results:
            response_status = ResponseStatus.COMPLETE
        else:
            response_status = ResponseStatus.UNAVAILABLE

        response = OrchestratorResponse(
            request_id=state["request_id"],
            run_id=state["run_id"],
            status=response_status,
            medication=state["medication"],
            executed_agents=[result.agent for result in results],
            skipped_agents=skipped,
            sections=results,
            insights=insights,
            disclaimer=(
                "AscThem presents source-backed facts for informational purposes. "
                "It does not diagnose, prescribe, or recommend starting, stopping, "
                "or changing medication."
            ),
            generated_at=datetime.now(UTC),
        )
        await run_logger.log(
            RunLogEvent.create(
                run_id=state["run_id"],
                request_id=state["request_id"],
                event="node_completed",
                node="finalize_response",
                duration_ms=int((time.perf_counter() - started) * 1000),
                details={
                    "status": response_status.value,
                    "insightCount": len(insights),
                },
            )
        )
        return {"response": response}

    builder = StateGraph(OrchestratorState)
    builder.add_node("normalize_medication", normalize_medication)
    builder.add_node("route_agents", route_agents)
    builder.add_node("pricing_agent", run_pricing_agent)
    builder.add_node("ddi_agent", run_ddi_agent)
    builder.add_node("covigilance_agent", run_covigilance_agent)
    builder.add_node("finalize_response", finalize_response)
    builder.add_edge(START, "normalize_medication")
    builder.add_edge("normalize_medication", "route_agents")
    builder.add_edge("route_agents", "pricing_agent")
    builder.add_edge("route_agents", "ddi_agent")
    builder.add_edge("route_agents", "covigilance_agent")
    builder.add_edge(
        ["pricing_agent", "ddi_agent", "covigilance_agent"],
        "finalize_response",
    )
    builder.add_edge("finalize_response", END)
    return builder.compile()
