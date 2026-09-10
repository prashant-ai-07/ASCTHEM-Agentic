from __future__ import annotations

from fastapi import FastAPI

from app.api import router


app = FastAPI(
    title="AscThem Medication Intelligence Agent Service",
    version="0.1.0",
    description=(
        "LangGraph orchestrator for source-backed medication intelligence. "
        "Pricing, DDI, and Covigilance agents use OpenAI web search and strict "
        "structured output."
    ),
)
app.include_router(router)
