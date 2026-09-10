from __future__ import annotations

from functools import lru_cache

from app.agents.covigilance.agent import CovigilanceAgent
from app.agents.covigilance.openai_provider import (
    OpenAIOfficialCovigilanceProvider,
)
from app.agents.ddi.agent import DdiAgent
from app.agents.ddi.openai_provider import OpenAIOfficialDdiProvider
from app.agents.pricing.agent import PricingAgent
from app.agents.pricing.openai_provider import OpenAIWebPricingProvider
from app.config import Settings
from app.observability import JsonlRunLogger
from app.orchestrator.service import OrchestratorService


@lru_cache
def get_settings() -> Settings:
    return Settings.from_environment()


@lru_cache
def get_pricing_agent() -> PricingAgent:
    settings = get_settings()
    provider = OpenAIWebPricingProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        allowed_domains=settings.pricing_source_domains,
    )
    return PricingAgent(provider)


@lru_cache
def get_ddi_agent() -> DdiAgent:
    settings = get_settings()
    provider = OpenAIOfficialDdiProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_ddi_model,
        allowed_domains=settings.ddi_source_domains,
    )
    return DdiAgent(provider)


@lru_cache
def get_covigilance_agent() -> CovigilanceAgent:
    settings = get_settings()
    provider = OpenAIOfficialCovigilanceProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_covigilance_model,
        allowed_domains=settings.covigilance_source_domains,
    )
    return CovigilanceAgent(provider)


@lru_cache
def get_orchestrator_service() -> OrchestratorService:
    settings = get_settings()
    logger = JsonlRunLogger(settings.orchestrator_log_path)
    return OrchestratorService(
        get_pricing_agent(),
        get_ddi_agent(),
        get_covigilance_agent(),
        logger,
    )
