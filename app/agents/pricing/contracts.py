from __future__ import annotations

from typing import Protocol

from pydantic import Field, HttpUrl

from app.models import ApiModel, EvidenceSource, OrchestratorRequest


class PricingSearchRequest(OrchestratorRequest):
    """Internal pricing-agent view of the shared orchestrator request."""


class PricingSource(ApiModel):
    provider: str
    title: str
    url: HttpUrl


class PriceOffer(ApiModel):
    provider: str
    pharmacy: str
    price: float = Field(ge=0)
    currency: str = "USD"
    savings_type: str
    coupon_code: str | None
    source_url: HttpUrl


class PricingSearchResponse(ApiModel):
    offers: list[PriceOffer]
    sources: list[PricingSource]


class PricingProviderResult(ApiModel):
    response: PricingSearchResponse
    evidence_sources: list[EvidenceSource]
    request_parameters: dict[str, object]


class PricingProviderUnavailable(RuntimeError):
    """Raised when the live pricing provider cannot be called."""


class PricingProvider(Protocol):
    async def search(self, request: PricingSearchRequest) -> PricingProviderResult: ...
