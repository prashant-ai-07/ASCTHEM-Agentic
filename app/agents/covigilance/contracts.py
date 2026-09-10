from __future__ import annotations

from datetime import date
from typing import Literal, Protocol

from pydantic import Field, HttpUrl

from app.models import ApiModel, EvidenceSource, OrchestratorRequest


CovigilanceCategory = Literal[
    "safety_communication",
    "recall",
    "adverse_event_signal",
    "labeling_change",
]


class CovigilanceSearchRequest(OrchestratorRequest):
    """Internal Covigilance view of the shared orchestrator request."""

    drug_name: str | None = None
    strength: str | None = None
    postal_code: str | None = None


class CovigilanceFinding(ApiModel):
    category: CovigilanceCategory
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    regulatory_action: str | None
    published_date: date | None
    source_url: HttpUrl


class CovigilanceSource(ApiModel):
    provider: str
    title: str
    url: HttpUrl


class CovigilanceSearchResponse(ApiModel):
    findings: list[CovigilanceFinding] = Field(default_factory=list, max_length=12)
    sources: list[CovigilanceSource]


class CovigilanceProviderResult(ApiModel):
    response: CovigilanceSearchResponse
    evidence_sources: list[EvidenceSource]
    request_parameters: dict[str, object]


class CovigilanceProviderUnavailable(RuntimeError):
    """Raised when official post-market safety evidence cannot be searched."""


class CovigilanceProvider(Protocol):
    async def search(
        self, request: CovigilanceSearchRequest
    ) -> CovigilanceProviderResult: ...
