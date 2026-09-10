from __future__ import annotations

from typing import Literal, Protocol

from pydantic import Field, HttpUrl

from app.models import ApiModel, EvidenceSource, OrchestratorRequest


DdiSeverity = Literal[
    "contraindicated",
    "major",
    "moderate",
    "minor",
    "not_stated",
]


class DdiSearchRequest(OrchestratorRequest):
    """Internal DDI-agent view of the shared orchestrator request."""

    # These fields are mandatory for the public pricing workflow, but the internal
    # DDI lookup can still operate from another medication identifier.
    drug_name: str | None = None
    strength: str | None = None
    postal_code: str | None = None


class DrugInteraction(ApiModel):
    interacting_drug_or_class: str = Field(min_length=1)
    severity: DdiSeverity
    clinical_effect: str = Field(min_length=1)
    management: str | None
    label_section: str = Field(min_length=1)
    source_url: HttpUrl


class BoxedWarning(ApiModel):
    title: str = Field(min_length=1)
    risk_summary: str = Field(min_length=1)
    contraindications: list[str]
    patient_counseling: str | None
    label_section: str = Field(min_length=1)
    source_url: HttpUrl
    priority: Literal["highest"] = "highest"


class DdiSource(ApiModel):
    provider: str
    title: str
    url: HttpUrl


class DdiSearchResponse(ApiModel):
    boxed_warnings: list[BoxedWarning] = Field(default_factory=list, max_length=5)
    interactions: list[DrugInteraction] = Field(max_length=10)
    sources: list[DdiSource]


class DdiProviderResult(ApiModel):
    response: DdiSearchResponse
    evidence_sources: list[EvidenceSource]
    request_parameters: dict[str, object]


class DdiProviderUnavailable(RuntimeError):
    """Raised when official DDI evidence cannot be searched."""


class DdiProvider(Protocol):
    async def search(self, request: DdiSearchRequest) -> DdiProviderResult: ...
