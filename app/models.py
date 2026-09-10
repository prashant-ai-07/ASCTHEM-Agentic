from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)


def to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(word.capitalize() for word in rest)


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class AgentName(StrEnum):
    PRICING = "pricing"
    DDI = "ddi"
    PHARMACOVIGILANCE = "pharmacovigilance"


class OrchestratorRequest(ApiModel):
    """One public request contract shared by all orchestrated agents."""

    ndc: str | None = Field(default=None, min_length=1)
    gtin: str | None = Field(default=None, min_length=1)
    drug_name: str = Field(min_length=1)
    strength: str = Field(min_length=1)
    dosage_form: str | None = Field(default=None, min_length=1)
    manufacturer: str | None = Field(default=None, min_length=1)
    quantity: float | None = Field(default=None, gt=0)
    user_id: str | None = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("userId", "UserId", "user_id"),
        serialization_alias="userId",
    )
    postal_code: str = Field(
        min_length=1,
        validation_alias=AliasChoices("postalCode", "postalcode", "postal_code"),
        serialization_alias="postalCode",
    )

    @model_validator(mode="before")
    @classmethod
    def discard_swagger_placeholders(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        # Swagger's example value "string" is not medication data. Treat it as
        # an omitted optional value so it cannot contaminate search terms.
        return {
            key: None
            if isinstance(item, str) and item.strip().lower() == "string"
            else item
            for key, item in value.items()
        }

    def openai_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", by_alias=True)


class NormalizationStatus(StrEnum):
    PARTIAL = "partial"
    NORMALIZED = "normalized"


class NormalizedMedication(ApiModel):
    ndc: str | None = None
    gtin: str | None = None
    drug_name: str | None = None
    ingredient: str | None = None
    strength: str | None = None
    dosage_form: str | None = None
    manufacturer: str | None = None
    quantity: float | None = None
    normalization_status: NormalizationStatus


class EvidenceType(StrEnum):
    COMMERCIAL_PRICE = "commercial_price"
    COUPON = "coupon"
    CONSUMER_REVIEW = "consumer_review"
    REGULATORY = "regulatory"
    INTERACTION = "interaction"


class EvidenceSource(ApiModel):
    id: str
    provider: str
    title: str
    url: HttpUrl | None = None
    evidence_type: EvidenceType
    retrieved_at: datetime
    published_at: datetime | None = None
    is_mock: bool = False


class ResultStatus(StrEnum):
    INFORMATION_FOUND = "information_found"
    NO_INFORMATION = "no_information"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOT_ASSESSED = "not_assessed"


class AgentResult(ApiModel):
    agent: AgentName
    status: ResultStatus
    summary: str
    data: dict[str, Any]
    sources: list[EvidenceSource]
    last_checked: datetime
    confidence: Confidence
    limitations: list[str]


class SkipReason(StrEnum):
    NOT_IMPLEMENTED = "not_implemented"
    MISSING_CONTEXT = "missing_context"


class SkippedAgent(ApiModel):
    agent: AgentName
    reason: SkipReason
    detail: str


class Insight(ApiModel):
    text: str
    evidence_source_ids: list[str]


class ResponseStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class OrchestratorResponse(ApiModel):
    request_id: str
    run_id: str
    status: ResponseStatus
    medication: NormalizedMedication
    executed_agents: list[AgentName]
    skipped_agents: list[SkippedAgent]
    sections: list[AgentResult]
    insights: list[Insight]
    disclaimer: str
    generated_at: datetime
