from __future__ import annotations

import hashlib
import json

import pytest

from app.agents.covigilance.contracts import CovigilanceSearchRequest
from app.agents.covigilance.openai_provider import (
    OpenAIOfficialCovigilanceProvider,
)


class FakeResponse:
    def __init__(self, output_text: str, sources: list[dict[str, str]]) -> None:
        self.output_text = output_text
        self._sources = sources

    def model_dump(self, mode: str = "python") -> dict:
        return {
            "output": [
                {
                    "type": "web_search_call",
                    "action": {"sources": self._sources},
                }
            ]
        }


class FakeResponsesApi:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAIClient:
    def __init__(self, response: FakeResponse) -> None:
        self.responses = FakeResponsesApi(response)


@pytest.mark.asyncio
async def test_only_accepts_retrieved_official_authority_findings() -> None:
    official_url = (
        "https://www.fda.gov/drugs/drug-safety-and-availability/"
        "fda-adds-warning-example-drug"
    )
    commercial_url = "https://example-news.test/example-drug-warning"
    response = FakeResponse(
        output_text=json.dumps(
            {
                "findings": [
                    {
                        "category": "safety_communication",
                        "subject_drug": "EXAMPLEBRAND (example ingredient)",
                        "title": "FDA adds a warning for ExampleBrand",
                        "summary": "FDA states that the labeling now includes the risk.",
                        "regulatory_action": "FDA required a labeling warning.",
                        "published_date": "2026-03-20",
                        "source_url": official_url,
                    },
                    {
                        "category": "adverse_event_signal",
                        "subject_drug": "EXAMPLEBRAND",
                        "title": "Commercially inferred signal",
                        "summary": "An unsupported claim.",
                        "regulatory_action": None,
                        "published_date": None,
                        "source_url": commercial_url,
                    },
                    {
                        "category": "recall",
                        "subject_drug": "A different medicine",
                        "title": "Wrong medicine recall",
                        "summary": "This does not concern the requested drug.",
                        "regulatory_action": "Class II",
                        "published_date": "2026-01-01",
                        "source_url": official_url,
                    },
                ]
            }
        ),
        sources=[
            {"url": official_url, "title": "FDA Drug Safety Communication"},
            {"url": commercial_url, "title": "Commercial news"},
        ],
    )
    client = FakeOpenAIClient(response)
    provider = OpenAIOfficialCovigilanceProvider(
        api_key=None,
        model="test-model",
        client=client,
    )

    result = await provider.search(
        CovigilanceSearchRequest(
            drug_name="ExampleBrand",
            user_id="private-user",
        )
    )

    assert len(result.response.findings) == 1
    finding = result.response.findings[0]
    assert finding.category == "safety_communication"
    assert finding.published_date.isoformat() == "2026-03-20"
    assert str(finding.source_url) == official_url
    assert result.response.sources[0].provider == "U.S. Food and Drug Administration"
    assert result.evidence_sources[0].evidence_type.value == "regulatory"
    assert result.evidence_sources[0].published_at is not None

    assert len(client.responses.calls) == 3
    sent = client.responses.calls[0]
    assert sent["tools"][0]["filters"]["allowed_domains"] == ["fda.gov"]
    assert sent["tool_choice"] == "required"
    assert sent["text"]["format"]["strict"] is True
    assert sent["store"] is False
    assert sent["safety_identifier"] == hashlib.sha256(
        b"private-user"
    ).hexdigest()
    assert "Do not turn raw spontaneous" in sent["instructions"]


def test_rejects_non_official_configured_domains() -> None:
    with pytest.raises(ValueError, match="recognized official authorities"):
        OpenAIOfficialCovigilanceProvider(
            api_key=None,
            model="test-model",
            allowed_domains=("fda.gov", "commercial-example.test"),
        )


@pytest.mark.asyncio
async def test_empty_official_search_does_not_invent_findings() -> None:
    response = FakeResponse(
        output_text=json.dumps({"findings": []}),
        sources=[],
    )
    provider = OpenAIOfficialCovigilanceProvider(
        api_key=None,
        model="test-model",
        client=FakeOpenAIClient(response),
    )

    result = await provider.search(CovigilanceSearchRequest(drug_name="ExampleBrand"))

    assert result.response.findings == []
    assert result.response.sources == []
