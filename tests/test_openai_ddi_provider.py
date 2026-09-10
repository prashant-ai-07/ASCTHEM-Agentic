from __future__ import annotations

import hashlib
import json

import pytest

from app.agents.ddi.contracts import DdiSearchRequest
from app.agents.ddi.openai_provider import OpenAIOfficialDdiProvider


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
async def test_uses_official_sources_strict_json_and_rejects_unretrieved_urls() -> None:
    official_url = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc"
    invented_url = "https://www.fda.gov/invented-label"
    commercial_url = "https://www.drugs.com/drug-interactions/example.html"
    response = FakeResponse(
        output_text=json.dumps(
            {
                "boxed_warnings": [
                    {
                        "subject_drug": "WEGOVY (semaglutide)",
                        "title": "WARNING: RISK OF THYROID C-CELL TUMORS",
                        "risk_summary": (
                            "Semaglutide caused thyroid C-cell tumors in rodents; "
                            "whether WEGOVY causes these tumors in humans is unknown."
                        ),
                        "contraindications": [
                            "Personal or family history of medullary thyroid carcinoma",
                            "Multiple Endocrine Neoplasia syndrome type 2",
                        ],
                        "patient_counseling": (
                            "Counsel patients about the potential risk and symptoms "
                            "of thyroid tumors."
                        ),
                        "label_section": "BOXED WARNING",
                        "source_url": official_url,
                    }
                ],
                "interactions": [
                    {
                        "subject_drug": "WEGOVY (semaglutide)",
                        "interacting_drug_or_class": "Insulin secretagogues",
                        "severity": "not_stated",
                        "clinical_effect": "The label states an increased hypoglycemia risk.",
                        "management": "The label states that dose reduction may be necessary.",
                        "label_section": "7.1 Concomitant Use with an Insulin Secretagogue",
                        "source_url": official_url,
                    },
                    {
                        "subject_drug": "WEGOVY (semaglutide)",
                        "interacting_drug_or_class": "Invented Drug",
                        "severity": "major",
                        "clinical_effect": "Invented effect.",
                        "management": None,
                        "label_section": "7 DRUG INTERACTIONS",
                        "source_url": invented_url,
                    },
                    {
                        "subject_drug": "Different Product",
                        "interacting_drug_or_class": "Example Drug",
                        "severity": "moderate",
                        "clinical_effect": "Wrong subject.",
                        "management": None,
                        "label_section": "7 DRUG INTERACTIONS",
                        "source_url": official_url,
                    },
                    {
                        "subject_drug": "WEGOVY",
                        "interacting_drug_or_class": "Commercial Result",
                        "severity": "major",
                        "clinical_effect": "Commercial claim.",
                        "management": None,
                        "label_section": "Interactions",
                        "source_url": commercial_url,
                    },
                ]
            }
        ),
        sources=[
            {"url": official_url, "title": "DailyMed Wegovy Label"},
            {"url": commercial_url, "title": "Commercial Interaction Checker"},
        ],
    )
    client = FakeOpenAIClient(response)
    provider = OpenAIOfficialDdiProvider(
        api_key=None,
        model="test-model",
        client=client,
    )

    result = await provider.search(
        DdiSearchRequest(
            drug_name="wegovy",
            strength="1.5",
            quantity=1,
            user_id="private-user",
            postal_code="22108",
        )
    )

    assert len(result.response.boxed_warnings) == 1
    boxed_warning = result.response.boxed_warnings[0]
    assert boxed_warning.priority == "highest"
    assert boxed_warning.title == "WARNING: RISK OF THYROID C-CELL TUMORS"
    assert boxed_warning.label_section == "BOXED WARNING"
    assert "medullary thyroid carcinoma" in boxed_warning.contraindications[0]

    assert len(result.response.interactions) == 1
    interaction = result.response.interactions[0]
    assert interaction.interacting_drug_or_class == "Insulin secretagogues"
    assert interaction.severity == "not_stated"
    assert str(interaction.source_url) == official_url
    assert len(result.response.sources) == 1
    assert result.response.sources[0].provider == "DailyMed"

    assert len(client.responses.calls) == 4
    sent = next(
        call
        for call in client.responses.calls
        if call["tools"][0]["filters"]["allowed_domains"]
        == ["dailymed.nlm.nih.gov"]
    )
    assert "dailymed/search.cfm?query=wegovy" in sent["instructions"]
    assert sent["tool_choice"] == "required"
    assert sent["text"]["format"]["type"] == "json_schema"
    assert sent["text"]["format"]["strict"] is True
    assert sent["store"] is False
    assert sent["safety_identifier"] == hashlib.sha256(
        b"private-user"
    ).hexdigest()
    warning_call = next(
        call
        for call in client.responses.calls
        if call["metadata"]["search_focus"] == "boxed_warning"
    )
    assert warning_call["metadata"]["source_domain"] == "dailymed.nlm.nih.gov"
    assert "dedicated omission-prevention pass" in warning_call["instructions"]


@pytest.mark.asyncio
async def test_returns_empty_instead_of_inventing_interactions() -> None:
    response = FakeResponse(
        output_text=json.dumps({"boxed_warnings": [], "interactions": []}),
        sources=[],
    )
    provider = OpenAIOfficialDdiProvider(
        api_key=None,
        model="test-model",
        client=FakeOpenAIClient(response),
    )

    result = await provider.search(DdiSearchRequest(drug_name="wegovy"))

    assert result.response.interactions == []
    assert result.response.boxed_warnings == []
    assert result.response.sources == []


@pytest.mark.asyncio
async def test_any_drug_uses_the_same_dynamic_official_source_pipeline() -> None:
    source_url = (
        "https://www.accessdata.fda.gov/drugsatfda_docs/label/"
        "2026/example-label.pdf"
    )
    response = FakeResponse(
        output_text=json.dumps(
            {
                "boxed_warnings": [],
                "interactions": [
                    {
                        "subject_drug": "EXAMPLEBRAND (example ingredient)",
                        "interacting_drug_or_class": "Example Drug Class",
                        "severity": "not_stated",
                        "clinical_effect": "Effect explicitly stated by the official label.",
                        "management": None,
                        "label_section": "7 DRUG INTERACTIONS",
                        "source_url": source_url,
                    }
                ]
            }
        ),
        sources=[{"url": source_url, "title": "FDA Prescribing Information"}],
    )
    client = FakeOpenAIClient(response)
    provider = OpenAIOfficialDdiProvider(
        api_key=None,
        model="test-model",
        client=client,
    )

    result = await provider.search(DdiSearchRequest(drug_name="examplebrand"))

    assert len(result.response.interactions) == 1
    assert (
        result.response.interactions[0].interacting_drug_or_class
        == "Example Drug Class"
    )
    searched_domains = {
        call["tools"][0]["filters"]["allowed_domains"][0]
        for call in client.responses.calls
    }
    assert searched_domains == {
        "accessdata.fda.gov",
        "fda.gov",
        "dailymed.nlm.nih.gov",
    }
    dailymed_call = next(
        call
        for call in client.responses.calls
        if call["tools"][0]["filters"]["allowed_domains"]
        == ["dailymed.nlm.nih.gov"]
    )
    assert (
        "dailymed/search.cfm?query=examplebrand" in dailymed_call["instructions"]
    )
    assert 'exact product name "examplebrand"' in dailymed_call["instructions"]
