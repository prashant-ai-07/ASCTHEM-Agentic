from __future__ import annotations

import hashlib
import json

import pytest

from app.agents.pricing.contracts import PricingSearchRequest
from app.agents.pricing.openai_provider import OpenAIWebPricingProvider
from app.agents.pricing.openai_provider import (
    _collect_retrieved_sources,
    _strength_matches,
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


def test_collects_opened_web_page_as_exact_source() -> None:
    url = "https://www.goodrx.com/soaanz?dosage=40mg&quantity=30"
    response = {
        "output": [
            {
                "type": "web_search_call",
                "action": {"type": "open_page", "url": url},
            }
        ]
    }

    assert _collect_retrieved_sources(response) == {url: url}


@pytest.mark.asyncio
async def test_uses_web_search_strict_json_and_keeps_only_retrieved_urls() -> None:
    goodrx_url = "https://www.goodrx.com/example-drug"
    retrieved_goodrx_url = f"{goodrx_url}?location=10001"
    fabricated_url = "https://www.singlecare.com/fabricated"
    unrelated_url = "https://rxsaver.com/conditions/heart-failure"
    response = FakeResponse(
        output_text=json.dumps(
            {
                "offers": [
                    {
                        "provider": "GoodRx",
                        "pharmacy": "Example Pharmacy",
                        "price": 12.34,
                        "currency": "USD",
                        "savings_type": "coupon",
                        "coupon_code": None,
                        "source_url": goodrx_url,
                        "matched_strength": "10 mg",
                    },
                    {
                        "provider": "GoodRx",
                        "pharmacy": "Walmart",
                        "price": 10.00,
                        "currency": "USD",
                        "savings_type": "coupon",
                        "coupon_code": None,
                        "source_url": goodrx_url,
                        "matched_strength": "10 mg",
                    },
                    {
                        "provider": "SingleCare",
                        "pharmacy": "Invented Pharmacy",
                        "price": 1.23,
                        "currency": "USD",
                        "savings_type": "coupon",
                        "coupon_code": None,
                        "source_url": fabricated_url,
                        "matched_strength": "10 mg",
                    },
                    {
                        "provider": "RxSaver",
                        "pharmacy": "Unknown",
                        "price": 99.99,
                        "currency": "USD",
                        "savings_type": "unknown",
                        "coupon_code": None,
                        "source_url": unrelated_url,
                        "matched_strength": "10 mg",
                    },
                ]
            }
        ),
        sources=[
            {"url": retrieved_goodrx_url, "title": "Example Drug Prices"},
            {"url": unrelated_url, "title": "Heart Failure"},
        ],
    )
    client = FakeOpenAIClient(response)
    provider = OpenAIWebPricingProvider(
        api_key=None,
        model="test-model",
        allowed_domains=(
            "goodrx.com",
            "singlecare.com",
            "wellrx.com",
            "rxsaver.com",
            "drugs.com",
        ),
        client=client,
    )
    request = PricingSearchRequest(
        drug_name="Example Drug",
        strength="10 mg",
        quantity=30,
        user_id="private-user-id",
        postal_code="10001",
    )

    result = await provider.search(request)

    assert len(result.response.offers) == 2
    assert result.response.offers[0].pharmacy == "Walmart"
    assert str(result.response.offers[0].source_url) == retrieved_goodrx_url
    assert str(result.response.sources[0].url) == retrieved_goodrx_url
    assert result.response.sources[0].title == "Example Drug Prices"

    assert len(client.responses.calls) == 9
    searched_domains = {
        call["tools"][0]["filters"]["allowed_domains"][0]
        for call in client.responses.calls
    }
    assert searched_domains == {
        "goodrx.com",
        "singlecare.com",
        "wellrx.com",
        "rxsaver.com",
        "drugs.com",
    }
    sent = next(
        call
        for call in client.responses.calls
        if call["tools"][0]["filters"]["allowed_domains"] == ["goodrx.com"]
    )
    assert sent["tools"][0]["type"] == "web_search"
    assert sent["tools"][0]["filters"]["allowed_domains"] == ["goodrx.com"]
    assert sent["tools"][0]["search_context_size"] == "high"
    assert sent["tool_choice"] == "required"
    assert sent["max_tool_calls"] == 3
    assert sent["include"] == ["web_search_call.action.sources"]
    assert sent["text"]["format"]["type"] == "json_schema"
    assert sent["text"]["format"]["strict"] is True
    assert sent["store"] is False
    assert sent["safety_identifier"] == hashlib.sha256(
        b"private-user-id"
    ).hexdigest()
    assert json.loads(sent["input"])["userId"] == "private-user-id"
    assert "mandatory postal code 10001" in sent["instructions"]
    assert "every visible pharmacy price" in sent["instructions"]
    assert "https://www.goodrx.com/example-drug" in sent["instructions"]
    assert (
        "label_override=example-drug&dosage=10mg&location=10001&quantity=30"
        in sent["instructions"]
    )


@pytest.mark.parametrize(
    ("requested", "observed", "expected"),
    [
        ("40", "40 mg", True),
        ("40 mg", "40mg", True),
        ("60", "40 mg", False),
        ("60 mg", "40 mg", False),
        ("60", None, False),
        (None, None, True),
    ],
)
def test_strength_match_is_exact_and_rejects_missing_evidence(
    requested: str | None,
    observed: str | None,
    expected: bool,
) -> None:
    assert _strength_matches(requested, observed) is expected


@pytest.mark.asyncio
async def test_optional_fields_do_not_suppress_source_backed_soaanz_price() -> None:
    url = "https://www.goodrx.com/soaanz"
    location_url = f"{url}?location=22108"
    response = FakeResponse(
        output_text=json.dumps(
            {
                "offers": [
                    {
                        "provider": "GoodRx",
                        "pharmacy": "Walmart",
                        "price": 103.56,
                        "currency": "USD",
                        "savings_type": "coupon",
                        "coupon_code": None,
                        "source_url": url,
                        "matched_strength": "40 mg",
                    }
                ]
            }
        ),
        sources=[{"url": location_url, "title": "Soaanz Prices"}],
    )
    client = FakeOpenAIClient(response)
    provider = OpenAIWebPricingProvider(
        api_key=None,
        model="test-model",
        allowed_domains=("goodrx.com",),
        client=client,
    )
    request = PricingSearchRequest.model_validate(
        {
            "ndc": "string",
            "gtin": "string",
            "drugName": "soaanz",
            "strength": "40",
            "dosageForm": "string",
            "manufacturer": "sarfez",
            "quantity": 1,
            "userId": "2",
            "postalCode": "22108",
        }
    )

    result = await provider.search(request)

    assert [(offer.pharmacy, offer.price) for offer in result.response.offers] == [
        ("Walmart", 103.56)
    ]
    goodrx_call = next(
        call
        for call in client.responses.calls
        if call["metadata"]["pricing_source_domain"] == "goodrx.com"
    )
    assert "mandatory postal code 22108" in goodrx_call["instructions"]
    assert (
        "label_override=soaanz&dosage=40mg&location=22108"
        in goodrx_call["instructions"]
    )
    assert "quantity=1" not in goodrx_call["instructions"]
    assert len(client.responses.calls) == 2
    assert "exhaustive retry" in client.responses.calls[1]["instructions"]
    assert client.responses.calls[1]["max_tool_calls"] == 5


@pytest.mark.asyncio
async def test_rejects_offer_for_a_different_strength() -> None:
    url = "https://www.goodrx.com/soaanz"
    response = FakeResponse(
        output_text=json.dumps(
            {
                "offers": [
                    {
                        "provider": "GoodRx",
                        "pharmacy": "Walmart",
                        "price": 103.56,
                        "currency": "USD",
                        "savings_type": "coupon",
                        "coupon_code": None,
                        "source_url": url,
                        "matched_strength": "40 mg",
                    }
                ]
            }
        ),
        sources=[{"url": url, "title": "Soaanz Prices"}],
    )
    provider = OpenAIWebPricingProvider(
        api_key=None,
        model="test-model",
        allowed_domains=("goodrx.com",),
        client=FakeOpenAIClient(response),
    )

    result = await provider.search(
        PricingSearchRequest(
            drug_name="Soaanz",
            strength="60",
            postal_code="22182",
        )
    )

    assert result.response.offers == []
    assert result.response.sources == []
    assert len(provider._client.responses.calls) == 2


@pytest.mark.asyncio
async def test_rejects_offer_from_an_explicitly_different_location() -> None:
    url = "https://www.goodrx.com/soaanz"
    response = FakeResponse(
        output_text=json.dumps(
            {
                "offers": [
                    {
                        "provider": "GoodRx",
                        "pharmacy": "Walmart",
                        "price": 103.56,
                        "currency": "USD",
                        "savings_type": "coupon",
                        "coupon_code": None,
                        "source_url": url,
                        "matched_strength": "40 mg",
                    }
                ]
            }
        ),
        sources=[{"url": f"{url}?location=90210", "title": "Soaanz Prices"}],
    )
    provider = OpenAIWebPricingProvider(
        api_key=None,
        model="test-model",
        allowed_domains=("goodrx.com",),
        client=FakeOpenAIClient(response),
    )

    result = await provider.search(
        PricingSearchRequest(
            drug_name="Soaanz",
            strength="40 mg",
            postal_code="22182",
        )
    )

    assert result.response.offers == []
    assert result.response.sources == []
