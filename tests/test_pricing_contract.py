from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.pricing.contracts import PricingSearchRequest
from app.config import _pricing_domains


def test_accepts_exact_pricing_api_inputs_and_canonicalizes_aliases() -> None:
    request = PricingSearchRequest.model_validate(
        {
            "ndc": "00002821501",
            "gtin": "00300028215018",
            "drugName": "Example Drug",
            "strength": "10 mg",
            "dosageForm": "tablet",
            "manufacturer": "Example Manufacturer",
            "quantity": 30,
            "UserId": "user-123",
            "postalcode": "10001",
        }
    )

    assert request.openai_payload() == {
        "ndc": "00002821501",
        "gtin": "00300028215018",
        "drugName": "Example Drug",
        "strength": "10 mg",
        "dosageForm": "tablet",
        "manufacturer": "Example Manufacturer",
        "quantity": 30.0,
        "userId": "user-123",
        "postalCode": "10001",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"drugName": "Lipitor", "quantity": 0},
        {"drugName": "Lipitor", "unexpected": "value"},
    ],
)
def test_rejects_invalid_pricing_requests(payload: dict) -> None:
    with pytest.raises(ValidationError):
        PricingSearchRequest.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"strength": "40", "postalCode": "22108"},
        {"drugName": "Soaanz", "postalCode": "22108"},
        {"drugName": "Soaanz", "strength": "40"},
    ],
)
def test_requires_drug_name_strength_and_postal_code_for_pricing(
    payload: dict,
) -> None:
    with pytest.raises(ValidationError):
        PricingSearchRequest.model_validate(payload)


def test_discards_swagger_string_placeholders() -> None:
    request = PricingSearchRequest.model_validate(
        {
            "ndc": "string",
            "gtin": "string",
            "drugName": "soaanz",
            "strength": "60",
            "dosageForm": "string",
            "manufacturer": "string",
            "quantity": 1,
            "userId": "string",
            "postalCode": "22182",
        }
    )

    assert request.ndc is None
    assert request.gtin is None
    assert request.dosage_form is None
    assert request.manufacturer is None
    assert request.user_id is None
    assert request.strength == "60"


def test_pricing_configuration_uses_only_uncommented_env_domains() -> None:
    assert _pricing_domains("rxsaver.com,drugs.com # disabled note") == (
        "rxsaver.com",
        "drugs.com",
    )
    assert _pricing_domains("alpha.example # off.example, also-off.example") == (
        "alpha.example",
    )
