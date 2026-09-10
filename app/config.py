from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _pricing_domains(value: str) -> tuple[str, ...]:
    # Strip the entire inline comment before splitting. Otherwise a value such as
    # "a.com,b.com #c.com, d.com" accidentally enables d.com.
    configured_value = value.split("#", 1)[0]
    return tuple(
        dict.fromkeys(
            domain.strip().lower()
            for domain in configured_value.split(",")
            if domain.strip()
        )
    )


@dataclass(frozen=True, slots=True)
class Settings:
    port: int
    orchestrator_log_path: Path
    openai_api_key: str | None
    openai_model: str
    openai_ddi_model: str
    openai_covigilance_model: str
    pricing_source_domains: tuple[str, ...]
    ddi_source_domains: tuple[str, ...]
    covigilance_source_domains: tuple[str, ...]

    @classmethod
    def from_environment(cls) -> "Settings":
        try:
            port = int(os.getenv("PORT", "3000"))
        except ValueError:
            port = 3000

        return cls(
            port=port,
            orchestrator_log_path=Path(
                os.getenv("ORCHESTRATOR_LOG_PATH", "logs/orchestrator.jsonl")
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_PRICING_MODEL", "gpt-5.4-mini"),
            openai_ddi_model=os.getenv(
                "OPENAI_DDI_MODEL",
                os.getenv("OPENAI_PRICING_MODEL", "gpt-5.4-mini"),
            ),
            openai_covigilance_model=os.getenv(
                "OPENAI_COVIGILANCE_MODEL",
                os.getenv("OPENAI_PRICING_MODEL", "gpt-5.4-mini"),
            ),
            pricing_source_domains=_pricing_domains(
                os.getenv("PRICING_SOURCE_DOMAINS", "")
            ),
            ddi_source_domains=tuple(
                domain.strip().lower()
                for domain in os.getenv(
                    "DDI_SOURCE_DOMAINS",
                    "accessdata.fda.gov,fda.gov,dailymed.nlm.nih.gov",
                ).split(",")
                if domain.strip()
            ),
            covigilance_source_domains=_pricing_domains(
                os.getenv(
                    "COVIGILANCE_SOURCE_DOMAINS",
                    "fda.gov,accessdata.fda.gov,dailymed.nlm.nih.gov",
                )
            ),
        )
