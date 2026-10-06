import json
import os
from pathlib import Path
from pydantic import BaseModel

SETTINGS_FILE = Path(__file__).resolve().parent.parent / "data" / "search_settings.json"

class SearchSettings(BaseModel):
    top_k: int = 10
    min_relevance_score: float = 0.05
    keyword_boost: float = 3.0
    fusion_alpha: float = 0.7
    synonym_enabled: bool = True
    synonym_beam: int = 5
    rerank_pool: int = 100
    use_splade_fallback: bool = False

def load_settings() -> SearchSettings:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return SearchSettings.model_validate_json(f.read())
        except Exception as e:
            print("Failed to load settings:", e)
    return SearchSettings()

def save_settings(settings: SearchSettings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        f.write(settings.model_dump_json(indent=2))
