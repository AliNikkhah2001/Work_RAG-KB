import json
import os
from pathlib import Path

class ArtifactManager:
    def __init__(self, commit_hash: str):
        self.commit_hash = commit_hash or "unknown_version"
        self.base_dir = Path("data/pipeline_artifacts") / self.commit_hash
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, stage: str, file_path: str, data: dict | list):
        """Save an intermediate pipeline state as JSON."""
        # Clean file path to create a safe filename
        safe_name = Path(file_path).name.replace(" ", "_")
        
        stage_dir = self.base_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = stage_dir / f"{safe_name}.json"
        
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save artifact for {file_path} at {stage}: {e}")
