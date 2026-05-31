"""
ColabConfig - Programmatic Configuration Manager for Google Colab
"""
import os
import json
from pathlib import Path

class ColabConfig:
    """Manages application state and inputs programmatically for Google Colab sessions"""
    def __init__(self, config_dict: dict = None):
        self.config_dict = config_dict or {}
        self.base_dir = Path("/content/yt-short-clipper")
        self.output_dir = Path("/content/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.base_dir / "colab_config.json"
        
        # Load and update config
        self.config = self.load()
        if config_dict:
            self.config.update(config_dict)
            self.save()
            
    def load(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default structures
        from clipper_core import AutoClipperCore
        return {
            "url": "",
            "provider": "openrouter",
            "api_key": "",
            "model": "auto",
            "whisper_provider": "openai",
            "whisper_api_key": "",
            "whisper_model": "whisper-1",
            "num_clips": 5,
            "subtitle_language": "id",
            "face_mode": "opencv",
            "add_captions": True,
            "add_hook": True,
            "system_prompt": AutoClipperCore.get_default_prompt(),
            "temperature": 1.0
        }
        
    def save(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)
            
    def validate(self):
        if not self.config.get("url"):
            raise ValueError("❌ YouTube Video URL is required!")
        if self.config.get("provider") != "none" and not self.config.get("api_key"):
            raise ValueError(f"❌ API key is required for provider '{self.config.get('provider')}'!")
            
    @property
    def url(self) -> str:
        return self.config.get("url", "")
        
    @property
    def num_clips(self) -> int:
        return int(self.config.get("num_clips", 5))
        
    @property
    def subtitle_language(self) -> str:
        return self.config.get("subtitle_language", "id")
        
    @property
    def face_mode(self) -> str:
        return self.config.get("face_mode", "opencv")
        
    @property
    def add_captions(self) -> bool:
        return bool(self.config.get("add_captions", True))
        
    @property
    def add_hook(self) -> bool:
        return bool(self.config.get("add_hook", True))
        
    @property
    def system_prompt(self) -> str:
        return self.config.get("system_prompt")
        
    @property
    def temperature(self) -> float:
        return float(self.config.get("temperature", 1.0))
        
    @property
    def cookies_path(self) -> str:
        locs = [
            self.base_dir / "cookies.txt",
            Path("cookies.txt")
        ]
        for loc in locs:
            if loc.exists():
                return str(loc)
        return None
