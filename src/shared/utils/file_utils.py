"""
File utility functions for the JJongal-AI project
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List


def ensure_directory(path: str) -> str:
    """Ensure directory exists, create if it doesn't"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """Save data to JSON file"""
    try:
        ensure_directory(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def list_files(directory: str, extension: str = None) -> List[str]:
    """List files in directory with optional extension filter"""
    try:
        files = os.listdir(directory)
        if extension:
            files = [f for f in files if f.endswith(extension)]
        return files
    except Exception:
        return []


def get_project_root() -> str:
    """Get project root directory"""
    current_file = Path(__file__)
    # Go up until we find the project root (where CLAUDE.md exists)
    for parent in current_file.parents:
        if (parent / "CLAUDE.md").exists():
            return str(parent)
    # Fallback to current working directory
    return os.getcwd()