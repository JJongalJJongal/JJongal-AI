"""
Data Storage Module for JJongal-AI

File management and storage utilities for handling temporary files,
story assets, and S3 integration for the storytelling pipeline.
"""

from .file_manager import FileManager, get_file_manager
from .s3_manager import S3Manager

__all__ = [
    "FileManager", 
    "get_file_manager",
    "S3Manager"
]

__version__ = "1.0.0"