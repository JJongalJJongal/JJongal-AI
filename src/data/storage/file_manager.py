"""
Local File Manager for JJongal-AI

Handles local file operations, temporary file management, and storage utilities
for the story generation pipeline with integrated health monitoring.
"""

import os
import shutil
import tempfile
import json
import uuid
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from io import BytesIO
import mimetypes

from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)


class FileManager:
    """
    Comprehensive local file management system for JJongal-AI.
    
    Features:
    - Temporary file lifecycle management
    - Story asset organization
    - Safe file operations with error handling
    - Audio/image file type validation
    - Storage analytics and cleanup
    """
    
    def __init__(self, base_temp_dir: Optional[str] = None):
        """
        Initialize File Manager with configurable temporary directory
        
        Args:
            base_temp_dir: Base directory for temporary files (optional)
        """
        self.base_temp_dir = base_temp_dir or tempfile.gettempdir()
        self.jjongal_temp_dir = os.path.join(self.base_temp_dir, "jjongal_temp")
        
        # Supported file types
        self.supported_audio_formats = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
        self.supported_image_formats = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
        
        # Initialize storage directories
        self._ensure_directories()
        
        logger.info(f"FileManager initialized with temp dir: {self.jjongal_temp_dir}")
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.jjongal_temp_dir,
            os.path.join(self.jjongal_temp_dir, "audio"),
            os.path.join(self.jjongal_temp_dir, "images"),
            os.path.join(self.jjongal_temp_dir, "stories"),
            os.path.join(self.jjongal_temp_dir, "sessions")
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise
    
    def create_story_session(self, story_id: str) -> str:
        """
        Create dedicated session directory for story assets
        
        Args:
            story_id: Unique story identifier
            
        Returns:
            Path to created session directory
        """
        session_dir = os.path.join(self.jjongal_temp_dir, "sessions", story_id)
        
        try:
            os.makedirs(session_dir, exist_ok=True)
            
            # Create subdirectories for different asset types
            for asset_type in ["audio", "images", "metadata"]:
                asset_dir = os.path.join(session_dir, asset_type)
                os.makedirs(asset_dir, exist_ok=True)
            
            logger.info(f"Created story session directory: {session_dir}")
            return session_dir
            
        except Exception as e:
            logger.error(f"Failed to create story session {story_id}: {e}")
            raise
    
    def save_temporary_file(
        self,
        content: Union[bytes, str, BytesIO], 
        filename: str,
        story_id: Optional[str] = None,
        subdirectory: str = ""
    ) -> str:
        """
        Save content to temporary file with automatic type detection
        
        Args:
            content: File content (bytes, string, or BytesIO)
            filename: Target filename
            story_id: Story ID for organization (optional)
            subdirectory: Additional subdirectory (optional)
            
        Returns:
            Path to saved file
        """
        try:
            # Determine target directory
            if story_id:
                base_dir = os.path.join(self.jjongal_temp_dir, "sessions", story_id)
                if subdirectory:
                    target_dir = os.path.join(base_dir, subdirectory)
                else:
                    # Auto-detect subdirectory based on file extension
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in self.supported_audio_formats:
                        target_dir = os.path.join(base_dir, "audio")
                    elif file_ext in self.supported_image_formats:
                        target_dir = os.path.join(base_dir, "images")
                    else:
                        target_dir = os.path.join(base_dir, "metadata")
            else:
                target_dir = os.path.join(self.jjongal_temp_dir, subdirectory) if subdirectory else self.jjongal_temp_dir
            
            # Ensure target directory exists
            os.makedirs(target_dir, exist_ok=True)
            
            # Generate unique filename if needed
            target_path = os.path.join(target_dir, filename)
            if os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}{ext}"
                target_path = os.path.join(target_dir, filename)
            
            # Write content based on type
            if isinstance(content, bytes):
                with open(target_path, 'wb') as f:
                    f.write(content)
            elif isinstance(content, str):
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            elif isinstance(content, BytesIO):
                content.seek(0)
                with open(target_path, 'wb') as f:
                    f.write(content.read())
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
            
            logger.info(f"Saved temporary file: {target_path}")
            return target_path
            
        except Exception as e:
            logger.error(f"Failed to save temporary file {filename}: {e}")
            raise
    
    def load_file_content(self, file_path: str, as_bytes: bool = False) -> Union[str, bytes]:
        """
        Load file content with automatic encoding detection
        
        Args:
            file_path: Path to file to load
            as_bytes: Return content as bytes if True
            
        Returns:
            File content as string or bytes
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if as_bytes:
                with open(file_path, 'rb') as f:
                    content = f.read()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            logger.debug(f"Loaded file content: {file_path}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            raise
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive file information
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            if not os.path.exists(file_path):
                return {"exists": False, "error": "File not found"}
            
            stat = os.stat(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Determine file type
            if file_ext in self.supported_audio_formats:
                file_type = "audio"
            elif file_ext in self.supported_image_formats:
                file_type = "image"
            else:
                file_type = "other"
            
            info = {
                "exists": True,
                "path": file_path,
                "filename": os.path.basename(file_path),
                "size": stat.st_size,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": file_ext,
                "file_type": file_type,
                "mime_type": mimetypes.guess_type(file_path)[0]
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {"exists": False, "error": str(e)}
    
    def list_story_files(self, story_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all files associated with a story
        
        Args:
            story_id: Story identifier
            
        Returns:
            Dictionary categorized by file type
        """
        try:
            story_dir = os.path.join(self.jjongal_temp_dir, "sessions", story_id)
            
            if not os.path.exists(story_dir):
                return {"audio": [], "images": [], "metadata": [], "total_count": 0}
            
            categorized_files = {"audio": [], "images": [], "metadata": []}
            
            for root, dirs, files in os.walk(story_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_info = self.get_file_info(file_path)
                    
                    if file_info["exists"]:
                        # Determine category based on path or extension
                        if "audio" in root or file_info["file_type"] == "audio":
                            categorized_files["audio"].append(file_info)
                        elif "images" in root or file_info["file_type"] == "image":
                            categorized_files["images"].append(file_info)
                        else:
                            categorized_files["metadata"].append(file_info)
            
            # Add summary statistics
            total_count = sum(len(files) for files in categorized_files.values())
            categorized_files["total_count"] = total_count
            categorized_files["stats"] = {
                "audio_count": len(categorized_files["audio"]),
                "image_count": len(categorized_files["images"]),
                "metadata_count": len(categorized_files["metadata"])
            }
            
            logger.info(f"Listed {total_count} files for story {story_id}")
            return categorized_files
            
        except Exception as e:
            logger.error(f"Failed to list files for story {story_id}: {e}")
            return {"audio": [], "images": [], "metadata": [], "total_count": 0, "error": str(e)}
    
    def cleanup_story_session(self, story_id: str) -> Dict[str, Any]:
        """
        Clean up all files associated with a story session
        
        Args:
            story_id: Story identifier
            
        Returns:
            Cleanup operation results
        """
        try:
            story_dir = os.path.join(self.jjongal_temp_dir, "sessions", story_id)
            
            if not os.path.exists(story_dir):
                return {"success": True, "message": "No session directory found", "files_removed": 0}
            
            # Get file count before cleanup
            file_list = self.list_story_files(story_id)
            files_count = file_list["total_count"]
            
            # Remove directory and all contents
            shutil.rmtree(story_dir)
            
            logger.info(f"Cleaned up story session {story_id} - removed {files_count} files")
            return {
                "success": True,
                "message": f"Successfully cleaned up story session {story_id}",
                "files_removed": files_count,
                "directory_removed": story_dir
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup story session {story_id}: {e}")
            return {"success": False, "error": str(e), "files_removed": 0}
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics
        
        Returns:
            Storage usage and statistics
        """
        try:
            total_size = 0
            file_counts = {"audio": 0, "images": 0, "metadata": 0, "other": 0}
            active_sessions = 0
            
            # Walk through all temp files
            if os.path.exists(self.jjongal_temp_dir):
                for root, dirs, files in os.walk(self.jjongal_temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            file_size = os.path.getsize(file_path)
                            total_size += file_size
                            
                            # Categorize file
                            file_ext = os.path.splitext(file)[1].lower()
                            if file_ext in self.supported_audio_formats:
                                file_counts["audio"] += 1
                            elif file_ext in self.supported_image_formats:
                                file_counts["images"] += 1
                            elif file_ext in {'.json', '.txt', '.md'}:
                                file_counts["metadata"] += 1
                            else:
                                file_counts["other"] += 1
                        except OSError:
                            continue
                
                # Count active sessions
                sessions_dir = os.path.join(self.jjongal_temp_dir, "sessions")
                if os.path.exists(sessions_dir):
                    active_sessions = len([d for d in os.listdir(sessions_dir) 
                                         if os.path.isdir(os.path.join(sessions_dir, d))])
            
            stats = {
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "total_files": sum(file_counts.values()),
                "file_counts": file_counts,
                "active_sessions": active_sessions,
                "base_temp_dir": self.base_temp_dir,
                "jjongal_temp_dir": self.jjongal_temp_dir,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Storage stats: {stats['total_files']} files, {stats['total_size_mb']} MB")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e), "total_size_mb": 0, "total_files": 0}
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Clean up files older than specified age
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Cleanup operation results
        """
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            removed_files = []
            total_size_freed = 0
            
            if os.path.exists(self.jjongal_temp_dir):
                for root, dirs, files in os.walk(self.jjongal_temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            file_stat = os.stat(file_path)
                            if file_stat.st_mtime < cutoff_time:
                                file_size = file_stat.st_size
                                os.remove(file_path)
                                removed_files.append({
                                    "path": file_path,
                                    "size": file_size,
                                    "age_hours": round((datetime.now().timestamp() - file_stat.st_mtime) / 3600, 1)
                                })
                                total_size_freed += file_size
                        except OSError:
                            continue
                
                # Remove empty directories
                for root, dirs, files in os.walk(self.jjongal_temp_dir, topdown=False):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            if not os.listdir(dir_path):  # Empty directory
                                os.rmdir(dir_path)
                        except OSError:
                            continue
            
            result = {
                "success": True,
                "files_removed": len(removed_files),
                "size_freed_mb": round(total_size_freed / 1024 / 1024, 2),
                "max_age_hours": max_age_hours,
                "removed_files": removed_files[:10]  # Limit list for logging
            }
            
            logger.info(f"Cleanup completed: {len(removed_files)} files removed, {result['size_freed_mb']} MB freed")
            return result
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
            return {"success": False, "error": str(e), "files_removed": 0, "size_freed_mb": 0}
    
    def move_file(self, source_path: str, destination_path: str, create_dirs: bool = True) -> bool:
        """
        Move file to new location with directory creation option
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            create_dirs: Create destination directories if needed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(source_path):
                logger.error(f"Source file not found: {source_path}")
                return False
            
            if create_dirs:
                destination_dir = os.path.dirname(destination_path)
                os.makedirs(destination_dir, exist_ok=True)
            
            shutil.move(source_path, destination_path)
            logger.info(f"Moved file: {source_path} -> {destination_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move file {source_path} -> {destination_path}: {e}")
            return False
    
    def copy_file(self, source_path: str, destination_path: str, create_dirs: bool = True) -> bool:
        """
        Copy file to new location with directory creation option
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            create_dirs: Create destination directories if needed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(source_path):
                logger.error(f"Source file not found: {source_path}")
                return False
            
            if create_dirs:
                destination_dir = os.path.dirname(destination_path)
                os.makedirs(destination_dir, exist_ok=True)
            
            shutil.copy2(source_path, destination_path)
            logger.info(f"Copied file: {source_path} -> {destination_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy file {source_path} -> {destination_path}: {e}")
            return False
    
    def is_healthy(self) -> bool:
        """
        Check if FileManager is healthy and operational
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if temp directory is accessible and writable
            test_file = os.path.join(self.jjongal_temp_dir, f"health_check_{uuid.uuid4().hex[:8]}.tmp")
            
            # Write test
            with open(test_file, 'w') as f:
                f.write("health_check")
            
            # Read test
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Cleanup test file
            os.remove(test_file)
            
            return content == "health_check"
            
        except Exception as e:
            logger.error(f"FileManager health check failed: {e}")
            return False


# Global instance
_file_manager = None

def get_file_manager() -> FileManager:
    """Get global FileManager instance"""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager