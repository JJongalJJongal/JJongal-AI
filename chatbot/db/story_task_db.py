import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any
import os

from shared.utils.logging_utils import get_module_logger

logger = get_module_logger("story_task_db") # 로깅 설정

DATABASE_NAME = "story_tasks.db" # DB 파일 이름
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # 프로젝트 루트 경로
DB_DIR = os.path.join(PROJECT_ROOT, "data") # DB 파일 저장 경로
DATABASE_PATH = os.path.join(DB_DIR, DATABASE_NAME) # DB 파일 경로


def get_db_connection():
    """Establishes a connection to the SQLite database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Initializes the database and creates the story_tasks table if it doesn't exist."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS story_tasks (
                    story_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT,
                    details_json TEXT,
                    input_outline_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_story_tasks_status ON story_tasks (status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_story_tasks_created_at ON story_tasks (created_at)")
            conn.commit()
            logger.info(f"Database initialized successfully at {DATABASE_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database {DATABASE_PATH}: {e}", exc_info=True)
        raise

def upsert_task(story_id: str, status: str, message: Optional[str], 
                input_outline: Dict[str, Any], details: Optional[Dict[str, Any]] = None):
    """Creates a new task or updates an existing one if story_id matches."""
    now_iso = datetime.utcnow().isoformat()
    details_json = json.dumps(details) if details is not None else None # ensure details can be None
    input_outline_json = json.dumps(input_outline)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO story_tasks (story_id, status, message, input_outline_json, details_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(story_id) DO UPDATE SET
                    status = excluded.status,
                    message = excluded.message,
                    input_outline_json = excluded.input_outline_json,
                    details_json = excluded.details_json,
                    updated_at = excluded.updated_at
            """, (story_id, status, message, input_outline_json, details_json, now_iso, now_iso))
            conn.commit()
            logger.info(f"Task {story_id} upserted with status: {status}")
    except sqlite3.Error as e:
        logger.error(f"Error upserting task {story_id}: {e}", exc_info=True)
        raise

def update_task_status(story_id: str, status: str, message: Optional[str], details_update: Optional[Dict[str, Any]] = None):
    """Updates the status, message, and optionally merges new details for a task."""
    now_iso = datetime.utcnow().isoformat()
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if details_update is not None:
                cursor.execute("SELECT details_json FROM story_tasks WHERE story_id = ?", (story_id,))
                row = cursor.fetchone()
                current_details = {}
                if row and row["details_json"]:
                    try:
                        current_details = json.loads(row["details_json"])
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding existing details_json for task {story_id} before update: {e}", exc_info=True)
                        # Decide on recovery: raise, use empty, or keep malformed if critical
                        # For now, let's proceed with an empty dict for current_details if decode fails
                        pass # current_details remains {}
                
                current_details.update(details_update)
                new_details_json = json.dumps(current_details)
                
                cursor.execute("""
                    UPDATE story_tasks 
                    SET status = ?, message = ?, details_json = ?, updated_at = ?
                    WHERE story_id = ?
                """, (status, message, new_details_json, now_iso, story_id))
            else:
                cursor.execute("""
                    UPDATE story_tasks 
                    SET status = ?, message = ?, updated_at = ?
                    WHERE story_id = ?
                """, (status, message, now_iso, story_id))
            
            conn.commit()
            if cursor.rowcount == 0:
                logger.warning(f"Attempted to update non-existent task {story_id}")
                return False
            logger.info(f"Task {story_id} status updated to: {status}")
            return True
    except sqlite3.Error as e:
        logger.error(f"Error updating task {story_id} status: {e}", exc_info=True)
        raise
    except json.JSONDecodeError as e: # Catch error from final json.dumps if current_details becomes un-serializable
        logger.error(f"Error encoding details_json for task {story_id} during update: {e}", exc_info=True)
        raise

def get_task(story_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a task by its story_id."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT story_id, status, message, input_outline_json, details_json, created_at, updated_at FROM story_tasks WHERE story_id = ?", (story_id,))
            row = cursor.fetchone()
            if row:
                task_data = dict(row)
                if task_data.get("details_json"):
                    try:
                        task_data["details"] = json.loads(task_data["details_json"])
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding details_json for task {story_id} on retrieval: {e}", exc_info=True)
                        task_data["details"] = {"error": "Failed to decode details_json", "raw_value": task_data["details_json"]}
                else:
                    task_data["details"] = None
                if "details_json" in task_data: del task_data["details_json"]

                if task_data.get("input_outline_json"):
                    try:
                        task_data["input_outline"] = json.loads(task_data["input_outline_json"])
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding input_outline_json for task {story_id} on retrieval: {e}", exc_info=True)
                        task_data["input_outline"] = {"error": "Failed to decode input_outline_json", "raw_value": task_data["input_outline_json"]}
                else:
                    task_data["input_outline"] = None
                if "input_outline_json" in task_data: del task_data["input_outline_json"]
                return task_data
            return None
    except sqlite3.Error as e:
        logger.error(f"Error retrieving task {story_id}: {e}", exc_info=True)
        raise

def delete_task(story_id: str) -> bool:
    """Deletes a task by its story_id. Returns True if deleted, False otherwise."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM story_tasks WHERE story_id = ?", (story_id,))
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Task {story_id} deleted successfully.")
                return True
            logger.warning(f"Attempted to delete non-existent task {story_id}.")
            return False
    except sqlite3.Error as e:
        logger.error(f"Error deleting task {story_id}: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    print(f"Database will be created/checked at: {DATABASE_PATH}")
    initialize_db()
    print("Database initialization check complete.")

    # # Example Usage (uncomment to test)
    # test_story_id = "test_story_123"
    # test_outline = {"child_name": "Test Child", "age": 5, "story_summary": "A brave little star"}
    #
    # # Create
    # print(f"Creating task {test_story_id}...")
    # upsert_task(test_story_id, "queued", "Task is queued for processing", input_outline=test_outline)
    # task = get_task(test_story_id)
    # print(f"Retrieved task: {task}")
    #
    # # Update
    # print(f"Updating task {test_story_id} status...")
    # update_task_status(test_story_id, "in_progress", "Task generation is in progress", details_update={"progress": "50%"})
    # task = get_task(test_story_id)
    # print(f"Retrieved task after update: {task}")
    #
    # # Update again, merging details
    # print(f"Updating task {test_story_id} status and merging details...")
    # update_task_status(test_story_id, "completed", "Task completed successfully", details_update={"final_output": "story_content.txt", "progress": "100%"})
    # task = get_task(test_story_id)
    # print(f"Retrieved task after second update: {task}")
    #
    # # Delete
    # # print(f"Deleting task {test_story_id}...")
    # # if delete_task(test_story_id):
    # #     print("Task deleted.")
    # # task = get_task(test_story_id)
    # # print(f"Retrieved task after delete: {task}") 