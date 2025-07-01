from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

from chatbot.data.vector_db.core import VectorDB
from shared.utils.file_utils import ensure_directory
from shared.utils.logging_utils import get_module_logger

logger = get_module_logger(__name__)

def get_db_type_path(base_directory: str, db_type: str) -> str:
    """
    Constructs the path for a specific DB type and ensures the directory exists.
    e.g., base_directory = "chatbot/data/vector_db", db_type = "main"
          -> "chatbot/data/vector_db/main"
    Args:
        base_directory: The base directory for vector databases (e.g., '.../vector_db').
        db_type: The type of the database (e.g., 'main', 'detailed', 'summary').
    Returns:
        The full path to the specific DB type directory.
    """
    db_path = Path(base_directory) / db_type
    ensure_directory(str(db_path)) # Uses shared.utils.file_utils.ensure_directory
    logger.debug(f"DB type path: {str(db_path)}")
    return str(db_path)

def check_collection_info(vector_db_instance: VectorDB, collection_name: str) -> Dict[str, Any]:
    """
    Retrieves detailed information about a specified ChromaDB collection.
    Uses the VectorDB instance to access the collection.
    Args:
        vector_db_instance: Initialized VectorDB instance.
        collection_name: Name of the collection to inspect.
    Returns:
        A dictionary containing collection information (name, count, metadata stats, etc.).
        Returns {'error': 'message'} if an error occurs.
    """
    try:
        # The get_collection method in VectorDB sets self.collection and returns it.
        collection = vector_db_instance.get_collection(collection_name)
        # No need to check if collection is None, get_collection would raise if not found.
        
        count = collection.count()
        info = {
            "name": collection.name,
            "count": count,
            "id": str(collection.id), # Convert UUID to string for broader compatibility (e.g. JSON)
            "metadata_summary": {}
        }
        
        # ChromaDB collection.metadata stores metadata provided at collection creation
        if hasattr(collection, 'metadata') and collection.metadata:
            info["collection_creation_metadata"] = collection.metadata
        else:
            info["collection_creation_metadata"] = "N/A"

        if count == 0:
            logger.info(f"컬렉션 '{collection_name}'에 항목이 없습니다.")
            info["metadata_summary"]["status"] = "항목 없음"
            return info

        # Retrieve all metadatas. This might be memory-intensive for very large collections.
        try:
            # include=['metadatas'] fetches only metadatas, which is efficient.
            collection_items = collection.get(include=["metadatas"])
            metadatas = collection_items.get('metadatas', [])
        except Exception as get_err:
            logger.error(f"컬렉션 '{collection_name}'의 모든 메타데이터를 가져오는 중 오류 발생: {get_err}", exc_info=True)
            return {"error": f"컬렉션 '{collection_name}'의 메타데이터를 가져올 수 없습니다: {str(get_err)}"}

        if not metadatas:
            logger.info(f"컬렉션 '{collection_name}'에 메타데이터가 있는 항목이 없습니다.")
            info["metadata_summary"]["status"] = "항목에 메타데이터가 없거나 비어있습니다."
            return info

        age_min_values = Counter()
        age_max_values = Counter()
        theme_values = Counter()
        type_values = Counter() # e.g., 'main', 'summary', 'detailed' from document metadata
        
        for meta_item in metadatas:
            if not isinstance(meta_item, dict): 
                logger.warning(f"Expected dict for metadata, got {type(meta_item)}. Skipping.")
                continue

            age_min = meta_item.get("age_min")
            if age_min is not None: # Allow 0 as a valid age
                try:
                    age_min_values[int(age_min)] += 1
                except ValueError:
                    logger.warning(f"Invalid age_min value '{age_min}' in metadata, skipping.")

            age_max = meta_item.get("age_max")
            if age_max is not None:
                try:
                    age_max_values[int(age_max)] += 1
                except ValueError:
                    logger.warning(f"Invalid age_max value '{age_max}' in metadata, skipping.")
            
            theme = meta_item.get("theme")
            if theme:
                theme_values[str(theme)] += 1 # Ensure theme is string

            doc_type = meta_item.get("type") # This is document type from populate_vector_db
            if doc_type:
                type_values[str(doc_type)] += 1
        
        info["metadata_summary"]["age_min_distribution"] = dict(age_min_values) if age_min_values else "데이터 없음"
        info["metadata_summary"]["age_max_distribution"] = dict(age_max_values) if age_max_values else "데이터 없음"
        info["metadata_summary"]["theme_distribution"] = dict(theme_values) if theme_values else "데이터 없음"
        info["metadata_summary"]["document_type_distribution"] = dict(type_values) if type_values else "데이터 없음"
        
        logger.info(f"컬렉션 '{collection_name}' 정보 확인 완료. 총 {count}개 항목.")
        return info

    except Exception as e:
        logger.error(f"컬렉션 '{collection_name}' 정보 확인 중 예상치 못한 오류 발생: {e}", exc_info=True)
        # Ensure the error message from the exception is captured.
        return {"error": f"컬렉션 정보 확인 중 오류: {str(e)}"} 