# 꼬꼬북 프로젝트: 벡터 데이터베이스 가이드

이 문서는 꼬꼬북 프로젝트에서 사용되는 벡터 데이터베이스의 구조, 관리 방법, 사용법을 안내합니다. 벡터 데이터베이스는 동화 이야기 데이터를 효과적으로 저장하고 검색하기 위해 사용됩니다.

## 1. 개요

꼬꼬북 벡터 DB 패키지 (`chatbot.data.vector_db`)는 이야기 데이터를 벡터 임베딩으로 변환하고, 이를 기반으로 유사도 검색을 수행할 수 있는 기능을 제공합니다. 핵심적으로 ChromaDB를 백엔드로 사용하며, 세 가지 주요 목적에 따라 데이터를 분리하여 관리할 수 있는 구조를 지원합니다:

*   **Main DB (`main`)**: 일반적인 검색 목적으로 사용됩니다. 이야기의 요약과 전체 내용을 포함하여 포괄적인 검색이 가능합니다. (기본 DB 유형)
*   **Detailed DB (`detailed`)**: 이야기의 전체 내용, 캐릭터 설명, 배경 설정 등 세부 정보 검색에 최적화되어 있습니다. 특정 장면에 대한 깊이 있는 분석이나 정보 추출에 유용합니다.
*   **Summary DB (`summary`)**: 동화의 주제, 교훈, 핵심 키워드, 짧은 요약 등 핵심 정보 검색에 최적화되어 있습니다. 이야기의 전반적인 아이디어나 특정 주제와 관련된 이야기를 빠르게 찾는 데 사용됩니다.

각 DB 유형은 고유한 디렉토리에 저장되며 (`vector_db/main`, `vector_db/detailed`, `vector_db/summary`), 필요에 따라 선택하여 사용할 수 있습니다.

## 2. 주요 모듈 및 기능

패키지는 다음과 같은 주요 모듈로 구성됩니다:

*   **`core.py`**:
    *   `VectorDB` 클래스: ChromaDB 클라이언트를 래핑하여 데이터베이스 연결, 컬렉션 생성 및 접근, 문서 추가 및 검색과 같은 기본적인 DB 작업을 수행하는 핵심 클래스입니다.
*   **`importers.py`**:
    *   `process_story_data(story_data: Dict) -> Dict`: 원본 JSON 파일에서 로드된 동화 데이터(딕셔셔리)를 받아, DB에 저장하기 적합한 표준화된 딕셔너리 형태로 전처리합니다. 이 함수는 제목, 요약, 전체 내용, 캐릭터, 설정, 주제, 교육적 가치, 키워드, 태그, 그리고 가장 중요한 `age_min`, `age_max` 등의 필드를 추출하고 정규화합니다.
*   **`query.py`**:
    *   `get_similar_stories(vector_db: VectorDB, query_text: str, n_results: int, collection_name: str, metadata_filter: Optional[Dict], doc_type: Optional[str]) -> List[Dict]`: 사용자의 텍스트 쿼리와 선택적인 메타데이터 필터를 기반으로 지정된 컬렉션에서 가장 유사한 이야기들을 검색하는 주된 인터페이스 함수입니다. RAG (Retrieval Augmented Generation) 시스템에서 관련성 높은 컨텍스트를 가져오는 데 사용됩니다.
*   **`utils.py`**:
    *   다양한 유틸리티 함수를 제공합니다:
        *   `ensure_directory()`: 디렉토리 존재 확인 및 생성.
        *   `get_db_type_path()`: DB 유형에 따른 저장 경로 생성.
        *   `load_json_file()`: JSON 파일 로드.
        *   `save_json_file()`: JSON 파일 저장.
        *   `check_collection_info()`: 특정 컬렉션의 통계 정보 (문서 수, 메타데이터 필드별 유니크 값 요약 등) 확인.

## 3. CLI 도구

두 가지 주요 CLI 도구를 통해 데이터베이스 관리 및 데이터 임포트 작업을 수행할 수 있습니다.

### 3.1. 컬렉션 관리: `manage_vector_db.py`

이 스크립트는 벡터 데이터베이스의 컬렉션을 관리(생성, 삭제, 목록 조회, 정보 확인)하는 데 사용됩니다.

**실행 방법:**

```bash
python -m chatbot.data.vector_db.manage_vector_db <action> [options]
```

**주요 `action`:**

*   `list`: 사용 가능한 DB 유형의 컬렉션 목록과 각 컬렉션의 아이템 수를 표시합니다.
*   `create`: 새 컬렉션을 생성합니다.
*   `delete`: 기존 컬렉션을 삭제합니다. (주의: 삭제 작업은 되돌릴 수 없습니다.)
*   `info`: 특정 컬렉션의 상세 정보(문서 수, 저장된 메타데이터 통계 등)를 표시합니다.

**주요 `options`:**

*   `--db-dir-type <type>`: 작업을 수행할 DB의 유형을 지정합니다. 선택 가능: `main`, `detailed`, `summary`. (기본값: `main`)
*   `--collection <name>`: 작업 대상 컬렉션의 이름을 지정합니다. (기본값: `fairy_tales`)

**예제:**

*   모든 'main' DB의 컬렉션 목록 보기:
    ```bash
    python -m chatbot.data.vector_db.manage_vector_db list --db-dir-type main
    ```
*   'detailed' DB에 'my_stories' 컬렉션 생성:
    ```bash
    python -m chatbot.data.vector_db.manage_vector_db create --db-dir-type detailed --collection my_stories
    ```
*   'summary' DB의 'fairy_tales' 컬렉션 정보 보기:
    ```bash
    python -m chatbot.data.vector_db.manage_vector_db info --db-dir-type summary --collection fairy_tales
    ```
*   'main' DB의 'temp_collection' 삭제 (확인 프롬프트 표시):
```bash
    python -m chatbot.data.vector_db.manage_vector_db delete --collection temp_collection
    ```

### 3.2. 데이터 임포트: `populate_vector_db.py`

이 스크립트는 전처리된 JSON 동화 데이터 파일들을 지정된 벡터 데이터베이스 및 컬렉션으로 가져오는(임포트하는) 데 사용됩니다. 원본 데이터는 `chatbot/data/processed/story_data/` 디렉토리에 JSON 파일 형태로 존재해야 합니다.

**실행 방법:**

```bash
python -m chatbot.data.vector_db.populate_vector_db [options]
```

**주요 `options`:**

*   `--db-dir-type <type>`: 데이터를 임포트할 DB의 유형을 지정합니다. 선택 가능: `main`, `detailed`, `summary`. (기본값: `main`)
*   `--collection <name>`: 데이터를 저장할 컬렉션의 이름을 지정합니다. (기본값: `fairy_tales`)
*   `--filter-age <age>`: 지정된 연령 태그와 관련된 스토리만 필터링하여 임포트합니다. (예: `5`는 '5-6세' 관련)
*   `--filter-theme <theme>`: 지정된 테마와 관련된 스토리만 필터링하여 임포트합니다. (예: "우정")
*   `--import-all`: 모든 스토리를 필터링 없이 기본 설정('main' DB, 'fairy_tales' 컬렉션)으로 임포트합니다. 이 옵션 사용 시 다른 필터 옵션은 무시됩니다.
*   `--verbose`: 데이터 임포트 과정에 대한 상세 로그를 출력합니다.

**예제:**

*   모든 스토리를 'main' DB의 'fairy_tales' 컬렉션으로 임포트:
    ```bash
    python -m chatbot.data.vector_db.populate_vector_db --import-all
    ```
*   'summary' DB의 'story_summaries' 컬렉션에 5세 대상 이야기만 임포트:
    ```bash
    python -m chatbot.data.vector_db.populate_vector_db --db-dir-type summary --collection story_summaries --filter-age 5
    ```
*   'detailed' DB에 "모험" 테마의 이야기만 임포트 (상세 로그 출력):
    ```bash
    python -m chatbot.data.vector_db.populate_vector_db --db-dir-type detailed --filter-theme "모험" --verbose
    ```

## 4. 데이터베이스 구조 및 주요 메타데이터

ChromaDB에 문서를 저장할 때, 텍스트 내용과 함께 다양한 메타데이터가 저장됩니다. 이 메타데이터는 검색 시 필터링 조건으로 활용되어 더 정확하고 관련성 높은 결과를 얻는 데 도움을 줍니다.

### 4.1. DB 유형별 저장 내용

*   **Main DB**:
    *   텍스트: 이야기의 요약(summary)과 전체 내용(content)을 결합한 텍스트.
    *   주요 메타데이터: `story_id`, `title`, `age_min`, `age_max`, `tags`, `keywords`, `theme`, `type: "main"`.
*   **Detailed DB**:
    *   텍스트: 이야기의 전체 내용(content). (챕터별 `narration`이 합쳐진 형태일 수 있음)
    *   주요 메타데이터: `story_id`, `title`, `age_min`, `age_max`, `characters` (등장인물 목록), `tags`, `theme`, `type: "detailed"`.
*   **Summary DB**:
    *   텍스트: 이야기의 요약(summary).
    *   주요 메타데이터: `story_id`, `title`, `age_min`, `age_max`, `keywords`, `tags`, `theme`, `type: "summary"`.

### 4.2. 주요 메타데이터 필드

*   `story_id` (str): 각 이야기를 고유하게 식별하는 ID.
*   `title` (str): 이야기의 제목.
*   `age_min` (int, Optional): 대상 독자의 최소 연령. (RAG 검색 시 연령 필터링에 사용)
*   `age_max` (int, Optional): 대상 독자의 최대 연령. (RAG 검색 시 연령 필터링에 사용)
*   `tags` (str, Optional): 이야기에 관련된 태그들 (쉼표로 구분된 문자열 또는 리스트). 예: "5-6세", "교훈적", "동물".
*   `keywords` (str, Optional): 이야기의 핵심 키워드들 (쉼표로 구분된 문자열 또는 리스트).
*   `theme` (str, Optional): 이야기의 주제. 예: "우정", "용기", "가족".
*   `characters` (str, Optional): 주요 등장인물 목록 (쉼표로 구분된 문자열, 주로 detailed DB에서 사용).
*   `educational_value` (str, Optional): 교육적 가치.
*   `type` (str): 해당 문서가 어떤 DB 유형의 컨텍스트를 위해 생성되었는지 나타냅니다 (`main`, `detailed`, `summary`). RAG 시스템이 특정 유형의 컨텍스트를 필터링하는 데 사용됩니다.

## 5. Python 코드 예제 (기본 사용법)

다음은 Python 코드 내에서 `VectorDB` 클래스와 `get_similar_stories` 함수를 사용하는 간단한 예제입니다.

```python
from chatbot.data.vector_db import VectorDB, get_similar_stories

# 1. VectorDB 인스턴스 생성 (예: 'main' DB 사용)
# 실제 경로 설정은 프로젝트 구성에 따라 달라질 수 있습니다.
db_persist_path = "chatbot/data/vector_db/main" 
vector_db_main = VectorDB(persist_directory=db_persist_path)

# 2. 컬렉션 이름 설정
collection_name = "fairy_tales" # populate_vector_db.py로 생성한 컬렉션

# 3. 유사 스토리 검색
query = "용감한 토끼 이야기"
metadata_조건 = {
    "age_min": {"$lte": 5}, # 5세 이하
    "age_max": {"$gte": 5}, # 5세 이상 (즉, 5세 대상)
    "theme": "용기"
}

# get_similar_stories는 내부적으로 query_vector_db를 호출하며, 
# doc_type은 해당 함수 내에서 metadata_filter와 결합될 수 있습니다.
# TextGenerator에서는 doc_type="summary"를 기본으로 사용할 수 있습니다.
try:
    similar_stories = get_similar_stories(
        vector_db=vector_db_main,
        query_text=query,
        n_results=3,
        collection_name=collection_name,
        metadata_filter=metadata_조건,
        doc_type="summary" # 예: 요약 문서를 대상으로 검색
    )

    if similar_stories:
        print(f"'{query}'와 유사한 이야기들 (요약 기반):")
        for story_info in similar_stories:
            print(f"  - 제목: {story_info.get('title', '제목 없음')}")
            # story_info는 검색 결과 문서의 메타데이터와 유사도 점수 등을 포함할 수 있습니다.
            # 실제 반환 형식은 get_similar_stories 함수의 구현에 따릅니다.
            # 예: story_info.get('document_content') 또는 story_info.get('metadata')
    else:
        print("유사한 이야기를 찾지 못했습니다.")

except Exception as e:
    print(f"스토리 검색 중 오류 발생: {e}")

```

이 가이드가 꼬꼬북 프로젝트의 벡터 데이터베이스를 이해하고 활용하는 데 도움이 되기를 바랍니다.