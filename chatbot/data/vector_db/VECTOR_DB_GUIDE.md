# 꼬꼬북 벡터 데이터베이스 활용 가이드

## 벡터 데이터베이스 구조

꼬꼬북 프로젝트는 세 가지 목적별 벡터 데이터베이스를 활용합니다:

1. **main DB**: 일반적인 검색 목적으로 사용, 전체 스토리와 메타데이터 포함
   - 균형 잡힌 검색 결과 제공
   - 기본 데이터베이스로 사용

2. **detailed DB**: 스토리 전개, 캐릭터 설명, 배경 설정 등 세부 내용 검색에 최적화
   - 스토리의 구체적인 내용에 대한 검색에 활용
   - 캐릭터 정보, 장면 설명 등 세부 내용 포함

3. **summary DB**: 동화의 주제, 교훈, 키워드, 짧은 요약 등 핵심 정보 검색에 최적화
   - 핵심 주제와 교훈 등에 대한 검색에 활용
   - 간결한 요약과 키워드 중심

## 명령어 가이드

### 컬렉션 확인

각 데이터베이스에 저장된 컬렉션을 확인합니다:

```bash
# 메인 DB 컬렉션 확인
python chatbot/data/vector_db/check_collections.py --db-dir main

# 상세 DB 컬렉션 확인
python chatbot/data/vector_db/check_collections.py --db-dir detailed

# 요약 DB 컬렉션 확인
python chatbot/data/vector_db/check_collections.py --db-dir summary
```

### 데이터 추가

전처리된 스토리 데이터를 각 데이터베이스에 추가합니다:

```bash
# 메인 DB에 데이터 추가
python chatbot/data/vector_db/import_stories_with_vector_db.py --db-dir main --verbose

# 상세 DB에 데이터 추가 (연령대 필터링 예시)
python chatbot/data/vector_db/import_stories_with_vector_db.py --db-dir detailed --filter-age 5 --verbose

# 요약 DB에 데이터 추가 (테마 필터링 예시)
python chatbot/data/vector_db/import_stories_with_vector_db.py --db-dir summary --filter-theme "우주" --verbose
```

### 데이터 검색

저장된 동화 데이터를 검색합니다:

```bash
# 메인 DB 검색 (일반적인 검색)
python chatbot/data/vector_db/query_vector_db.py --query "우주 모험" --db-dir main

# 상세 DB 검색 (구체적인 내용 검색)
python chatbot/data/vector_db/query_vector_db.py --query "주인공이 로켓을 타고 화성으로 여행하는 내용" --db-dir detailed

# 요약 DB 검색 (주제나 교훈 검색)
python chatbot/data/vector_db/query_vector_db.py --query "용기와 모험심" --db-dir summary

# 태그 필터링 검색 (예: 5-6세 대상 동화)
python chatbot/data/vector_db/query_vector_db.py --query "동물" --filter-tags "5-6세" --db-dir main
```

## RAG 시스템 연계 방안

RAG(Retrieval-Augmented Generation) 시스템에서 벡터 데이터베이스를 활용하는 방법:

1. **초기 검색**: summary DB 사용
   - 아이의 관심사나 요청된 주제와 관련된 핵심 키워드와 주제 파악
   - 예: "우주에 관한 이야기 들려줘" → 우주 관련 핵심 키워드와 주제 검색

2. **세부 정보 검색**: detailed DB 사용
   - 스토리 전개에 필요한 구체적인 요소와 세부 내용 검색
   - 예: 우주선 구성 요소, 행성 설명, 캐릭터 특성 등 구체적 내용 검색

3. **균형 잡힌 참조**: main DB 사용
   - 전체적인 스토리 구조와 내용 참조
   - 예: 유사한 우주 모험 동화의 전체 구조와 전개 방식 참조

이러한 3단계 접근법을 통해 더 맥락에 맞고 풍부한 동화 생성이 가능합니다.

## 데이터베이스 관리 팁

1. **정기적인 백업**: `chroma.sqlite3` 파일과 UUID 디렉토리 정기 백업
2. **중복 데이터 관리**: 동일한 스토리는 각 DB에 한 번만 추가
3. **검증 쿼리 활용**: 데이터 추가 후 검증 쿼리로 정상 작동 확인
4. **로그 확인**: 오류 발생 시 로그 확인하여 디버깅 수행 