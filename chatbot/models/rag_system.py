import os
import json
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
# langchain-chroma 패키지에서 직접 Chroma 임포트
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import tiktoken
from dotenv import load_dotenv
from pathlib import Path

# 환경 변수 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Chroma 초기화 실패 시 사용할 대체 클래스
class DummyVectorStore:
    """벡터 저장소 초기화 실패 시 사용할 대체 클래스"""
    
    def __init__(self):
        self.items = []
        print("DummyVectorStore 초기화됨 (테스트 모드)")
    
    def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
        """텍스트 추가 (더미 구현)"""
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            item_id = ids[i] if ids else f"item_{len(self.items)}"
            self.items.append({"id": item_id, "text": text, "metadata": meta})
        return ids or [f"item_{i}" for i in range(len(texts))]
    
    def similarity_search_with_score(self, query, k=3, filter=None, **kwargs):
        """유사성 검색 (더미 구현)"""
        results = []
        for _ in range(min(k, len(self.items) or 1)):
            doc = Document(page_content="샘플 콘텐츠", metadata={"title": "샘플 제목", "tags": "5-6세", "story_id": "dummy_story"})
            results.append((doc, 0.5))
        return results
    
    def as_retriever(self, search_kwargs=None, **kwargs):
        """검색기로 사용 (더미 구현)"""
        return DummyRetriever(self)
    
    def get(self, **kwargs):
        """저장소 데이터 가져오기 (더미 구현)"""
        return {"ids": [item["id"] for item in self.items]}
    
    def persist(self):
        """저장소 저장 (더미 구현)"""
        print("테스트 모드: 저장 작업 무시됨")
        pass

class DummyRetriever:
    """더미 검색기 구현"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def get_relevant_documents(self, query):
        """관련 문서 검색 (더미 구현)"""
        return [Document(page_content="샘플 콘텐츠", metadata={"title": "샘플 제목", "tags": "5-6세", "story_id": "dummy_story"})]

class RAGSystem:
    """
    LangChain과 Chroma DB 기반 RAG (Retrieval-Augmented Generation) 시스템
    
    동화 창작 과정에서 기존 동화 지식을 활용하여 창의적이고 일관된 스토리를 생성하도록 지원합니다.
    
    Attributes:
        embeddings: OpenAI 임베딩 모델
        summary_vectorstore: 요약 벡터 저장소
        detailed_vectorstore: 상세 내용 벡터 저장소
        text_splitter: 텍스트 분할기
        llm: 언어 모델 (ChatOpenAI)
    """
    
    def __init__(self, persist_directory: str = "data/vector_db"):
        """
        RAG 시스템 초기화
        
        Args:
            persist_directory: 벡터 데이터베이스 저장 디렉토리
        """
        # OpenAI 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings()
        
        # 벡터 저장소 디렉토리 설정
        self.persist_directory = persist_directory
        self.summary_persist_dir = os.path.join(persist_directory, "summary")
        self.detailed_persist_dir = os.path.join(persist_directory, "detailed")
        
        # 디렉토리 생성
        os.makedirs(self.summary_persist_dir, exist_ok=True)
        os.makedirs(self.detailed_persist_dir, exist_ok=True)
        
        # 벡터 저장소 초기화
        try:
            self.summary_vectorstore = Chroma(
                persist_directory=self.summary_persist_dir,
                embedding_function=self.embeddings
            )
            
            self.detailed_vectorstore = Chroma(
                persist_directory=self.detailed_persist_dir,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"Chroma 초기화 오류: {str(e)}")
            # 임시 해결책: 테스트 모드로 작동하도록 더미 메서드 추가
            self.summary_vectorstore = DummyVectorStore()
            self.detailed_vectorstore = DummyVectorStore()
        
        # 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # LLM 초기화
        self.llm = ChatOpenAI(model_name="gpt-4o")
    
    def add_story(self, title: str, tags: str, summary: str, content: str, story_id: Optional[str] = None) -> str:
        """
        새로운 동화를 RAG 시스템에 추가
        
        Args:
            title: 동화 제목
            tags: 태그 (쉼표로 구분)
            summary: 동화 요약
            content: 동화 전체 내용
            story_id: 스토리 ID (없으면 자동 생성)
            
        Returns:
            str: 스토리 ID
        """
        # 스토리 ID 생성 (없는 경우)
        if not story_id:
            story_id = f"story_{len(self.summary_vectorstore.get()['ids']) + 1}"
        
        # 요약 메타데이터 설정
        summary_metadata = {
            "story_id": story_id,
            "title": title,
            "tags": tags,
            "type": "summary"
        }
        
        # 요약을 벡터 저장소에 추가
        self.summary_vectorstore.add_texts(
            texts=[summary],
            metadatas=[summary_metadata],
            ids=[f"{story_id}_summary"]
        )
        
        # 상세 내용 분할
        content_docs = self.text_splitter.create_documents(
            texts=[content],
            metadatas=[{
                "story_id": story_id,
                "title": title,
                "tags": tags,
                "type": "detailed"
            }]
        )
        
        # 상세 내용을 벡터 저장소에 추가
        texts = [doc.page_content for doc in content_docs]
        metadatas = [doc.metadata for doc in content_docs]
        ids = [f"{story_id}_detailed_{i}" for i in range(len(texts))]
        
        self.detailed_vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # 벡터 저장소 저장
        self.summary_vectorstore.persist()
        self.detailed_vectorstore.persist()
        
        return story_id
    
    def query(self, query: str, use_summary: bool = False, age_group: Optional[int] = None, k: int = 3) -> Dict[str, Any]:
        """
        RAG를 사용하여 질의에 응답
        
        Args:
            query: 질의 텍스트
            use_summary: 요약 벡터 저장소 사용 여부
            age_group: 연령대 필터링 (숫자, 예: 5는 5-6세)
            k: 검색할 문서 수
            
        Returns:
            Dict: {"answer": 응답 텍스트, "sources": 참조 문서들}
        """
        try:
            # 연령대 필터링 설정
            filter_dict = {}
            if age_group:
                age_tag = f"{age_group}-{age_group+1}세"
                filter_dict = {"tags": {"$contains": age_tag}}
            
            # 검색에 사용할 벡터 저장소 선택
            vectorstore = self.summary_vectorstore if use_summary else self.detailed_vectorstore
            
            # 검색 실행
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": k, "filter": filter_dict if filter_dict else None}
            )
            
            # 프롬프트 템플릿 설정
            template = """
            당신은 동화 전문가입니다. 주어진 정보를 바탕으로 질문에 답변해주세요.
            
            정보:
            {context}
            
            질문: {question}
            
            답변:
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            try:
                # QA 체인 생성 및 실행
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt}
                )
                
                result = qa_chain({"query": query})
                
                return {
                    "answer": result["result"],
                    "sources": [doc.metadata for doc in result.get("source_documents", [])]
                }
            except Exception as chain_error:
                print(f"QA 체인 오류: {str(chain_error)}")
                # 직접 LLM 호출로 대체
                response = self.llm.predict(
                    f"다음 질문에 동화 전문가로서 답변해주세요: {query}"
                )
                return {
                    "answer": response,
                    "sources": []
                }
                
        except Exception as e:
            print(f"쿼리 처리 중 오류 발생: {str(e)}")
            return {
                "answer": f"죄송합니다, 질문을 처리하는 중에 오류가 발생했습니다. 다른 질문을 해주세요.",
                "sources": []
            }
    
    def get_similar_stories(self, theme: str, age_group: Optional[int] = None, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        주제와 유사한 동화 검색
        
        Args:
            theme: 동화 주제
            age_group: 연령대 (숫자, 예: 5는 5-6세)
            n_results: 결과 수
            
        Returns:
            List[Dict]: 유사한 동화 목록
        """
        # 연령대 필터링 설정
        filter_dict = {}
        if age_group:
            age_tag = f"{age_group}-{age_group+1}세"
            filter_dict = {"tags": {"$contains": age_tag}}
        
        # 요약 벡터 저장소에서 검색
        results = self.summary_vectorstore.similarity_search_with_score(
            query=theme,
            k=n_results,
            filter=filter_dict if filter_dict else None
        )
        
        # 결과 가공
        similar_stories = []
        for doc, score in results:
            similar_stories.append({
                "title": doc.metadata.get("title"),
                "summary": doc.page_content,
                "tags": doc.metadata.get("tags"),
                "story_id": doc.metadata.get("story_id"),
                "similarity_score": float(score)
            })
        
        return similar_stories
    
    def enrich_story_theme(self, theme: str, age_group: Optional[int] = None) -> str:
        """
        동화 주제를 RAG를 통해 풍부하게 만들기
        
        Args:
            theme: 동화 주제
            age_group: 연령대
            
        Returns:
            str: 풍부해진 주제
        """
        # 유사한 동화 검색
        similar_stories = self.get_similar_stories(theme, age_group, n_results=3)
        
        # 비슷한 동화가 없는 경우
        if not similar_stories:
            return theme
        
        # 풍부한 주제 생성을 위한 프롬프트
        prompt = f"""
        다음 주제를 바탕으로 더 풍부하고 창의적인 동화 주제를 만들어주세요:
        주제: {theme}
        
        참고할 수 있는 비슷한 동화:
        """
        
        for i, story in enumerate(similar_stories):
            prompt += f"\n{i+1}. {story['title']}: {story['summary'][:200]}..."
        
        # LLM으로 풍부한 주제 생성
        response = self.llm.predict(prompt)
        
        return response.strip()
    
    def generate_few_shot_examples(self, age_group: int, theme: str, n_examples: int = 2) -> str:
        """
        Few-shot 학습을 위한 예시 생성
        
        Args:
            age_group: 연령대
            theme: 동화 주제
            n_examples: 예시 수
            
        Returns:
            str: Few-shot 예시
        """
        # 유사한 동화 검색
        similar_stories = self.get_similar_stories(theme, age_group, n_results=n_examples)
        
        # 예시 생성
        examples = ""
        
        # 비슷한 동화가 없는 경우 빈 문자열 반환
        if not similar_stories:
            return examples
        
        # 예시 포맷팅
        for i, story in enumerate(similar_stories):
            examples += f"\n예시 {i+1}:\n"
            examples += f"제목: {story['title']}\n"
            examples += f"대상 연령: {story['tags']}\n"
            examples += f"줄거리: {story['summary']}\n\n"
        
        return examples.strip()
    
    def import_sample_stories(self, sample_dir: str = "data/sample_stories"):
        """
        샘플 동화 데이터 가져오기
        
        Args:
            sample_dir: 샘플 동화 디렉토리
        """
        # 샘플 디렉토리 확인
        if not os.path.exists(sample_dir):
            print(f"샘플 디렉토리가 존재하지 않습니다: {sample_dir}")
            return
        
        # 샘플 파일 목록 가져오기
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.json')]
        
        for file in sample_files:
            file_path = os.path.join(sample_dir, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    story_data = json.load(f)
                
                # 필수 필드 확인
                if all(key in story_data for key in ['title', 'tags', 'summary', 'content']):
                    self.add_story(
                        title=story_data['title'],
                        tags=story_data['tags'],
                        summary=story_data['summary'],
                        content=story_data['content'],
                        story_id=story_data.get('story_id')
                    )
                    print(f"샘플 동화 추가됨: {story_data['title']}")
                else:
                    print(f"필수 필드가 누락된 샘플 파일: {file}")
            
            except Exception as e:
                print(f"샘플 파일 처리 중 오류 발생: {file}, 오류: {str(e)}")
    
    def get_few_shot_prompt(self, age_group: int, theme: str, n_examples: int = 2) -> str:
        """
        Few-shot 프롬프트 생성
        
        Args:
            age_group: 연령대
            theme: 동화 주제
            n_examples: 예시 수
            
        Returns:
            str: Few-shot 프롬프트
        """
        # 예시 가져오기
        examples = self.generate_few_shot_examples(age_group, theme, n_examples)
        
        # 기본 프롬프트 템플릿
        prompt = f"""
        당신은 {age_group}-{age_group+1}세 아이들을 위한 동화를 창작하는 전문가입니다.
        아래 예시들을 참고하여 "{theme}" 주제의 동화를 작성해주세요.
        
        {examples}
        
        위 예시들처럼 {age_group}-{age_group+1}세 아이들이 이해하고 즐길 수 있는 창의적인 동화를 만들어주세요.
        """
        
        return prompt.strip()
    
    def import_processed_stories(self, stories_data: List[Dict[str, Any]], required_fields: Optional[List[str]] = None) -> List[str]:
        """
        전처리된 스토리 데이터에서 필요한 필드만 추출하여 ChromaDB에 저장
        
        Args:
            stories_data: 전처리된 스토리 데이터 리스트
            required_fields: 필수 필드 목록 (기본값: title, tags, summary, content)
            
        Returns:
            List[str]: 추가된 스토리 ID 목록
        """
        if required_fields is None:
            required_fields = ['title', 'tags', 'summary', 'content']
        
        added_story_ids = []
        
        for story in stories_data:
            # 필수 필드 확인
            if not all(field in story for field in required_fields):
                print(f"필수 필드가 누락된 스토리 데이터: {story.get('title', '제목 없음')}")
                continue
            
            try:
                # 스토리 ID 가져오기 또는 생성
                story_id = story.get('story_id')
                
                # 필요한 필드만 추출하여 ChromaDB에 추가
                added_id = self.add_story(
                    title=story.get('title', '제목 없음'),
                    tags=story.get('tags', ''),
                    summary=story.get('summary', ''),
                    content=story.get('content', ''),
                    story_id=story_id
                )
                
                added_story_ids.append(added_id)
                print(f"스토리 데이터 추가됨: {story.get('title', '제목 없음')} (ID: {added_id})")
            
            except Exception as e:
                print(f"스토리 데이터 처리 중 오류 발생: {story.get('title', '제목 없음')}, 오류: {str(e)}")
        
        return added_story_ids
        
    def filter_and_import_stories(self, stories_data: List[Dict[str, Any]], filter_criteria: Dict[str, Any]) -> List[str]:
        """
        필터링 기준에 맞는 스토리만 ChromaDB에 저장
        
        Args:
            stories_data: 전처리된 스토리 데이터 리스트
            filter_criteria: 필터링 기준 (예: {"age_group": 5, "theme": "우주"})
            
        Returns:
            List[str]: 추가된 스토리 ID 목록
        """
        filtered_stories = []
        
        for story in stories_data:
            # 모든 필터 기준을 만족하는지 확인
            matches_all_criteria = True
            
            for key, value in filter_criteria.items():
                # 특정 필드에 대한 부분 일치 검색
                if key in story:
                    # 문자열의 경우 부분 일치 확인
                    if isinstance(story[key], str) and isinstance(value, str):
                        if value.lower() not in story[key].lower():
                            matches_all_criteria = False
                            break
                    # 숫자나 기타 타입의 경우 정확히 일치하는지 확인
                    elif story[key] != value:
                        matches_all_criteria = False
                        break
                else:
                    # 필드가 없는 경우 기준 불만족
                    matches_all_criteria = False
                    break
            
            if matches_all_criteria:
                filtered_stories.append(story)
        
        print(f"필터링 결과: 전체 {len(stories_data)}개 중 {len(filtered_stories)}개 스토리가 기준에 부합")
        
        # 필터링된 스토리만 추가
        return self.import_processed_stories(filtered_stories)
    
    def import_stories_from_json(self, json_file_path: str, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        JSON 파일에서 전처리된 스토리 데이터를 로드하여 ChromaDB에 추가
        
        Args:
            json_file_path: JSON 파일 경로
            filter_criteria: 필터링 기준 (예: {"age_group": 5, "theme": "우주"})
            
        Returns:
            List[str]: 추가된 스토리 ID 목록
        """
        try:
            # JSON 파일 로드
            with open(json_file_path, 'r', encoding='utf-8') as f:
                stories_data = json.load(f)
            
            print(f"JSON 파일 로드됨: {json_file_path}, 총 {len(stories_data)}개 스토리 발견")
            
            # 리스트가 아닌 경우 리스트로 변환
            if not isinstance(stories_data, list):
                if isinstance(stories_data, dict) and "stories" in stories_data:
                    stories_data = stories_data["stories"]
                else:
                    stories_data = [stories_data]
            
            # 필터링 적용 여부에 따라 처리
            if filter_criteria:
                return self.filter_and_import_stories(stories_data, filter_criteria)
            else:
                return self.import_processed_stories(stories_data)
                
        except Exception as e:
            print(f"JSON 파일 처리 중 오류 발생: {json_file_path}, 오류: {str(e)}")
            return []
            
    def import_stories_from_directory(self, directory_path: str, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        디렉토리 내의 모든 JSON 파일에서 스토리 데이터를 로드하여 ChromaDB에 추가
        
        Args:
            directory_path: JSON 파일이 있는 디렉토리 경로
            filter_criteria: 필터링 기준 (예: {"age_group": 5, "theme": "우주"})
            
        Returns:
            List[str]: 추가된 스토리 ID 목록
        """
        # 디렉토리 확인
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            print(f"유효한 디렉토리가 아닙니다: {directory_path}")
            return []
        
        # JSON 파일 목록 가져오기
        json_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                     if f.endswith('.json') and os.path.isfile(os.path.join(directory_path, f))]
        
        if not json_files:
            print(f"디렉토리에 JSON 파일이 없습니다: {directory_path}")
            return []
        
        # 모든 파일에서 스토리 추가
        all_added_story_ids = []
        for json_file in json_files:
            print(f"파일 처리 중: {os.path.basename(json_file)}")
            added_ids = self.import_stories_from_json(json_file, filter_criteria)
            all_added_story_ids.extend(added_ids)
        
        print(f"총 {len(all_added_story_ids)}개 스토리가 추가되었습니다.")
        return all_added_story_ids
    
    def preview_filtered_stories(self, stories_data: List[Dict[str, Any]], filter_criteria: Dict[str, Any], verbose: bool = False) -> List[Dict[str, Any]]:
        """
        필터링 기준에 맞는 스토리 목록을 미리 확인
        
        Args:
            stories_data: 전처리된 스토리 데이터 리스트
            filter_criteria: 필터링 기준 (예: {"age_group": 5, "theme": "우주"})
            verbose: 상세 정보 출력 여부
            
        Returns:
            List[Dict]: 필터링된 스토리 목록
        """
        filtered_stories = []
        
        for story in stories_data:
            # 모든 필터 기준을 만족하는지 확인
            matches_all_criteria = True
            
            for key, value in filter_criteria.items():
                # 특정 필드에 대한 부분 일치 검색
                if key in story:
                    # 문자열의 경우 부분 일치 확인
                    if isinstance(story[key], str) and isinstance(value, str):
                        if value.lower() not in story[key].lower():
                            matches_all_criteria = False
                            break
                    # 숫자나 기타 타입의 경우 정확히 일치하는지 확인
                    elif story[key] != value:
                        matches_all_criteria = False
                        break
                else:
                    # 필드가 없는 경우 기준 불만족
                    matches_all_criteria = False
                    break
            
            if matches_all_criteria:
                filtered_stories.append(story)
        
        print(f"필터링 결과: 전체 {len(stories_data)}개 중 {len(filtered_stories)}개 스토리가 기준에 부합")
        
        # 미리보기 정보 출력
        if verbose and filtered_stories:
            print("\n===== 필터링된 스토리 목록 =====")
            for i, story in enumerate(filtered_stories, 1):
                print(f"\n[{i}] {story.get('title', '제목 없음')}")
                print(f"   - ID: {story.get('story_id', '자동 생성')}")
                print(f"   - 태그: {story.get('tags', '')}")
                if 'summary' in story:
                    summary = story['summary']
                    # 긴 요약은 일부만 표시
                    if len(summary) > 100:
                        summary = summary[:100] + "..."
                    print(f"   - 요약: {summary}")
                    
        return filtered_stories
    
    def filter_preview_and_import(self, stories_data: List[Dict[str, Any]], filter_criteria: Dict[str, Any]) -> List[str]:
        """
        필터링된 스토리를 미리 확인하고 ChromaDB에 추가
        
        Args:
            stories_data: 전처리된 스토리 데이터 리스트
            filter_criteria: 필터링 기준 (예: {"age_group": 5, "theme": "우주"})
            
        Returns:
            List[str]: 추가된 스토리 ID 목록
        """
        # 필터링된 스토리 미리보기
        filtered_stories = self.preview_filtered_stories(stories_data, filter_criteria, verbose=True)
        
        if not filtered_stories:
            print("필터링 조건에 맞는 스토리가 없습니다.")
            return []
        
        # 사용자에게 확인 (실제 사용 시에는 입력 받기)
        # 여기서는 자동으로 추가하도록 처리
        print(f"\n필터링된 {len(filtered_stories)}개 스토리를 ChromaDB에 추가합니다...")
        
        # 필터링된 스토리 추가
        return self.import_processed_stories(filtered_stories)
        
    def extract_fields_from_stories(self, stories_data: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
        """
        스토리 데이터에서 특정 필드만 추출
        
        Args:
            stories_data: 스토리 데이터 리스트
            fields: 추출할 필드 목록
            
        Returns:
            List[Dict]: 추출된 필드만 포함한 스토리 데이터
        """
        extracted_data = []
        
        for story in stories_data:
            extracted_story = {}
            for field in fields:
                if field in story:
                    extracted_story[field] = story[field]
            
            # 최소한 하나 이상의 필드가 추출된 경우에만 추가
            if extracted_story:
                extracted_data.append(extracted_story)
        
        return extracted_data 