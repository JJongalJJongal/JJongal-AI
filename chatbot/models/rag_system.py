import os
import json
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tiktoken
from dotenv import load_dotenv
from pathlib import Path

# 환경 변수 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

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
        self.summary_vectorstore = Chroma(
            persist_directory=self.summary_persist_dir,
            embedding_function=self.embeddings
        )
        
        self.detailed_vectorstore = Chroma(
            persist_directory=self.detailed_persist_dir,
            embedding_function=self.embeddings
        )
        
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