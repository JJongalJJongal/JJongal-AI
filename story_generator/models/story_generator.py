import openai
import pinecone
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import json
import time

# 환경 변수 설정
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

class StoryGenerator:
    """
    GPT-4o-mini와 RAG를 활용한 동화 생성 클래스
    
    Attributes:
        index (pinecone.Index): 벡터 데이터베이스 인덱스
        fine_tuned_model (str): Fine-tuning된 모델 ID
        rag_chain (RunnablePassthrough): RAG 체인
    """
    
    def __init__(self):
        """StoryGenerator 초기화"""
        # Pinecone 초기화
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        self.index = pinecone.Index('story-index')
        self.fine_tuned_model = None
        
        # RAG 체인 설정
        self.setup_rag_chain()
    
    def setup_rag_chain(self):
        """RAG 체인 설정"""
        # 템플릿 설정
        template = """당신은 {age_group}세 아이들을 위한 동화 작가입니다.
        주어진 컨텍스트를 참고하여 희망적이고 교육적인 동화를 작성해주세요.
        
        컨텍스트:
        {context}
        
        주제: {theme}
        
        지침:
        1. 아이들의 연령대에 맞는 어휘와 문장 구조 사용
        2. 긍정적이고 교육적인 메시지 포함
        3. 명확한 시작, 중간, 끝 구조
        4. 적절한 길이와 페이스
        """
        
        # RAG 체인 구성
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-4o-mini")
        self.rag_chain = (
            {"context": self.retrieve_context, "theme": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
    
    def retrieve_context(self, query: str) -> str:
        """
        벡터 데이터베이스에서 관련 컨텍스트 검색
        
        Args:
            query (str): 검색 쿼리
            
        Returns:
            str: 검색된 컨텍스트
        """
        # 쿼리 임베딩 생성
        query_embedding = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )['data'][0]['embedding']
        
        # 유사 컨텍스트 검색
        results = self.index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        # 컨텍스트 결합
        context = "\n".join([match.metadata['text'] for match in results.matches])
        return context
    
    def prepare_training_data(self, data_path: str) -> List[Dict[str, str]]:
        """
        Fine-tuning을 위한 데이터 준비
        
        Args:
            data_path (str): 데이터 파일 경로
            
        Returns:
            List[Dict[str, str]]: 전처리된 학습 데이터
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # GPT API를 사용하여 데이터 전처리 및 분석
            processed_data = []
            for item in data:
                # 데이터 구조화 및 품질 검증
                analysis_prompt = f"""
                Analyze this story data and ensure it's properly formatted for fine-tuning:
                {json.dumps(item, ensure_ascii=False)}
                
                Return a properly formatted training example with:
                1. Clear system message
                2. Well-structured user message
                3. Appropriate assistant response
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a data processing expert."},
                        {"role": "user", "content": analysis_prompt}
                    ]
                )
                
                processed_item = json.loads(response.choices[0].message.content)
                processed_data.append(processed_item)
            
            return processed_data
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            return []
    
    def fine_tune_model(self, training_data: List[Dict[str, str]]):
        """
        GPT-4o-mini 모델 Fine-tuning
        
        Args:
            training_data (List[Dict[str, str]]): 학습 데이터
        """
        try:
            # 데이터 파일 생성
            training_file = "data/story/training/fine_tuning_data.jsonl"
            with open(training_file, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 파일 업로드
            with open(training_file, 'rb') as f:
                response = openai.File.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            # Fine-tuning 작업 생성
            fine_tune_response = openai.FineTuningJob.create(
                training_file=response.id,
                model="gpt-4o-mini",
                hyperparameters={
                    "n_epochs": 3,
                    "batch_size": 4,
                    "learning_rate_multiplier": 0.1
                }
            )
            
            # Fine-tuning 모니터링
            while True:
                job_status = openai.FineTuningJob.retrieve(fine_tune_response.id)
                if job_status.status == 'succeeded':
                    self.fine_tuned_model = job_status.fine_tuned_model
                    break
                elif job_status.status == 'failed':
                    raise Exception("Fine-tuning failed")
                time.sleep(60)  # 1분마다 상태 확인
            
        except Exception as e:
            print(f"Error during fine-tuning: {str(e)}")
            raise
    
    def generate_story(self, prompt: str, age_group: int, theme: str) -> str:
        """
        동화 줄거리를 생성하는 함수
        
        Args:
            prompt (str): 사용자 입력 프롬프트
            age_group (int): 아이의 연령대
            theme (str): 동화 주제
            
        Returns:
            str: 생성된 동화
        """
        try:
            # RAG 체인을 사용하여 동화 생성
            story = self.rag_chain.invoke({
                "theme": theme,
                "age_group": age_group
            })
            
            return story
            
        except Exception as e:
            print(f"Error generating story: {str(e)}")
            return "동화를 생성하는 중 오류가 발생했습니다." 