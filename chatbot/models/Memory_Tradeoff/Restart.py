import chromadb
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm

# 테스트 디렉토리
persistent_dir = "./chroma_persistent_test"

# 다양한 크기의 벡터 세트
vector_counts = [1000, 5000, 10000, 50000]
dimensions = 384

# 결과 저장 변수
recovery_times = []

for count in vector_counts:
    # 이전 데이터 정리
    if os.path.exists(persistent_dir):
        shutil.rmtree(persistent_dir)
    
    print(f"\n{count}개 벡터로 재시작 테스트 중...")
    
    # 1. 첫 번째 클라이언트로 데이터 생성
    client = chromadb.PersistentClient(path=persistent_dir)
    collection = client.create_collection(name=f"test_collection_{count}")
    
    # 데이터 삽입
    for i in tqdm(range(count), desc="데이터 삽입"):
        vector = np.random.rand(dimensions).tolist()
        collection.add(
            embeddings=[vector],
            documents=[f"Document {i}"],
            ids=[f"id_{i}"]
        )
    
    # 클라이언트 종료 (세션 종료 시뮬레이션)
    del client
    
    # 2. 두 번째 클라이언트로 데이터 로드 (재시작 시뮬레이션)
    start_time = time.time()
    new_client = chromadb.PersistentClient(path=persistent_dir)
    
    # 컬렉션 가져오기
    loaded_collection = new_client.get_collection(name=f"test_collection_{count}")
    
    # 간단한 쿼리 (로드 완료 확인용)
    query_vector = np.random.rand(dimensions).tolist()
    results = loaded_collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )
    
    # 복구 시간 계산
    recovery_time = time.time() - start_time
    recovery_times.append({
        "vector_count": count,
        "recovery_time": recovery_time
    })
    
    print(f"{count}개 벡터 복구 시간: {recovery_time:.4f}초")
    
    # 클라이언트 정리
    del new_client

# 결과 시각화
plt.figure(figsize=(10, 6))
counts = [r["vector_count"] for r in recovery_times]
times = [r["recovery_time"] for r in recovery_times]

plt.plot(counts, times, marker='o', linestyle='-', linewidth=2)
plt.title('ChromaDB 재시작 후 데이터 복구 시간')
plt.xlabel('벡터 수')
plt.ylabel('복구 시간 (초)')
plt.grid(True)
plt.xticks(counts)

for i, (count, time_val) in enumerate(zip(counts, times)):
    plt.annotate(f"{time_val:.2f}s", 
                 (count, time_val),
                 xytext=(5, 10), 
                 textcoords='offset points')

plt.tight_layout()
plt.savefig('chroma_recovery_time.png', dpi=300)
plt.show()