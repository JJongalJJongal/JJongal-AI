import chromadb
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import psutil

# 성능 측정 함수
def measure_performance(client, collection_name, num_vectors, dimensions, query_count):
    # 메모리 사용량 측정
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB 단위
    
    # 컬렉션 생성
    collection = client.create_collection(name=collection_name)
    
    # 시간 측정 시작
    start_time = time.time()
    
    # 랜덤 벡터 생성 및 삽입
    for i in tqdm(range(num_vectors), desc="데이터 삽입"):
        vector = np.random.rand(dimensions).tolist()
        collection.add(
            embeddings=[vector],
            documents=[f"Document {i}"],
            ids=[f"id_{i}"]
        )
    
    # 삽입 시간 계산
    insertion_time = time.time() - start_time
    
    # 메모리 사용량 계산
    memory_usage = process.memory_info().rss / 1024 / 1024 - initial_memory
    
    # 쿼리 성능 측정
    query_times = []
    for i in tqdm(range(query_count), desc="쿼리 실행"):
        query_vector = np.random.rand(dimensions).tolist()
        
        query_start = time.time()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=10
        )
        query_times.append(time.time() - query_start)
    
    avg_query_time = np.mean(query_times) * 1000  # ms 단위
    
    # 결과 반환
    return {
        "insertion_time": insertion_time,
        "memory_usage": memory_usage,
        "avg_query_time": avg_query_time
    }

# 실험 파라미터
vector_counts = [1000, 5000, 10000, 50000]
dimensions = 384  # 벡터 차원수
query_count = 100  # 각 설정마다 실행할 쿼리 수

# 결과 저장 변수
results = []

# 1. In-Memory 클라이언트 테스트
print("In-Memory 클라이언트 테스트 실행 중...")
in_memory_client = chromadb.Client()

for count in vector_counts:
    print(f"\n{count}개 벡터 테스트 중...")
    result = measure_performance(
        in_memory_client, 
        f"in_memory_collection_{count}", 
        count, 
        dimensions, 
        query_count
    )
    results.append({
        "type": "In-Memory",
        "vector_count": count,
        **result
    })
    # 컬렉션 삭제
    in_memory_client.delete_collection(f"in_memory_collection_{count}")

# 2. Disk 기반 클라이언트 테스트
print("\nDisk 기반 클라이언트 테스트 실행 중...")
persistent_dir = "./chroma_db_test"
disk_client = chromadb.PersistentClient(path=persistent_dir)

for count in vector_counts:
    print(f"\n{count}개 벡터 테스트 중...")
    result = measure_performance(
        disk_client, 
        f"disk_collection_{count}", 
        count, 
        dimensions, 
        query_count
    )
    results.append({
        "type": "Disk",
        "vector_count": count,
        **result
    })
    # 컬렉션 삭제
    disk_client.delete_collection(f"disk_collection_{count}")

# 결과를 DataFrame으로 변환
df = pd.DataFrame(results)
print("\n결과:")
print(df)

# 결과 시각화
plt.figure(figsize=(15, 10))

# 1. 쿼리 시간 비교
plt.subplot(2, 2, 1)
for storage_type in ["In-Memory", "Disk"]:
    data = df[df["type"] == storage_type]
    plt.plot(data["vector_count"], data["avg_query_time"], marker='o', label=storage_type)
plt.xlabel('벡터 수')
plt.ylabel('평균 쿼리 시간 (ms)')
plt.title('벡터 수에 따른 쿼리 시간')
plt.legend()
plt.grid(True)

# 2. 삽입 시간 비교
plt.subplot(2, 2, 2)
for storage_type in ["In-Memory", "Disk"]:
    data = df[df["type"] == storage_type]
    plt.plot(data["vector_count"], data["insertion_time"], marker='o', label=storage_type)
plt.xlabel('벡터 수')
plt.ylabel('삽입 시간 (초)')
plt.title('벡터 수에 따른 삽입 시간')
plt.legend()
plt.grid(True)

# 3. 메모리 사용량 비교
plt.subplot(2, 2, 3)
for storage_type in ["In-Memory", "Disk"]:
    data = df[df["type"] == storage_type]
    plt.plot(data["vector_count"], data["memory_usage"], marker='o', label=storage_type)
plt.xlabel('벡터 수')
plt.ylabel('메모리 사용량 (MB)')
plt.title('벡터 수에 따른 메모리 사용량')
plt.legend()
plt.grid(True)

# 4. 트레이드오프 시각화: 쿼리 시간 vs 메모리 사용량
plt.subplot(2, 2, 4)
for storage_type in ["In-Memory", "Disk"]:
    data = df[df["type"] == storage_type]
    plt.scatter(data["avg_query_time"], data["memory_usage"], 
                label=storage_type, s=100, alpha=0.7)
    # 벡터 수 표시
    for i, count in enumerate(data["vector_count"]):
        plt.annotate(f"{count}", 
                     (data["avg_query_time"].iloc[i], data["memory_usage"].iloc[i]),
                     xytext=(5, 5), textcoords='offset points')
plt.xlabel('평균 쿼리 시간 (ms)')
plt.ylabel('메모리 사용량 (MB)')
plt.title('트레이드오프: 쿼리 시간 vs 메모리 사용량')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('chroma_performance_comparison.png', dpi=300)
plt.show()