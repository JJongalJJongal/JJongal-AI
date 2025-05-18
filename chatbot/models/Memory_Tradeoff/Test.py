import chromadb
import time
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# 프로젝트 루트 디렉토리 추가
sys.path.append(".")
from chatbot.data.vector_db import get_existing_collection_data, import_into_collection

# 테스트 디렉토리
in_memory_collection_name = "test_in_memory"
disk_collection_name = "test_disk"
persistent_dir = "./chroma_test_persistent"

# 벡터 DB 클라이언트 생성
in_memory_client = chromadb.Client()
disk_client = chromadb.PersistentClient(path=persistent_dir)

# 기존 컬렉션 데이터 가져오기
existing_docs, existing_ids, existing_embeddings = get_existing_collection_data(
    collection_name="main",
    collection_path="./chatbot/data/vector_db/main"
)

print(f"테스트할 데이터: {len(existing_ids)}개 벡터")

# 성능 측정 함수
def benchmark_operations(client, collection_name, embeddings, ids, docs):
    # 컬렉션 생성
    if collection_name in client.list_collections():
        client.delete_collection(collection_name)
    
    collection = client.create_collection(name=collection_name)
    
    # 삽입 시간 측정
    start_time = time.time()
    
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            embeddings=embeddings[i:end],
            documents=docs[i:end],
            ids=ids[i:end]
        )
    
    insertion_time = time.time() - start_time
    
    # 쿼리 시간 측정
    query_times = []
    query_count = 50
    
    for _ in range(query_count):
        # 무작위 벡터 선택해서 쿼리
        random_idx = np.random.randint(0, len(embeddings))
        query_vector = embeddings[random_idx]
        
        query_start = time.time()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=10
        )
        query_times.append(time.time() - query_start)
    
    avg_query_time = np.mean(query_times) * 1000  # ms 단위
    
    return {
        "insertion_time": insertion_time,
        "avg_query_time": avg_query_time
    }

# 인메모리 벤치마크
print("인메모리 벤치마크 실행 중...")
in_memory_results = benchmark_operations(
    in_memory_client, 
    in_memory_collection_name,
    existing_embeddings,
    existing_ids,
    existing_docs
)

# 디스크 벤치마크
print("디스크 벤치마크 실행 중...")
disk_results = benchmark_operations(
    disk_client,
    disk_collection_name,
    existing_embeddings,
    existing_ids,
    existing_docs
)

# 결과 출력
print("\n===== 결과 =====")
print(f"데이터 크기: {len(existing_ids)}개 벡터")
print("\n[In-Memory]")
print(f"삽입 시간: {in_memory_results['insertion_time']:.2f}초")
print(f"평균 쿼리 시간: {in_memory_results['avg_query_time']:.2f}ms")
print("\n[Disk]")
print(f"삽입 시간: {disk_results['insertion_time']:.2f}초")
print(f"평균 쿼리 시간: {disk_results['avg_query_time']:.2f}ms")

# 결과 시각화
plt.figure(figsize=(12, 6))

# 삽입 시간 비교
plt.subplot(1, 2, 1)
storage_types = ['In-Memory', 'Disk']
insertion_times = [in_memory_results['insertion_time'], disk_results['insertion_time']]

plt.bar(storage_types, insertion_times, color=['#3498db', '#2ecc71'])
plt.ylabel('시간 (초)')
plt.title('삽입 시간 비교')
for i, v in enumerate(insertion_times):
    plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')

# 쿼리 시간 비교
plt.subplot(1, 2, 2)
query_times = [in_memory_results['avg_query_time'], disk_results['avg_query_time']]

plt.bar(storage_types, query_times, color=['#3498db', '#2ecc71'])
plt.ylabel('시간 (ms)')
plt.title('평균 쿼리 시간 비교')
for i, v in enumerate(query_times):
    plt.text(i, v + 0.1, f"{v:.2f}ms", ha='center')

plt.tight_layout()
plt.savefig('chroma_real_data_comparison.png', dpi=300)
plt.show()