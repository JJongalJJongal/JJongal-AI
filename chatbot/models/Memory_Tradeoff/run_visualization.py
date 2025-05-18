import os
import sys
import argparse
from tradeoff_visualization import run_comprehensive_benchmarks

def main():
    """
    벡터 DB의 메모리-디스크 트레이드오프 시각화 도구 실행
    
    사용법:
      python run_visualization.py [--quick]
      
    옵션:
      --quick  빠른 모드로 실행 (작은 데이터셋만)
    """
    parser = argparse.ArgumentParser(description="벡터 DB 메모리-디스크 트레이드오프 시각화")
    parser.add_argument("--quick", action="store_true", help="빠른 모드로 실행 (작은 데이터셋만)")
    args = parser.parse_args()
    
    # 실행 위치 확인 및 경로 조정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 50)
    print(" ChromaDB 메모리 vs 디스크 트레이드오프 시각화")
    print("=" * 50)
    
    # 빠른 모드 실행 시 환경 변수 설정
    if args.quick:
        print("\n빠른 모드로 실행합니다 (작은 데이터셋만)...")
        os.environ["QUICK_MODE"] = "1"
    else:
        print("\n전체 모드로 실행합니다...")
        if "QUICK_MODE" in os.environ:
            del os.environ["QUICK_MODE"]
    
    # 결과 디렉토리 생성
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 기존 임시 파일 정리
    for temp_dir in ["./chroma_test_persistent", "./chroma_db_test"]:
        if os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
                print(f"임시 디렉토리 정리: {temp_dir}")
            except Exception as e:
                print(f"임시 디렉토리 정리 중 오류: {e}")
    
    # 벤치마크 실행
    try:
        run_comprehensive_benchmarks()
        print("\n시각화 결과가 저장되었습니다!")
        print(f"결과 위치: {os.path.abspath(script_dir)}")
    except Exception as e:
        print(f"\n벤치마크 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 