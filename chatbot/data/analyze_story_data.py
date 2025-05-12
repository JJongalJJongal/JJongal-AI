import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import re
from tqdm import tqdm
import matplotlib.font_manager as fm

# 경로 설정 (절대 경로로 변경)
STORY_DATA_DIR = '/Users/b._.chan/Documents/University/캡스톤디자인/AI/CCB_AI/chatbot/data/processed/story_data'
ANALYSIS_OUTPUT_DIR = '/Users/b._.chan/Documents/University/캡스톤디자인/AI/CCB_AI/chatbot/data/processed/analysis'

# 데이터 경로 확인
print(f"현재 작업 디렉토리: {os.getcwd()}")
print(f"스토리 데이터 경로: {STORY_DATA_DIR}")
print(f"스토리 데이터 경로 존재 여부: {os.path.exists(STORY_DATA_DIR)}")

# 분석 결과 저장 디렉토리 생성
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# 한글 폰트 설정 시도
try:
    # 맥OS용 한글 폰트 경로
    font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        print(f"한글 폰트 '{font_prop.get_name()}' 설정 성공")
    else:
        print("기본 한글 폰트를 찾을 수 없어 영어로 시각화를 진행합니다.")
except Exception as e:
    print(f"폰트 설정 중 오류 발생: {str(e)}")
    print("영어로 시각화를 진행합니다.")

def read_json_file(file_path):
    """BOM 유무에 관계없이 JSON 파일을 일관되게 읽는 함수"""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON 파일 읽기 오류: {file_path} - {str(e)}")
        return {}

def load_all_stories():
    """모든 스토리 데이터 로드"""
    print("스토리 데이터 로드 중...")
    stories = []
    file_paths = [os.path.join(STORY_DATA_DIR, f) for f in os.listdir(STORY_DATA_DIR) if f.endswith('.json')]
    
    for file_path in tqdm(file_paths):
        data = read_json_file(file_path)
        if data:
            stories.append(data)
    
    print(f"총 {len(stories)}개의 스토리 데이터를 로드했습니다.")
    return stories

def basic_statistics(stories):
    """기본 통계 정보 계산"""
    print("기본 통계 정보 계산 중...")
    stats = {
        "total_stories": len(stories),
        "content_lengths": [],
        "summary_lengths": [],
        "images_per_story": [],
        "has_title": 0,
        "has_content": 0,
        "has_images": 0,
        "has_metadata": {
            "author": 0,
            "publisher": 0,
            "isbn": 0,
            "published_year": 0
        },
        "age_groups": Counter(),
        "categories": Counter(),
        "publishers": Counter(),
        "authors": Counter(),
        "published_years": Counter()
    }
    
    for story in stories:
        # 텍스트 길이
        if story.get("content"):
            stats["content_lengths"].append(len(story["content"]))
            stats["has_content"] += 1
        
        if story.get("summary"):
            stats["summary_lengths"].append(len(story["summary"]))
        
        # 이미지 수
        images = story.get("images", [])
        stats["images_per_story"].append(len(images))
        if images:
            stats["has_images"] += 1
        
        # 제목 유무
        if story.get("title"):
            stats["has_title"] += 1
        
        # 메타데이터
        metadata = story.get("metadata", {})
        if metadata.get("author"):
            stats["has_metadata"]["author"] += 1
            stats["authors"][metadata["author"]] += 1
        
        if metadata.get("publisher"):
            stats["has_metadata"]["publisher"] += 1
            stats["publishers"][metadata["publisher"]] += 1
        
        if metadata.get("isbn"):
            stats["has_metadata"]["isbn"] += 1
        
        if metadata.get("published_year"):
            stats["has_metadata"]["published_year"] += 1
            stats["published_years"][str(metadata["published_year"])] += 1
        
        # 태그 분석 - 수정된 연령대 (4-6세, 7-9세)
        tags = story.get("tags", "").split(",")
        for tag in tags:
            tag = tag.strip()
            # 변경된 연령대 패턴 매칭 (4-6세 또는 7-9세)
            if tag == "4-6세" or tag == "7-9세":
                stats["age_groups"][tag] += 1
            elif tag in ["의사소통", "자연탐구", "사회관계", "예술경험", "신체운동_건강"]:
                stats["categories"][tag] += 1
    
    return stats

def analyze_text_content(stories):
    """텍스트 내용 분석 (단어 빈도 등)"""
    print("텍스트 내용 분석 중...")
    all_content = " ".join([story.get("content", "") for story in stories if story.get("content")])
    
    # 불용어 처리 (예시)
    stopwords = ["그리고", "그런데", "하지만", "그러나", "그래서", "있다", "있는", "한다", "하는", "된다", "되는"]
    
    # 단어 토큰화 (간단한 공백 기반 분리)
    words = re.findall(r'\w+', all_content)
    words = [word for word in words if len(word) > 1 and word not in stopwords]
    
    word_freq = Counter(words)
    return {
        "word_frequencies": word_freq,
        "total_words": len(words),
        "unique_words": len(word_freq)
    }

def visualize_data(stats, text_analysis):
    """분석 결과 시각화"""
    print("데이터 시각화 중...")
    
    # 1. 텍스트 길이 분포
    plt.figure(figsize=(12, 6))
    plt.hist(stats["content_lengths"], bins=50, alpha=0.7)
    plt.title('Content Length Distribution')
    plt.xlabel('Text Length (Characters)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "content_length_distribution.png"))
    plt.close()
    
    # 2. 연령대 분포 - 변경된 연령대 (4-6세, 7-9세)에 맞게 수정
    if stats["age_groups"]:
        plt.figure(figsize=(10, 6))
        # 연령대 영어로 변환
        age_mapping = {
            "4-6세": "4-6 Years",
            "7-9세": "7-9 Years"
        }
        age_labels = [age_mapping.get(k, k) for k in stats["age_groups"].keys()]
        age_values = list(stats["age_groups"].values())
        
        # 색상 설정
        colors = ['#3498db', '#2ecc71']
        
        # 바 그래프 그리기
        bars = plt.bar(age_labels, age_values, color=colors[:len(age_labels)])
        
        # 바 위에 숫자와 비율 표시
        for bar in bars:
            height = bar.get_height()
            percentage = height / stats["total_stories"] * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.title('Age Group Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Age Groups', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.ylim(0, max(age_values) * 1.1)  # y축 범위 설정
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "age_distribution.png"))
        plt.close()
    
    # 3. 카테고리 분포 - 개선된 부분
    if stats["categories"]:
        plt.figure(figsize=(12, 6))
        # 카테고리 영어로 변환
        category_mapping = {
            "의사소통": "Communication",
            "자연탐구": "Nature Exploration",
            "사회관계": "Social Relations",
            "예술경험": "Art Experience",
            "신체운동_건강": "Physical Health"
        }
        cat_labels = [category_mapping.get(k, k) for k in stats["categories"].keys()]
        cat_values = list(stats["categories"].values())
        
        # 색상 설정
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        # 바 그래프 그리기
        bars = plt.bar(cat_labels, cat_values, color=colors[:len(cat_labels)])
        
        # 바 위에 숫자와 비율 표시
        for bar in bars:
            height = bar.get_height()
            percentage = height / stats["total_stories"] * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.title('Category Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Categories', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.ylim(0, max(cat_values) * 1.1)  # y축 범위 설정
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "category_distribution.png"))
        plt.close()
    
    # 4. 출판 연도 분포
    if stats["published_years"]:
        plt.figure(figsize=(12, 6))
        years = sorted(stats["published_years"].keys())
        year_counts = [stats["published_years"][year] for year in years]
        plt.bar(years, year_counts)
        plt.title('Publication Year Distribution')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "published_year_distribution.png"))
        plt.close()
    
    # 5. 이미지 수 분포
    plt.figure(figsize=(10, 6))
    plt.hist(stats["images_per_story"], bins=range(0, max(stats["images_per_story"]) + 2), alpha=0.7)
    plt.title('Images per Story Distribution')
    plt.xlabel('Number of Images')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "images_per_story_distribution.png"))
    plt.close()
    
    # 6. 워드클라우드
    try:
        most_common_words = dict(text_analysis["word_frequencies"].most_common(200))
        wordcloud = WordCloud(
            font_path='/System/Library/Fonts/AppleSDGothicNeo.ttc',  # 한글 폰트 설정
            width=800, 
            height=400, 
            background_color='white'
        ).generate_from_frequencies(most_common_words)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "wordcloud.png"))
        plt.close()
    except Exception as e:
        print(f"워드클라우드 생성 중 오류 발생: {str(e)}")

def generate_report(stats, text_analysis):
    """분석 결과 보고서 생성"""
    print("분석 보고서 생성 중...")
    
    report = []
    report.append("# 스토리 데이터 분석 보고서\n")
    
    # 1. 기본 통계
    report.append("## 1. 기본 통계\n")
    report.append(f"- 전체 스토리 수: {stats['total_stories']}")
    report.append(f"- 제목이 있는 스토리: {stats['has_title']} ({stats['has_title']/stats['total_stories']*100:.1f}%)")
    report.append(f"- 컨텐츠가 있는 스토리: {stats['has_content']} ({stats['has_content']/stats['total_stories']*100:.1f}%)")
    report.append(f"- 이미지가 있는 스토리: {stats['has_images']} ({stats['has_images']/stats['total_stories']*100:.1f}%)")
    
    # 2. 메타데이터 통계
    report.append("\n## 2. 메타데이터 통계\n")
    report.append(f"- 저자 정보가 있는 스토리: {stats['has_metadata']['author']} ({stats['has_metadata']['author']/stats['total_stories']*100:.1f}%)")
    report.append(f"- 출판사 정보가 있는 스토리: {stats['has_metadata']['publisher']} ({stats['has_metadata']['publisher']/stats['total_stories']*100:.1f}%)")
    report.append(f"- ISBN 정보가 있는 스토리: {stats['has_metadata']['isbn']} ({stats['has_metadata']['isbn']/stats['total_stories']*100:.1f}%)")
    report.append(f"- 출판 연도 정보가 있는 스토리: {stats['has_metadata']['published_year']} ({stats['has_metadata']['published_year']/stats['total_stories']*100:.1f}%)")
    
    # 3. 텍스트 통계
    report.append("\n## 3. 텍스트 통계\n")
    if stats["content_lengths"]:
        report.append(f"- 평균 텍스트 길이: {np.mean(stats['content_lengths']):.1f} 글자")
        report.append(f"- 최대 텍스트 길이: {max(stats['content_lengths'])} 글자")
        report.append(f"- 최소 텍스트 길이: {min(stats['content_lengths'])} 글자")
    
    if stats["summary_lengths"]:
        report.append(f"- 평균 요약 길이: {np.mean(stats['summary_lengths']):.1f} 글자")
    
    # 4. 이미지 통계
    report.append("\n## 4. 이미지 통계\n")
    report.append(f"- 평균 이미지 수: {np.mean(stats['images_per_story']):.1f}")
    report.append(f"- 최대 이미지 수: {max(stats['images_per_story'])}")
    
    # 5. 단어 통계
    report.append("\n## 5. 단어 통계\n")
    report.append(f"- 총 단어 수: {text_analysis['total_words']}")
    report.append(f"- 고유 단어 수: {text_analysis['unique_words']}")
    
    # 6. 빈도 상위 단어
    report.append("\n## 6. 빈도 상위 30 단어\n")
    for word, count in text_analysis["word_frequencies"].most_common(30):
        report.append(f"- {word}: {count}")
    
    # 7. 연령대 분포
    report.append("\n## 7. 연령대 분포\n")
    for age, count in stats["age_groups"].most_common():
        report.append(f"- {age}: {count} ({count/stats['total_stories']*100:.1f}%)")
    
    # 8. 카테고리 분포
    report.append("\n## 8. 카테고리 분포\n")
    for category, count in stats["categories"].most_common():
        report.append(f"- {category}: {count} ({count/stats['total_stories']*100:.1f}%)")
    
    # 9. 상위 출판사
    report.append("\n## 9. 상위 10 출판사\n")
    for publisher, count in stats["publishers"].most_common(10):
        report.append(f"- {publisher}: {count}")
    
    # 10. 상위 저자
    report.append("\n## 10. 상위 10 저자\n")
    for author, count in stats["authors"].most_common(10):
        report.append(f"- {author}: {count}")
    
    # 보고서 저장
    with open(os.path.join(ANALYSIS_OUTPUT_DIR, "data_analysis_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"분석 보고서가 저장되었습니다: {os.path.join(ANALYSIS_OUTPUT_DIR, 'data_analysis_report.md')}")

def main():
    print(f"스토리 데이터 분석을 시작합니다. 데이터 경로: {STORY_DATA_DIR}")
    
    # 1. 모든 스토리 데이터 로드
    stories = load_all_stories()
    
    if not stories:
        print("분석할 스토리 데이터가 없습니다!")
        return
    
    # 2. 기본 통계 분석
    stats = basic_statistics(stories)
    
    # 3. 텍스트 내용 분석
    text_analysis = analyze_text_content(stories)
    
    # 4. 시각화
    visualize_data(stats, text_analysis)
    
    # 5. 보고서 생성
    generate_report(stats, text_analysis)
    
    print(f"분석이 완료되었습니다. 결과는 {ANALYSIS_OUTPUT_DIR} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 