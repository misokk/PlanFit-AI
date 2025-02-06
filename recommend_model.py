import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def search_places(query, location, radius):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "location": location,
        "radius": radius,
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    return response.json().get("results", [])

def get_place_details(place_id):
    """Place Details API를 호출하여 추가 정보 가져오는 함수"""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": API_KEY,
        "fields": "name,types"
    }
    response = requests.get(url, params=params)
    return response.json().get("result", {})

def vectorize_text(text):
    """문장을 벡터화"""
    return model.encode(text, convert_to_numpy=True)

def recommend_places(user_keywords, location, radius, query):
    places = search_places(query, location, radius)
    user_vector = vectorize_text(" ".join(user_keywords))
    recommendations = []

    for place in places:
        place_id = place.get("place_id", "")
        place_details = get_place_details(place_id)
        place_types = place_details.get("types", [])
        place_name = place.get("name", "정보 없음")

        # 장소 설명과 유형을 하나의 벡터로 변환
        place_text = f"{place_name} {' '.join(place_types)}"
        place_vector = vectorize_text(place_text)

        similarity = cosine_similarity([user_vector], [place_vector])[0][0]
        similarity_score = int(similarity * 100)  # 0~100의 정수 변환

        recommendations.append({
            "place_id": place_id,
            "type": place_types[0] if place_types else "unknown",
            "similarity": similarity_score
        })

    recommendations = sorted(recommendations, key=lambda x: x["similarity"], reverse=True)

    # JSON 파일 저장
    with open("recommendations.json", "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=4, ensure_ascii=False)
    
    return recommendations

# 실행 예시
user_keywords = ["조용한", "편안한", "카페"]
location = "37.5665,126.9780"
radius = 1000
query = "카페"

result = recommend_places(user_keywords, location, radius, query)
