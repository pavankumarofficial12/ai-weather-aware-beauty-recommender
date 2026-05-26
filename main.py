
import logging
import os
import math
import json
import asyncio
from typing import List, Dict, Any
from datetime import timedelta

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Query, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
import uvicorn

from config.settings import (
    DB_HOST,
    DB_USER,
    DB_PASSWORD,
    DB_PORT,
    DB_NAME2,
    DB_NAME3,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    BEDROCK_MODEL_ID,
    WEATHER_API_KEY,
    REDIS_URL  # ← Add this to your settings.py / .env
)

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
RECOMMENDATION_COUNT = 5
HIGH_QUALITY_SCORE = 70
MEDIUM_QUALITY_SCORE = 55

# Redis TTLs (seconds)
USER_RECOMMENDATION_CACHE_TTL = 300   # 5 minutes
WEATHER_CACHE_TTL = 600               # 10 minutes

# Rate limit: 5 requests per minute per IP
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Async Redis client
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# DATABASE (unchanged)
# -------------------------------------------------------------------
def get_db_connection(db_name: str):
    conn = psycopg2.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        dbname=db_name,
        cursor_factory=RealDictCursor
    )
    return conn

def fetch_user_profile(user_id: int) -> Dict[str, Any]:
    conn = get_db_connection(DB_NAME2)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_id, gender, latitude, longitude,
                       skin_tone, skin_type, scalp_type
                FROM user_profiles
                WHERE user_id = %s
            """, (user_id,))
            profile = cur.fetchone()
            return dict(profile) if profile else {}
    finally:
        conn.close()

def fetch_hair_analysis(user_id: int) -> Dict[str, Any]:
    conn = get_db_connection(DB_NAME2)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dandruff_detected, targeted_remedies
                FROM hair_analysis_results
                WHERE user_id = %s
            """, (user_id,))
            data = cur.fetchone()
            return dict(data) if data else {}
    finally:
        conn.close()

# -------------------------------------------------------------------
# WEATHER (now async + cached)
# -------------------------------------------------------------------
async def get_weather(lat: float, lon: float) -> Dict[str, Any]:
    cache_key = f"weather:{lat:.4f}:{lon:.4f}"
    cached = await redis_client.get(cache_key)
    if cached:
        logger.info("Weather cache hit")
        return json.loads(cached)

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    weather = {
        "location": data.get("name"),
        "description": data["weather"][0]["description"],
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"]
    }

    await redis_client.setex(cache_key, WEATHER_CACHE_TTL, json.dumps(weather))
    logger.info("Weather cached")
    return weather

# -------------------------------------------------------------------
# DISTANCE & PROVIDERS (unchanged)
# -------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def fetch_nearby_providers(user_lat, user_lon):
    conn = get_db_connection(DB_NAME3)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id AS service_provider_id,
                       latitude, longitude
                FROM service_providers
            """)
            providers = cur.fetchall()
            nearby = []
            for p in providers:
                if p["latitude"] is None or p["longitude"] is None:
                    continue
                dist = haversine(user_lat, user_lon, p["latitude"], p["longitude"])
                if dist <= 50:
                    p = dict(p)
                    p["distance_km"] = round(dist, 2)
                    nearby.append(p)
            return nearby
    finally:
        conn.close()

def fetch_services(provider_ids: List[int]):
    if not provider_ids:
        return []
    conn = get_db_connection(DB_NAME3)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    service_provider_id,
                    name,
                    service_type,
                    price,
                    is_active
                FROM services
                WHERE service_provider_id = ANY(%s)
                AND is_active = TRUE
            """, (provider_ids,))
            return [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()

# -------------------------------------------------------------------
# LLAMA SCORING (unchanged)
# -------------------------------------------------------------------
from botocore.config import Config


def llama_score_service(service, profile, weather):

    # ✅ Correct timeout configuration
    bedrock_config = Config(
        connect_timeout=30,
        read_timeout=30,
        retries={"max_attempts": 2}
    )

    bedrock = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
        config=bedrock_config   # ✅ correct way
    )

    prompt = f"""
You are scoring a beauty service relevance.

Weather:
Temperature: {weather['temp']}
Humidity: {weather['humidity']}
Wind: {weather['wind_speed']}
Condition: {weather['description']}

User:
Skin type: {profile.get('skin_type')}
Scalp type: {profile.get('scalp_type')}

Service:
Name: {service['name']}
Type: {service['service_type']}

Return ONLY a score between 0 and 100.
"""

    body = {
        "prompt": prompt,
        "max_gen_len": 20,
        "temperature": 0.2
    }

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body).encode("utf-8")
    )

    result = json.loads(response["body"].read())
    score_text = result.get("generation", "0").strip()

    try:
        score = float(''.join(c for c in score_text if c.isdigit()))
    except Exception:
        score = 0

    service["score"] = score
    return service
# -------------------------------------------------------------------
# FINAL SELECTION (unchanged)
# -------------------------------------------------------------------
def finalize_recommendations(scored_services, required_count):
    scored_services.sort(key=lambda x: x["score"], reverse=True)
    skin_services = [s for s in scored_services if s["service_type"].lower() == "skin"]
    hair_services = [s for s in scored_services if s["service_type"].lower() == "hair"]
    selected = []
    if skin_services:
        selected.append(skin_services.pop(0))
    if hair_services:
        selected.append(hair_services.pop(0))
    remaining_pool = [s for s in scored_services if s not in selected]
    for s in remaining_pool:
        if len(selected) >= required_count:
            break
        selected.append(s)
    for s in selected:
        s.pop("score", None)
    return selected

def select_best_services_parallel(services, profile, weather):
    scored = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(llama_score_service, s, profile, weather)
            for s in services
        ]
        for f in as_completed(futures):
            scored.append(f.result())
    return finalize_recommendations(scored, RECOMMENDATION_COUNT)

# -------------------------------------------------------------------
# API ENDPOINT — with caching, rate limit, async weather
# -------------------------------------------------------------------
@app.get("/recommend")
@limiter.limit("5/minute")  # 5 requests per minute per IP
async def recommend_services(user_id: int = Query(...), request: Request = None):
    # 1. Check user-specific cache first
    cache_key = f"recommend:user:{user_id}"
    cached_result = await redis_client.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for user {user_id}")
        return json.loads(cached_result)

    # 2. Fetch profile & hair (sync DB calls — acceptable for now)
    profile = fetch_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")

    hair_analysis = fetch_hair_analysis(user_id)

    lat = profile["latitude"]
    lon = profile["longitude"]

    # 3. Async weather call (cached)
    weather = await get_weather(lat, lon)

    # 4. Sync DB calls (nearby providers + services)
    nearby_providers = fetch_nearby_providers(lat, lon)
    provider_ids = [p["service_provider_id"] for p in nearby_providers]
    services = fetch_services(provider_ids)

    # Attach distances
    dist_map = {p["service_provider_id"]: p["distance_km"] for p in nearby_providers}
    for s in services:
        s["distance_km"] = dist_map.get(s["service_provider_id"])

    # 5. Parallel scoring (ThreadPoolExecutor — kept as-is)
    best_services = select_best_services_parallel(services, profile, weather)

    # 6. Final response
    response_data = {
        "user_id": user_id,
        "location": weather["location"],
        "temperature": weather["temp"],
        "humidity": weather["humidity"],
        "wind_speed": weather["wind_speed"],
        "Recommended_services": best_services
    }

    # 7. Cache the final result for 5 minutes
    await redis_client.setex(cache_key, 300, json.dumps(response_data))

    return response_data

# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
