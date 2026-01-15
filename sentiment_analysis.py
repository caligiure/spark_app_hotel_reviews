import requests
import json
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from datetime import datetime

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def fetch_hotel_reviews(df: DataFrame, hotel_name: str, limit: int = 50):
    """
    Fetches reviews for a specific hotel from Spark.
    Returns a Pandas DataFrame.
    """
    print(f"Fetching reviews for hotel: {hotel_name}...")
    # Select relevant columns
    hotel_df = df.filter(col("Hotel_Name") == hotel_name) \
                 .select("Review_Date", "Reviewer_Score", "Negative_Review", "Positive_Review") \
                 .limit(limit)
    
    return hotel_df.toPandas()

def analyze_review_with_ollama(review_text: str, model_name: str = "llama3"):
    """
    Sends a single review to the local Ollama API.
    """
    prompt = f"""
    Analyze the sentiment of the following hotel review.
    Provide the output in JSON format with two keys:
    1. "sentiment": strictly one of ["Positive", "Negative", "Neutral"].
    2. "topics": a list of main topics mentioned (e.g., ["Service", "Food", "Room", "Booking", "Location"]).
    
    Review: "{review_text}"
    
    Output JSON only. Do not add any explanation.
    """
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        analysis = json.loads(result.get("response", "{}"))
        return analysis
    except Exception as e:
        return {"sentiment": "Error", "topics": [], "error": str(e)}

def enrich_reviews_with_llm(pandas_df: pd.DataFrame, model_name: str = "llama3"):
    """
    Iterates over the reviews and adds LLM analysis (Sentiment + Topics).
    """
    results = []
    print(f"Enriching {len(pandas_df)} reviews using {model_name}...")
    
    for index, row in pandas_df.iterrows():
        neg = row["Negative_Review"]
        pos = row["Positive_Review"]
        if neg == "No Negative": neg = ""
        if pos == "No Positive": pos = ""
        full_text = f"{pos} {neg}".strip()
        
        if not full_text:
            continue
            
        analysis = analyze_review_with_ollama(full_text, model_name)
        
        results.append({
            "Review_Date": row["Review_Date"],
            "Reviewer_Score": row["Reviewer_Score"],
            "Review_Text": full_text,
            "LLM_Sentiment": analysis.get("sentiment", "Unknown"),
            "LLM_Topics": analysis.get("topics", [])
        })
        
    return pd.DataFrame(results)

# --- QUERY 1: Parole Chiave (Topic) più frequenti nelle recensioni Negative ---
def get_negative_topic_frequency(enriched_df: pd.DataFrame):
    # Filter negative sentiment
    neg_reviews = enriched_df[enriched_df["LLM_Sentiment"] == "Negative"]
    
    # Flatten topics
    all_topics = []
    for topics in neg_reviews["LLM_Topics"]:
        all_topics.extend(topics)
        
    if not all_topics:
        return pd.DataFrame(columns=["Topic", "Frequency"])
        
    return pd.DataFrame(all_topics, columns=["Topic"]).value_counts().reset_index(name="Frequency")

# --- QUERY 2: Trend Temporali dei Topic ---
def get_topic_trends(enriched_df: pd.DataFrame):
    # Convert 'Review_Date' (usually string "M/D/YYYY") to datetime
    # Dataset format example: "8/3/2017"
    try:
        enriched_df["Date_Obj"] = pd.to_datetime(enriched_df["Review_Date"], format='mixed', dayfirst=True)
    except:
        # Fallback if mixed formats
        enriched_df["Date_Obj"] = pd.to_datetime(enriched_df["Review_Date"], infer_datetime_format=True, errors='coerce')

    enriched_df["Month_Year"] = enriched_df["Date_Obj"].dt.to_period('M').astype(str)

    # Explode topics so each topic has its own row effectively (aggregating)
    exploded = enriched_df.explode("LLM_Topics")
    
    # Group by Month and Topic
    trends = exploded.groupby(["Month_Year", "LLM_Topics"]).size().reset_index(name="Count")
    return trends

# --- QUERY 3: Correlazione Topic-Voto ---
def get_topic_score_correlation(enriched_df: pd.DataFrame):
    global_avg = enriched_df["Reviewer_Score"].mean()
    
    exploded = enriched_df.explode("LLM_Topics")
    # Avg score per topic
    topic_stats = exploded.groupby("LLM_Topics")["Reviewer_Score"].agg(["mean", "count"]).reset_index()
    # Filter only topics appearing at least a few times to be significant
    topic_stats = topic_stats[topic_stats["count"] >= 2] 
    
    topic_stats["Impact"] = topic_stats["mean"] - global_avg
    topic_stats = topic_stats.sort_values("Impact")
    
    return topic_stats, global_avg

# --- QUERY 4: Dissonanze (Voto Alto ma Sentiment Negativo) ---
def get_discrepancies(enriched_df: pd.DataFrame, score_threshold: float = 8.0):
    discrepancies = enriched_df[
        (enriched_df["Reviewer_Score"] >= score_threshold) & 
        (enriched_df["LLM_Sentiment"] == "Negative")
    ]
    return discrepancies[["Review_Date", "Reviewer_Score", "LLM_Sentiment", "Review_Text"]]
