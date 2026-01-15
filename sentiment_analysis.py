import requests
import json
import pandas as pd
from pyspark.sql import DataFrame

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def analyze_review_with_ollama(review_text: str, model_name: str = "llama3"):
    """
    Sends a single review to the local Ollama API for sentiment analysis.
    """
    prompt = f"""
    Analyze the sentiment of the following hotel review.
    Provide the output in JSON format with two keys:
    1. "sentiment": strictly one of ["Positive", "Negative", "Neutral"].
    2. "topics": a list of main topics mentioned (e.g., ["Service", "Food", "Room"]).
    
    Review: "{review_text}"
    
    Output JSON only. Do not add any explanation.
    """
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,  # Disabilita lo streaming, cioè non riceve una risposta in streaming
        "format": "json"  # Forza il formato JSON se il modello lo supporta (Ollama 0.1.28+)
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload) # effettua una richiesta POST all'API di Ollama
        response.raise_for_status() # solleva un'eccezione se la richiesta non ha successo
        result = response.json() # estrae il contenuto JSON dalla risposta
        response_text = result.get("response", "{}") # estrae il contenuto del campo "response" del JSON
        analysis = json.loads(response_text) # converte il contenuto del campo "response" in un oggetto Python
        return analysis
    except Exception as e:
        return {"sentiment": "Error", "topics": [], "error": str(e)}

def analyze_sentiment_sample(df: DataFrame, sample_size: int = 5, model_name: str = "llama3"):
    """
    Takes a Spark DataFrame, samples N rows, and performs Sentiment Analysis using Ollama.
    Returns a Pandas DataFrame suitable for Streamlit display.
    """
    print(f"Sampling {sample_size} reviews for sentiment analysis...")
    
    # 1. Estrae N righe dal DataFrame Spark
    # 2. Estrae righe che hanno testo in Positive o Negative review
    # 3. Costruisce un testo completo per l'analisi
    sampled_rows = df.filter("Negative_Review != 'No Negative' OR Positive_Review != 'No Positive'") \
                     .limit(sample_size * 2) \
                     .sample(withReplacement=False, fraction=1.0) \
                     .limit(sample_size) \
                     .collect()
    
    results = []
    
    for row in sampled_rows:
        neg = row["Negative_Review"]
        pos = row["Positive_Review"]
        
        if neg == "No Negative": neg = ""
        if pos == "No Positive": pos = ""
        
        full_text = f"{pos} {neg}".strip()
        if not full_text:
            continue
            
        print(f"Analyzing review: {full_text[:50]}...")
        analysis = analyze_review_with_ollama(full_text, model_name)
        
        results.append({
            "Hotel": row["Hotel_Name"],
            "Review Text": full_text,
            "Review Score": row["Reviewer_Score"],
            "LLM Sentiment": analysis.get("sentiment", "Unknown"),
            "LLM Topics": ", ".join(analysis.get("topics", [])),
            "Full Analysis": analysis
        })
        
    return pd.DataFrame(results)
