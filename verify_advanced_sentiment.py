from main import create_spark_session, load_data
from sentiment_analysis import (
    fetch_hotel_reviews, 
    enrich_reviews_with_llm,
    get_negative_topic_frequency,
    get_topic_trends,
    get_topic_score_correlation,
    get_discrepancies
)
import pandas as pd

def test_advanced_insights():
    print("--- 1. INIT SPARK ---")
    spark = create_spark_session("Verification")
    df = load_data(spark)
    if not df:
        print("Failed to load data.")
        return

    # Use "Ritz Paris" as default, but limit to very few reviews for speed
    hotel_name = "Ritz Paris"
    print(f"--- 2. FETCH REVIEWS ({hotel_name}) ---")
    raw_df = fetch_hotel_reviews(df, hotel_name, limit=3)
    print(f"Fetched {len(raw_df)} reviews.")
    if raw_df.empty:
        print("No reviews found. Verification Aborted.")
        return

    print("--- 3. MOCK ENRICHMENT (to save LLM time) ---")
    # We manually create what the LLM would return to test the aggregation logic separate from LLM connection
    # This ensures we test the Insight Logic even if Ollama is slow/offline
    mock_data = [
        {
            "Review_Date": "8/3/2017", "Reviewer_Score": 9.0, "Review_Text": "Great location but noisy.", 
            "LLM_Sentiment": "Negative", "LLM_Topics": ["Location", "Noise"]
        },
        {
            "Review_Date": "7/15/2017", "Reviewer_Score": 5.0, "Review_Text": "Terrible service.", 
            "LLM_Sentiment": "Negative", "LLM_Topics": ["Service"]
        },
        {
            "Review_Date": "6/10/2017", "Reviewer_Score": 10.0, "Review_Text": "Perfect.", 
            "LLM_Sentiment": "Positive", "LLM_Topics": ["Everything"]
        },
        {
            "Review_Date": "8/4/2017", "Reviewer_Score": 8.5, "Review_Text": "Good but expensive.", 
            "LLM_Sentiment": "Negative", "LLM_Topics": ["Value"]
        }
    ]
    enriched_df = pd.DataFrame(mock_data)
    print("Masked Enriched DataFrame Created.")

    print("--- 4. TEST INSIGHT 1: Negative Freq ---")
    freq = get_negative_topic_frequency(enriched_df)
    print(freq)

    print("--- 5. TEST INSIGHT 2: Trends ---")
    trends = get_topic_trends(enriched_df)
    print(trends)

    print("--- 6. TEST INSIGHT 3: Correlation ---")
    impact, avg = get_topic_score_correlation(enriched_df)
    print(f"Global Avg: {avg}")
    print(impact)

    print("--- 7. TEST INSIGHT 4: Discrepancies ---")
    disc = get_discrepancies(enriched_df, score_threshold=8.0)
    print(disc)
    
    spark.stop()
    print("\nVERIFICATION COMPLETE: All functions executed without error.")

if __name__ == "__main__":
    test_advanced_insights()
