from sentiment_analysis import analyze_review_with_ollama
import sys

def test_ollama():
    print("Testing connection to Ollama (localhost:11434)...")
    
    test_review = "The hotel was absolutely wonderful, the staff was friendly and the breakfast was delicious. However, the room was a bit small."
    print(f"Test Review: '{test_review}'")
    
    # We use "llama3" as default, but fallback or user might have "mistral"
    # We will try llama3 first
    model = "llama3"
    
    print(f"Sending request to model '{model}'...")
    result = analyze_review_with_ollama(test_review, model_name=model)
    
    print("\n--- RESULT ---")
    print(result)
    
    if result.get("sentiment") in ["Positive", "Negative", "Neutral"]:
        print("\nSUCCESS: Ollama responded with valid JSON sentiment.")
    else:
        print("\nWARNING: Unexpected response format or connection error.")
        print(f"Error details: {result.get('error')}")

if __name__ == "__main__":
    test_ollama()
