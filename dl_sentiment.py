"""
dl_sentiment.py
Deep Learning layer for our Multi-Agent Research System.
Uses a pre-trained BERT model from HuggingFace to perform
sentiment analysis on the research report.

Why BERT and not TextBlob (what we used in nlp_analyzer.py)?
- TextBlob = rule-based NLP (old school, simple)
- BERT = Deep Learning transformer model (modern, much more accurate)
  BERT actually UNDERSTANDS context, not just individual words
"""

from transformers import pipeline

# ---------------------------------------------------------
# Load the pre-trained BERT sentiment analysis model
# This runs completely locally - no API key needed!
# First run downloads the model (~270MB), then it's cached forever
# ---------------------------------------------------------
print("Loading BERT model... (first time takes a minute)")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("BERT model loaded! ✅")


def analyze_sentiment_dl(text: str) -> dict:
    """
    Uses BERT (Deep Learning) to analyze sentiment of the given text.
    Much more accurate than rule-based NLP sentiment analysis.
    Returns label (POSITIVE/NEGATIVE) and confidence score.
    """

    # BERT has a token limit - if text is too long, we truncate it
    # (we take first 512 words which is BERT's max input size)
    truncated_text = " ".join(text.split()[:400])

    # Run the BERT model on the text
    result = sentiment_pipeline(truncated_text)[0]

    # result looks like: {"label": "POSITIVE", "score": 0.9998}
    label = result["label"]           # POSITIVE or NEGATIVE
    confidence = round(result["score"] * 100, 2)  # convert to percentage

    # Add emoji for readability
    emoji = "😊" if label == "POSITIVE" else "😔"

    return {
        "dl_sentiment": f"{label} {emoji}",
        "dl_confidence": f"{confidence}%",
        "model_used": "distilbert-base-uncased-finetuned-sst-2-english"
    }


# Quick test when running this file directly
if __name__ == "__main__":
    test_texts = [
        "This research report is excellent and very informative. The findings are groundbreaking.",
        "The report is poorly written and lacks evidence. The conclusions are weak and unconvincing.",
        "AI agents are becoming more common in various industries and applications."
    ]

    print("\n=== BERT DEEP LEARNING SENTIMENT ANALYSIS ===\n")
    for text in test_texts:
        result = analyze_sentiment_dl(text)
        print(f"Text: {text[:60]}...")
        print(f"Sentiment:  {result['dl_sentiment']}")
        print(f"Confidence: {result['dl_confidence']}")
        print(f"Model:      {result['model_used']}")
        print("---")