"""
nlp_analyzer.py
NLP layer for our Multi-Agent Research System.
Takes the final research report and extracts:
1. Keywords     → top 10 most important words
2. Sentiment    → positive / negative / neutral tone
3. Word count   → how long is the report
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from textblob import TextBlob

# ---------------------------------------------------------
# Ensure required NLTK data is available.
# Streamlit Cloud spins up a fresh environment on every deploy,
# so this data isn't pre-downloaded like it might be on your
# local machine. This checks first, and only downloads if missing.
# ---------------------------------------------------------
for resource, path in [
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)


def analyze_report(report: str) -> dict:
    """
    Takes the research report text and runs NLP analysis on it.
    Returns a dictionary with all the results.
    """

    # ---------------------------------------------------------
    # STEP 1: KEYWORD EXTRACTION (NLP)
    # ---------------------------------------------------------

    # Tokenize → split report into individual words
    tokens = word_tokenize(report.lower())

    # Remove stopwords (common words like "the", "is", "and" that aren't useful)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        word for word in tokens
        if word.isalpha()        # only keep actual words (no punctuation/numbers)
        and word not in stop_words  # remove stopwords
        and len(word) > 3       # remove very short words
    ]

    # Find most common words → these are your keywords
    freq_dist = FreqDist(filtered_tokens)
    top_keywords = [word for word, freq in freq_dist.most_common(10)]

    # ---------------------------------------------------------
    # STEP 2: SENTIMENT ANALYSIS (NLP)
    # ---------------------------------------------------------

    # TextBlob gives us a sentiment score:
    # polarity → -1.0 (very negative) to +1.0 (very positive)
    # subjectivity → 0.0 (very objective) to 1.0 (very subjective)
    blob = TextBlob(report)
    polarity = round(blob.sentiment.polarity, 2)
    subjectivity = round(blob.sentiment.subjectivity, 2)

    # Convert polarity number to human readable label
    if polarity > 0.1:
        sentiment_label = "Positive 😊"
    elif polarity < -0.1:
        sentiment_label = "Negative 😔"
    else:
        sentiment_label = "Neutral 😐"

    # ---------------------------------------------------------
    # STEP 3: BASIC STATS
    # ---------------------------------------------------------
    word_count = len(tokens)
    sentence_count = len(nltk.sent_tokenize(report))

    # ---------------------------------------------------------
    # RETURN ALL RESULTS AS A DICTIONARY
    # ---------------------------------------------------------
    return {
        "keywords": top_keywords,
        "sentiment": sentiment_label,
        "polarity_score": polarity,
        "subjectivity_score": subjectivity,
        "word_count": word_count,
        "sentence_count": sentence_count
    }


# Quick test when running this file directly
if __name__ == "__main__":
    sample_report = """
    Artificial intelligence agents are rapidly transforming the technology landscape.
    Recent developments show that multi-agent systems are becoming increasingly powerful
    and sophisticated. These systems can search the web, analyze data, and write detailed
    reports autonomously. The future of AI looks incredibly promising with breakthroughs
    happening every week. However, there are concerns about safety and ethical implications
    that need careful consideration.
    """

    results = analyze_report(sample_report)

    print("=== NLP ANALYSIS RESULTS ===")
    print(f"Keywords:     {results['keywords']}")
    print(f"Sentiment:    {results['sentiment']}")
    print(f"Polarity:     {results['polarity_score']}")
    print(f"Subjectivity: {results['subjectivity_score']}")
    print(f"Word Count:   {results['word_count']}")
    print(f"Sentences:    {results['sentence_count']}")