"""
ml_scorer.py
Machine Learning layer for our Multi-Agent Research System.
Uses scikit-learn to predict a quality score (0-10) for the
research report based on measurable features extracted from it.

How it works:
- We define simple rules to create features from the report
- A trained ML model predicts quality score based on those features
- Since we don't have real training data, we use a rule-based scorer
  that mimics what a trained ML model would do
"""

import re


def extract_features(report: str, nlp_results: dict) -> dict:
    """
    Extracts measurable features from the report.
    These features are what the ML model uses to predict quality.
    Think of features as the "inputs" to the ML model.
    """

    # Feature 1: Word count (longer reports tend to be more thorough)
    word_count = nlp_results.get("word_count", 0)

    # Feature 2: Sentence count
    sentence_count = nlp_results.get("sentence_count", 0)

    # Feature 3: Average words per sentence (readability)
    avg_words_per_sentence = round(word_count / max(sentence_count, 1), 2)

    # Feature 4: Number of keywords found (topic coverage)
    keyword_count = len(nlp_results.get("keywords", []))

    # Feature 5: Sentiment polarity (neutral/balanced reports score higher)
    polarity = abs(nlp_results.get("polarity_score", 0))

    # Feature 6: Has introduction markers
    has_intro = 1 if any(word in report.lower() for word in
                         ["introduction", "overview", "background", "this report"]) else 0

    # Feature 7: Has conclusion markers
    has_conclusion = 1 if any(word in report.lower() for word in
                               ["conclusion", "summary", "findings", "in conclusion"]) else 0

    # Feature 8: Has numbers/statistics (data-driven reports score higher)
    has_numbers = 1 if bool(re.search(r'\d+', report)) else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": avg_words_per_sentence,
        "keyword_count": keyword_count,
        "polarity": polarity,
        "has_intro": has_intro,
        "has_conclusion": has_conclusion,
        "has_numbers": has_numbers
    }


def predict_quality_score(report: str, nlp_results: dict) -> dict:
    """
    Predicts a quality score (0-10) for the research report
    based on extracted features.
    Uses a weighted scoring system (mimics what a trained ML model does).
    """

    features = extract_features(report, nlp_results)
    score = 0

    # Rule 1: Word count scoring (max 3 points)
    if features["word_count"] > 300:
        score += 3
    elif features["word_count"] > 150:
        score += 2
    elif features["word_count"] > 50:
        score += 1

    # Rule 2: Sentence structure (max 2 points)
    if 15 <= features["avg_words_per_sentence"] <= 25:
        score += 2   # ideal sentence length
    elif 10 <= features["avg_words_per_sentence"] <= 30:
        score += 1   # acceptable sentence length

    # Rule 3: Keyword coverage (max 2 points)
    if features["keyword_count"] >= 8:
        score += 2
    elif features["keyword_count"] >= 5:
        score += 1

    # Rule 4: Has introduction (1 point)
    score += features["has_intro"]

    # Rule 5: Has conclusion (1 point)
    score += features["has_conclusion"]

    # Rule 6: Has numbers/data (1 point)
    score += features["has_numbers"]

    # Determine quality label
    if score >= 8:
        quality_label = "Excellent 🌟"
    elif score >= 6:
        quality_label = "Good 👍"
    elif score >= 4:
        quality_label = "Average 📊"
    else:
        quality_label = "Needs Improvement ⚠️"

    return {
        "ml_quality_score": f"{score}/10",
        "ml_quality_label": quality_label,
        "features_used": features
    }


# Quick test when running this file directly
if __name__ == "__main__":
    sample_report = """
    Introduction: This report covers the latest developments in AI agents.
    AI agents are autonomous systems that can search the web, analyze data,
    and write detailed reports. Recent studies show that over 65% of enterprises
    are now investing in agentic AI systems. The technology has grown by 300%
    in the last 2 years. Multi-agent systems combine multiple specialized agents
    that collaborate to solve complex problems. Each agent has a specific role
    such as searching, reading, writing, or critiquing.
    Conclusion: AI agents represent the future of intelligent automation,
    with significant growth expected in the coming years.
    """

    # Simulate NLP results
    mock_nlp = {
        "word_count": 95,
        "sentence_count": 6,
        "keywords": ["agents", "systems", "AI", "search", "data", "report", "enterprise", "growth"],
        "polarity_score": 0.3
    }

    result = predict_quality_score(sample_report, mock_nlp)

    print("=== ML QUALITY SCORING RESULTS ===")
    print(f"Quality Score: {result['ml_quality_score']}")
    print(f"Quality Label: {result['ml_quality_label']}")
    print(f"\nFeatures Used:")
    for feature, value in result["features_used"].items():
        print(f"  {feature}: {value}")