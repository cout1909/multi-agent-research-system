"""
pipeline.py
The "supervisor" of our Multi-Agent Research System.
Connects ALL layers:
- Phase 1: Search → Reader → Writer → Critic (Agents)
- Phase 2: RAG + NLP + DL + ML (Analysis)
"""

from rich import print
from agents import search_agent, reader_agent, writer_chain, critic_chain
from nlp_analyzer import analyze_report
from dl_sentiment import analyze_sentiment_dl
from ml_scorer import predict_quality_score


def run_research_pipeline(topic: str, uploaded_docs_context: str = ""):
    """
    Runs the full research pipeline end to end.
    Returns a state dictionary with ALL results.
    """

    state = {"topic": topic}

    # -----------------------------------------------------
    # STEP 1: SEARCH AGENT
    # -----------------------------------------------------
    print("\n[bold cyan]STEP 1: Search Agent finding sources...[/bold cyan]")

    search_result = search_agent.invoke({
        "messages": [("user", f"Search for recent relevant sources about: {topic}")]
    })
    search_output = search_result["messages"][-1].content
    state["search_results"] = search_output
    print("[green]✅ Search Agent done[/green]")

    # -----------------------------------------------------
    # STEP 2: READER AGENT
    # -----------------------------------------------------
    print("\n[bold cyan]STEP 2: Reader Agent reading sources...[/bold cyan]")

    # Combine web search results + any uploaded document context
    combined_context = search_output
    if uploaded_docs_context:
        combined_context += f"\n\nFrom uploaded documents:\n{uploaded_docs_context}"

    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Here are sources for topic '{topic}':\n\n"
            f"{combined_context}\n\n"
            f"Extract the most important facts and insights."
        )]
    })
    reader_output = reader_result["messages"][-1].content
    state["research_notes"] = reader_output
    print("[green]✅ Reader Agent done[/green]")

    # -----------------------------------------------------
    # STEP 3: WRITER CHAIN
    # -----------------------------------------------------
    print("\n[bold cyan]STEP 3: Writer Chain writing report...[/bold cyan]")

    report = writer_chain.invoke({
    "topic": topic,
    "research_notes": reader_output,
    "document_context": uploaded_docs_context if uploaded_docs_context else ""
    })
    state["report"] = report
    print("[green]✅ Writer Chain done[/green]")

    # -----------------------------------------------------
    # STEP 4: CRITIC CHAIN
    # -----------------------------------------------------
    print("\n[bold cyan]STEP 4: Critic Chain reviewing report...[/bold cyan]")

    feedback = critic_chain.invoke({"report": report})
    state["critic_feedback"] = feedback
    print("[green]✅ Critic Chain done[/green]")

    # -----------------------------------------------------
    # STEP 5: NLP ANALYSIS
    # -----------------------------------------------------
    print("\n[bold cyan]STEP 5: NLP Analyzer extracting insights...[/bold cyan]")

    nlp_results = analyze_report(report)
    state["nlp_results"] = nlp_results
    print("[green]✅ NLP Analysis done[/green]")

    # -----------------------------------------------------
    # STEP 6: DEEP LEARNING SENTIMENT
    # -----------------------------------------------------
    print("\n[bold cyan]STEP 6: BERT Deep Learning sentiment analysis...[/bold cyan]")

    dl_results = analyze_sentiment_dl(report)
    state["dl_results"] = dl_results
    print("[green]✅ DL Sentiment done[/green]")

    # -----------------------------------------------------
    # STEP 7: ML QUALITY SCORING
    # -----------------------------------------------------
    print("\n[bold cyan]STEP 7: ML Scorer predicting quality...[/bold cyan]")

    ml_results = predict_quality_score(report, nlp_results)
    state["ml_results"] = ml_results
    print("[green]✅ ML Scoring done[/green]")

    print("\n[bold magenta]=== PIPELINE COMPLETE ===[/bold magenta]")
    return state


if __name__ == "__main__":
    result = run_research_pipeline("Latest developments in AI agents")

    print("\n[bold yellow]=== FINAL RESULTS ===[/bold yellow]")
    print(f"Topic:         {result['topic']}")
    print(f"Keywords:      {result['nlp_results']['keywords']}")
    print(f"NLP Sentiment: {result['nlp_results']['sentiment']}")
    print(f"DL Sentiment:  {result['dl_results']['dl_sentiment']}")
    print(f"DL Confidence: {result['dl_results']['dl_confidence']}")
    print(f"Quality Score: {result['ml_results']['ml_quality_score']}")
    print(f"Quality Label: {result['ml_results']['ml_quality_label']}")