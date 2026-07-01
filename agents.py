"""
agents.py
Builds the 4 core pieces of our Multi-Agent Research System:
1. Search Agent  -> uses create_agent() + search_tool
2. Reader Agent  -> uses create_agent() + scrape_tool
3. Writer Chain  -> LCEL pipeline (prompt | llm | StrOutputParser)
4. Critic Chain  -> LCEL pipeline (prompt | llm | StrOutputParser)
"""

import os
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from rich import print

from tools import search_tool, scrape_tool

# Load API keys from .env
load_dotenv()

# ---------------------------------------------------------
# Shared LLM instance (the "brain") used by everything below
# ---------------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# ---------------------------------------------------------
# 1. SEARCH AGENT
# Just pass the model + tools -> create_agent builds a
# ready-to-use agent graph (powered by LangGraph internally)
# ---------------------------------------------------------
search_agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt="You are a research assistant. Use the search_tool to find "
                   "relevant, recent sources for the user's research topic."
)


# ---------------------------------------------------------
# 2. READER AGENT
# Same create_agent() pattern, but with the scrape_tool
# ---------------------------------------------------------
reader_agent = create_agent(
    model=llm,
    tools=[scrape_tool],
    system_prompt="You are a research assistant. Use the scrape_tool to read "
                   "the full content of the given URLs and extract the key facts."
)


# ---------------------------------------------------------
# 3. WRITER CHAIN (LCEL: prompt | llm | StrOutputParser)
# Takes all the gathered research and writes a full report
# ---------------------------------------------------------
writer_prompt = ChatPromptTemplate.from_template(
    """You are a professional research writer.

Topic: {topic}

Research notes (from web):
{research_notes}

Uploaded document content:
{document_context}

Instructions:
- If document_context is empty: write a normal research report from web findings only
- If document_context has content: 
  * Write the report combining both sources
  * Explicitly mention what the document says vs what web says
  * Point out outdated information in the document
  * Highlight new developments not present in the document
  * Use phrases like "According to your document..." and "However, latest research shows..."

Write a clear, well structured report with introduction, key findings, and conclusion."""
)

writer_chain = writer_prompt | llm | StrOutputParser()

# ---------------------------------------------------------
# 4. CRITIC CHAIN (LCEL: prompt | llm | StrOutputParser)
# Reads the report and gives a score + feedback
# ---------------------------------------------------------
critic_prompt = ChatPromptTemplate.from_template(
    """You are a senior research editor reviewing a junior researcher's report.

Report:
{report}

Review this report. Give:
1. A quality score out of 10
2. Specific feedback on what is good
3. Specific feedback on what could be improved"""
)

critic_chain = critic_prompt | llm | StrOutputParser()


# ---------------------------------------------------------
# Quick test when running this file directly
# ---------------------------------------------------------
if __name__ == "__main__":
    topic = "Latest developments in AI agents"

    print("[bold cyan]Running Search Agent...[/bold cyan]")
    search_result = search_agent.invoke({
        "messages": [{"role": "user", "content": f"Search for: {topic}"}]
    })
    print(search_result["messages"][-1].content)