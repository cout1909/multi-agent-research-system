import os
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain.tools import tool
from dotenv import load_dotenv
from rich import print 


# Load API keys from .env file
load_dotenv()
 
# Initialize Tavily client using our free API key
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
 
 
@tool
def search_tool(query: str) -> str:
    """
    Searches the web for the given query using Tavily
    and returns a list of relevant URLs with short snippets.
    Use this to find sources/links related to a research topic.
    """
    response = tavily_client.search(query=query, max_results=5)
 
    results = []
    for r in response.get("results", []):
        results.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}\n")
 
    return "\n---\n".join(results) if results else "No results found."
@tool
def scrape_tool(url: str) -> str:
    """
    Scrapes and extracts the main readable text content from a given URL
    using BeautifulSoup. Use this after finding a URL with search_tool,
    to read the actual content of that page.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove junk elements we don't want
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        # Limit length so we don't overload the LLM with huge pages
        return text[:3000]

    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

print(scrape_tool.invoke("https://www.bbc.com/news/topics/cx2jyv8j8gwt"))

