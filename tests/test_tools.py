from tools import search_tool, scrape_tool

def test_search_tool_exists():
    assert search_tool is not None

def test_scrape_tool_exists():
    assert scrape_tool is not None