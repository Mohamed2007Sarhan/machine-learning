import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import time

def search_for_topic(query, max_results=20): 
    """
    Searches DuckDuckGo for the query and returns a list of URLs.
    Includes retry logic for timeouts.
    """
    print(f"Searching for: {query}...")
    results = []
    retries = 3
    
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                ddgs_gen = ddgs.text(query, max_results=max_results)
                if ddgs_gen:
                    results = [r['href'] for r in ddgs_gen]
                    break
        except Exception as e:
            print(f"Search attempt {attempt+1} failed: {e}")
            time.sleep(2) # Wait before retry
            
    return results

def aggregate_content(urls):
    """
    Scrapes text content from the provided URLs.
    """
    aggregated_text = ""
    # Process top 15 URLs
    initial_urls = urls[:15] if urls else []
    
    for url in initial_urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                text = "\n".join([p.get_text() for p in paragraphs])
                # Only add if substantial content
                if len(text) > 100:
                     aggregated_text += f"\n--- Source: {url} ---\n{text[:5000]}...\n"
        except Exception:
            pass 
            
    return aggregated_text

def interactive_approval(query):
    pass
