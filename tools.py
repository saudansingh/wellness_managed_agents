import os
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from langchain_core.tools import tool
from google import genai

# ---------------------------------------------------------
# Authentication Configuration
# ---------------------------------------------------------
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "YOUR_GOOGLE_SEARCH_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX", "YOUR_CUSTOM_SEARCH_ENGINE_ID")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_SEARCH_API_KEY")

# =========================================================
# TOOL 1: YouTube Search Tool (Optimized for Speed)
# =========================================================
@tool
def search_youtube_videos(query: str, max_results: int = 2) -> str:
    """
    Searches YouTube for highly authoritative instructional or exercise videos.
    Returns a clean, raw text block containing the exact titles and URLs for the AI agent to synthesize.
    
    Args:
        query (str): The specific exercise or yoga pose movement phrase.
        max_results (int): Hard-capped to 2 for maximum latency reductions.
    """
    try:
        # Initialize client with minimized metadata request parameters
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
        
        request = youtube.search().list(
            q=query,
            part="id,snippet",
            type="video",
            maxResults=max_results
        )
        response = request.execute()
        
        video_results = []
        for item in response.get("items", []):
            title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_results.append(f"VIDEO_TITLE: {title} | VIDEO_URL: {video_url}")
            
        if not video_results:
            return f"No verified video assets found for query: '{query}'."
            
        return "\n".join(video_results)
        
    except Exception as e:
        return f"Error executing YouTube search tool: {str(e)}"

# =========================================================
# TOOL 2: Web Search & Scraping Tool (Optimized for Safety/Speed)
# =========================================================
@tool
def search_and_scrape_recipe(query: str) -> str:
    """
    Queries premium custom search engine sources for clinical or culinary data,
    then automatically parses structural elements from the text content.
    
    Args:
        query (str): Detailed macro or dietary query targeting high-quality recipes.
    """
    try:
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_SEARCH_API_KEY,
            "cx": GOOGLE_CX,
            "q": query,
            "num": 1
        }
        search_response = requests.get(search_url, params=params).json()
        
        items = search_response.get("items", [])
        if not items:
            return f"Could not isolate high-quality culinary databases for query: '{query}'."
            
        target_link = items[0]["link"]
        site_title = items[0]["title"]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        page_response = requests.get(target_link, headers=headers, timeout=4)
        
        if page_response.status_code != 200:
            return f"Resource found ({target_link}) but downstream socket refused content extraction."
            
        soup = BeautifulSoup(page_response.text, "html.parser")
        
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
            element.decompose()
            
        text_blocks = []
        for tag in soup.find_all(['p', 'li', 'h1', 'h2', 'h3']):
            text = tag.get_text().strip()
            if len(text) > 30:
                text_blocks.append(text)
                
        raw_scraped_body = "\n".join(text_blocks)
        
        # 🛠️ INTEGRATION: Safely prune the scraped content using token-level constraints before returning
        scraped_body = dynamic_token_chunker(raw_scraped_body, max_tokens=1200)
        
        output = (
            f"SOURCE_WEBSITE: {site_title}\n"
            f"SOURCE_URL: {target_link}\n\n"
            f"RAW_SCRAPED_DATA:\n{scraped_body}"
        )
        return output
        
    except Exception as e:
        return f"Error executing recipe scraper tool: {str(e)}"
    
def dynamic_token_chunker(raw_text: str, model_name: str = "gemini-2.5-flash", max_tokens: int = 1500) -> str:
    """
    Safely counts and trims strings down to strict structural token allocations using modern google-genai APIs.
    """
    if not API_KEY:
        return raw_text[:2000] # Fallback to rough character slice if API keys are decoupled
        
    try:
        # Corrected modern client initialization
        client = genai.Client(api_key=API_KEY)
        sentences = raw_text.split(". ")
        current_chunk = []
        
        for sentence in sentences:
            test_string = ". ".join(current_chunk + [sentence])
            
            # Correct client-side API model call syntax standard
            response = client.models.count_tokens(
                model=model_name,
                contents=test_string,
            )
            token_count = response.total_tokens
            
            if token_count < max_tokens:
                current_chunk.append(sentence)
            else:
                break 
                
        return ". ".join(current_chunk)
    except Exception:
        return raw_text[:2000] # Safe fallback string escape block
