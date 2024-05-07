import re
import requests
from bs4 import BeautifulSoup

def extract_website_name(url):
    """Extracts the website name from a given URL using regex"""
    match = re.search(r'(?P<url>https?://[^\s]+)', url)
    if match:
        url = match.group('url')
        return url.split('//')[1].split('/')[0].lower().replace('www.', '')
    return None

def scrape_text_from_website(url):
    """Scrapes text and metadata from a given website URL."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract all text from the website
            text = soup.get_text()

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        else:
            print(f"Failed to retrieve content from the URL: {url}")
            return None
    except Exception as e:
        print(f"Error during website scraping: {e}")
        return None