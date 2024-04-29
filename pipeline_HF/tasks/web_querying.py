import os
import json
import requests
from utils.web_utils import extract_website_name, scrape_text_from_website

# Make sure we don't scrape from known fact checking websites (that's cheating!)
SOURCE_BLACKLIST = ['politifact.org', 'factcheck.org']

def fetch_search_results(question, scrape_website=False):
    """
    Fetches search results for a given question using an API.

    Args:
        question (str): The question to search for.
        scrape_website (bool, optional): Whether to scrape the website content. Defaults to False.

    Returns:
        list: A list of organic search results.
    """
    api_key = os.environ.get('SERPER_API_KEY') # Replace with your actual API key

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    payload = json.dumps({"q": question})
    try:
        response = requests.post("https://google.serper.dev/search", headers=headers, data=payload)
        result = json.loads(response.text)

        # Extract the organic search results and transform them into our desired format
        results = []
        for item in result['organic']:
            # ALSO while iterating through the results, remove any websites on our source blacklist
            source = extract_website_name(item.get('link', ''))
            if source in SOURCE_BLACKLIST: continue
            website_text = scrape_text_from_website(item.get('link', '')) if scrape_website else item.get('snippet', '')
            if website_text is None or website_text == '':  # if we failed to scrape the website, use the snippet
                website_text = item.get('snippet', '')
            results.append({
                "title": item.get('title', ''),
                "source": source,
                "date_published": item.get('date', ''),
                "relevant_excerpt": item.get('snippet', ''),
                "text": website_text,
                "search_position": item.get('position', -1),
                "url": item.get('link', ''),
            })
        return results

    except Exception as e:
        print(f"Failed to fetch information: {e}")
        return []