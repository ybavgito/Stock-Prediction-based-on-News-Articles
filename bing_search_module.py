from dotenv import load_dotenv
import requests
import os
from datetime import datetime

load_dotenv()  # Load variables from .env

NEWS_API_KEY = os.getenv('NEWSAPI_KEY')  # Replace with your actual key name
search_url = "https://newsapi.org/v2/everything"

def bing_search_news(company_name: str):
    params = {
        "q": f"{company_name} stock finance",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(search_url, params=params)
    response.raise_for_status()
    articles = response.json().get("articles", [])

    # Filter by current month/year
    current_month = datetime.now().month
    current_year = datetime.now().year

    filtered_articles = []
    for article in articles:
        published_at = article.get("publishedAt", "")
        if not published_at:
            continue
        try:
            published_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            if published_date.month == current_month and published_date.year == current_year:
                filtered_articles.append(article)
        except ValueError:
            continue  # Skip malformed date entries

    return filtered_articles