import requests

API_KEY = "d5826921190d441ab59fea4ad438c020"

def get_news(ticker):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q=Apple&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    )

    response = requests.get(url)

    data = response.json()

    return data["articles"][:5]