import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import re
import string

def scrape_news_article(url):
    """
    Scrapes the text content from a news article URL.

    Args:
        url (str): The URL of the news article.

    Returns:
        str: The text content of the article, or None if an error occurs.
    """
    try:
        response = requests.get(url, timeout=10)  # Add timeout for robustness
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find the main article text.  This is highly site-specific and
        # may need to be adjusted.  Here's a common pattern, and some fallbacks.
        article_text_elements = soup.find_all('p')  #  Find all <p> tags
        if not article_text_elements:
            article_text_elements = soup.find_all('div', class_='article-body__content') # common class name
        if not article_text_elements:
            article_text_elements = soup.find_all('div', attrs={'itemprop': 'articleBody'})
        if not article_text_elements:
            return None #could not find any relevant tags
        article_text = ' '.join([p.get_text() for p in article_text_elements])
        article_text = re.sub(r'\s+', ' ', article_text).strip()  # Clean up extra spaces
        return article_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"Error processing article: {e}")
        return None

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using NLTK's VADER.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary containing the sentiment scores (positive, negative, neutral, compound).
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores

def generate_human_like_report(article_title, sentiment_scores, source_url):
    """
    Generates a human-like report summarizing the sentiment analysis.

    Args:
        article_title (str): The title of the article.
        sentiment_scores (dict): The sentiment scores.
        source_url(str): the url of the article

    Returns:
        str: A formatted report.
    """
    # Use more natural language and varied sentence structure.
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    neutral_score = sentiment_scores['neu']
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        sentiment_label = "positive"
        overall_feeling = "The article conveys a generally positive sentiment."
    elif compound_score <= -0.05:
        sentiment_label = "negative"
        overall_feeling = "The article expresses a predominantly negative sentiment."
    else:
        sentiment_label = "neutral"
        overall_feeling = "The article presents a largely neutral perspective."

    report = (
        f"Sentiment Analysis Report: {article_title}\n"
        f"Source Article URL: {source_url}\n\n"  # Include source URL
        "This report details the sentiment analysis performed on the text of the news article. "
        "The analysis aimed to determine the overall emotional tone and perspective conveyed by the writing.\n\n"
        "Key Sentiment Indicators:\n"
        f"- Positive sentiment: {positive_score:.3f}\n"
        f"- Negative sentiment: {negative_score:.3f}\n"
        f"- Neutral sentiment: {neutral_score:.3f}\n"
        f"- Compound sentiment score: {compound_score:.3f}\n\n"
        f"{overall_feeling}  Specifically:\n" #added transition
        f"The positive sentiment is measured at {positive_score:.3f}, indicating the presence of favorable language. "
        f"Conversely, the negative sentiment is {negative_score:.3f}, showing the extent of unfavorable expressions. "
        f"The neutral sentiment, with a score of {neutral_score:.3f}, reflects the portion of the text that lacks strong emotional coloring.\n\n"
        "In conclusion, the sentiment analysis suggests that the article's tone is primarily {sentiment_label}. "
        "It's important to note that sentiment analysis provides a general overview, and subtle nuances in human language might not be fully captured.\n"
        "The compound score provides a single metric summarizing the overall sentiment, but the individual positive, negative, and neutral scores offer a more detailed understanding."
    )
    return report

def main():
    """
    Main function to orchestrate the scraping, analysis, and reporting.
    """
    # 1. Choose a News Website and Article URL
    #  I've picked a specific article for consistency.  You could make this a user input.
    news_article_url = "https://www.bbc.com/news/world-us-canada-68840552"  # Example URL
    article_title = "Baltimore bridge collapse: Search for survivors paused" #added title

    # 2. Scrape the Article Text
    article_text = scrape_news_article(news_article_url)
    if article_text is None:
        print("Failed to retrieve article text. Exiting.")
        return

    # 3. Analyze the Sentiment
    sentiment_scores = analyze_sentiment(article_text)

    # 4. Generate the Report
    report = generate_human_like_report(article_title, sentiment_scores, news_article_url)
    print(report)

    # 5.  Demonstrate code used (optional, but good for transparency)
    print("\n" + "="*40 + "\n" + "Code Used:\n" + "="*40 + "\n")
    import inspect
    print(inspect.getsource(scrape_news_article))
    print(inspect.getsource(analyze_sentiment))
    print(inspect.getsource(generate_human_like_report))
    print(inspect.getsource(main)) # Include the main function

if __name__ == "__main__":
    main()