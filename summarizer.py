# Imports

import ollama
import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# Headers for web scraping
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    def __init__(self, url):
        """
        Initialize Website object to scrape and process content from a URL
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for tag in soup.body(["script", "style", "img", "input"]):
            tag.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

system_prompt = (
    "You are an assistant that analyzes the contents of a website "
    "and provides a short summary, ignoring text that might be navigation related. "
    "Respond in markdown."
)

def user_prompt_for(website):
    """
    Generate a prompt for the AI summarizer based on the website content.
    """
    return (
        f"You are looking at a website titled {website.title}\n"
        "The contents of this website are as follows; please provide a short summary.\n\n"
        f"{website.text}"
    )

def messages_for(website):
    """
    Create a message structure for the AI summarizer.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

def summarize_with_ollama(url):
    """
    Generate a summary for a given URL using the Ollama local model.
    """
    website = Website(url)
    messages = messages_for(website)
    response = ollama.chat(model="llama3.2", messages=messages)
    return response['message']['content']

def display_summary_with_ollama(url):
    """
    Fetch and display the summary of a URL using Ollama.
    """
    summary = summarize_with_ollama(url)
    display(Markdown(summary))

# Example usage:
if __name__ == "__main__":
    test_url = "https://example.com"
    display_summary_with_ollama(test_url)
