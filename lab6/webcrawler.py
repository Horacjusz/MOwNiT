import aiohttp
import aiofiles
import asyncio
import os
import re
import shutil
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import time
import multiprocessing
from useful_functions import time_sentence

# Get the directory path of the current script
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
# Define the directory to save documents
DOCS_DIR = os.path.join(SCRIPT_PATH, "docs")

# Define shared variables for multiprocessing
docs_saved = multiprocessing.Value('i', 0)  # Shared value for the number of saved documents
lock = multiprocessing.Lock()  # Lock for shared value

# Function to fetch HTML content from a URL asynchronously
async def fetch_html(session, url):
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

# Function to extract text content from HTML
def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Remove script and style tags
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    # Clean up whitespace, tabs, and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\t+', '', text)
    return text.strip()

# Function to save document asynchronously
async def save_document(url, text):
    global docs_saved
    with lock:  # Lock access to docs_saved variable
        filename = f"document_{docs_saved.value}.txt"
        filepath = os.path.join(DOCS_DIR, filename)
        async with aiofiles.open(filepath, "w", encoding="utf-8") as file:
            await file.write(text)
            print(f"{docs_saved.value} -> Saved document {url} as {filename}")
            docs_saved.value += 1

# Function to find links in HTML content
async def find_links(html, base_url):
    links = set()
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a', href=True):
        next_url = link['href']
        try:
            full_url = urljoin(base_url, next_url)
            links.add(full_url)
        except ValueError:
            print("Invalid URL:", next_url)
            continue
    return links

# Function to crawl a single page
async def crawl_page(session, url):
    html = await fetch_html(session, url)
    if html:
        text = extract_text(html)
        await save_document(url, text)

# Function to crawl the web
async def crawl_web(start_urls, depth, max_documents):
    visited_urls = set()
    documents = []

    # Remove existing docs directory and create a new one
    if os.path.exists(DOCS_DIR):
        shutil.rmtree(DOCS_DIR)
    os.makedirs(DOCS_DIR)

    domains = []
    for url in start_urls:
        parse = urlparse(url)
        domains.append(parse.netloc)

    async with aiohttp.ClientSession() as session:
        while len(documents) < max_documents:
            if not start_urls:
                break
            url, current_depth = start_urls.pop(0), 0
            if url in visited_urls or current_depth > depth or urlparse(url).netloc not in domains:
                continue

            html = await fetch_html(session, url)
            if html:
                visited_urls.add(url)
                text = url + "\n" + extract_text(html)
                documents.append((url, text))

                await asyncio.gather(
                    save_document(url, text),
                )

                links = await find_links(html, url)

                for next_url in links:
                    if next_url in visited_urls:
                        continue
                    if next_url.startswith('http'):
                        start_urls.append(next_url)
                    current_depth += 1

    return documents

if __name__ == "__main__":
    # List of starting URLs to crawl
    start_urls = ["https://www.bbc.com/news", "https://www.foxnews.com", "https://news.sky.com"]
    # Maximum depth to crawl
    depth = 20
    # Maximum number of documents to save
    max_documents = 100
    
    # Measure total execution time
    start_time = time.time()
    asyncio.run(crawl_web(start_urls, depth, max_documents))
    end_time = time.time()
    total_time = end_time - start_time
    print(time_sentence(total_time, header="Total time:"))