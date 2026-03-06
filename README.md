# Scraper, Embedding, and Reranker Service

A FastAPI-based service that provides endpoints for web scraping (using Selenium with stealth capabilities), generating text embeddings, and reranking documents based on relevance.

## Features

- **Web Scraper**: 
  - Headless Chrome scraping using Selenium.
  - Stealth mode to avoid bot detection (`selenium-stealth`).
  - Proxy support with random rotation from `proxies.txt`.
  - Human-like interaction (scrolling, random delays).
  - Extract page title, description, content, links, and optionally download/base64-encode images.
- **Embeddings**:
  - Generate high-quality text embeddings using `jinaai/jina-embeddings-v3`.
  - Supports GPU acceleration if CUDA is available.
- **Reranker**:
  - Rerank documents for better retrieval relevance using `jinaai/jina-reranker-v2-base-multilingual`.
  - Provides relevance scores for query-document pairs.

## Prerequisites

- Python 3.8+
- Chrome Browser installed.
- [ChromeDriver](https://googlechromelabs.github.io/chrome-for-testing/) (ensure the path in `scraper.py` matches your local installation).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd scraper-embedding-reranker
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Proxies:
   - Add your proxies to `proxies.txt` (one per line).

4. Update ChromeDriver Path:
   - Edit `scraper.py` and set `self.path_to_chromedriver` to your `chromedriver.exe` location.

## Usage

Start the server:
```bash
python app.py
```
The server will run on `http://0.0.0.0:1000`.

## API Endpoints

### 1. Scrape Website
- **URL**: `/scrape`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "url": "https://example.com",
    "download_images": false,
    "get_html": false
  }
  ```

### 2. Generate Embeddings
- **URL**: `/v1/embeddings`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "model": "jina-embeddings-v3",
    "input": ["text to embed", "another text"],
    "dimensions": 1024
  }
  ```

### 3. Rerank Documents
- **URL**: `/v1/rerank`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "model": "jina-reranker-v2-base-multilingual",
    "query": "What is the capital of France?",
    "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."],
    "top_n": 1
  }
  ```

## License
[MIT](LICENSE)
