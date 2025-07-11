from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch
import numpy as np
from sentence_transformers import util
import uvicorn
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import json
import asyncio  # Import asyncio
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor

app = FastAPI()

# --- Reranker Model Loading ---
reranker_model_name = "jinaai/jina-reranker-v2-base-multilingual"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name, trust_remote_code=True)

# --- Embeddings Model Loading ---
embeddings_model_name = "jinaai/jina-embeddings-v3"
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name)
embeddings_model = AutoModel.from_pretrained(embeddings_model_name, trust_remote_code=True)


# --- Device Selection ---
device = "cuda" if torch.cuda.is_available() else "cpu"

reranker_model.to(device)
embeddings_model.to(device)
embeddings_model.eval()

# --- scraper ---
path_to_chromedriver = "C:\\Users\\bhunp\\chrome-win64\\chromedriver.exe" # Укажите свой путь
use_cutting = False

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--remote-debugging-port=9222")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-infobars")
chrome_options.add_argument("--mute-audio")
chrome_options.add_argument("--disable-background-networking")
chrome_options.add_argument("--disable-background-timer-throttling")


# --- Initialize a single WebDriver instance ---
# This avoids creating a new driver for each request, improving performance
driver = None
def initialize_driver():
    global driver
    if driver is None:
        service = Service(executable_path=path_to_chromedriver)
        driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# Initialize the driver on startup
@app.on_event("startup")
async def startup_event():
    initialize_driver()

# Close the driver on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    if driver:
        driver.quit()

# --- Request and Response models ---
class RerankRequest(BaseModel):
    model: str
    query: str
    top_n: int
    documents: list[str]


class RerankResult(BaseModel):
    index: int
    document: dict
    relevance_score: float


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]
    usage: dict  # Simplified usage tracking


class EmbeddingInput(BaseModel):
    model: str = "jina-embeddings-v3"  # Enforce the model name
    input: List[str]
    task: Optional[str] = "text-matching"
    truncate: Optional[bool] = True
    dimensions: Optional[int] = 128  # Adjust as needed
    late_chunking: Optional[bool] = None
    embedding_type: Optional[str] = None


class EmbeddingData(BaseModel):
    index: int
    object: str = "embedding"
    embedding: List[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str = "jina-embeddings-v3"
    object: str = "list"
    usage: EmbeddingUsage


# Define the data model for the request
class ScrapeRequest(BaseModel):
    url: str


# Define the data model for the response
class ScrapeResponse(BaseModel):
    code: int
    status: int
    data: dict
    message: str
    readableMessage: str



# --- Reranking Endpoint ---
@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    try:
        query = request.query
        documents = request.documents

        try:  # Wrap model inference in a try...except block
            features = reranker_tokenizer(
                [query] * len(documents),
                documents,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            features = {k: v.to(device) for k, v in features.items()} #Move tensors to device

            with torch.no_grad():
                reranker_model.eval()
                scores = reranker_model(**features).logits.flatten().float().tolist()

        except Exception as e:
            print(f"Error during model inference: {e}") # Log the error
            raise HTTPException(status_code=500, detail=f"Error during model inference: {e}")


        results = []
        for i, score in enumerate(scores):
            results.append(
                RerankResult(
                    index=i,
                    document={"text": documents[i]},
                    relevance_score=score,
                )
            )

        results.sort(key=lambda x: x.relevance_score, reverse=True)


        response = RerankResponse(
            model=request.model,
            results=results,
            usage={"total_tokens": 0},
        )

        return response

    except Exception as e:
        print(f"General error in /rerank endpoint: {e}") # Log the outer error
        raise HTTPException(status_code=500, detail=str(e))


# --- Embedding Generation Function ---
def get_embeddings(texts: List[str], dimensions: int) -> List[List[float]]:
    """Generates embeddings for a list of texts using the loaded model."""
    # Tokenize the input texts
    encoded_input = embeddings_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute token usage
    prompt_tokens = torch.sum(encoded_input['attention_mask']).item()
    total_tokens = prompt_tokens

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        model_output = embeddings_model(**encoded_input, return_dict=True)
        sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings.cpu().tolist()

    return sentence_embeddings, prompt_tokens, total_tokens


# --- Embeddings Endpoint ---
@app.post("/embeddings", response_model=EmbeddingResponse)
async def embeddings_endpoint(request: Request, item: EmbeddingInput):
    """Endpoint for generating embeddings."""

    if item.model != "jina-embeddings-v3":
        raise HTTPException(status_code=400, detail="Only 'jina-embeddings-v3' model is supported.")

    embeddings, prompt_tokens, total_tokens = get_embeddings(item.input, item.dimensions)

    data = []
    for index, embedding in enumerate(embeddings):
        data.append(EmbeddingData(index=index, embedding=embedding))

    usage = EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)

    return EmbeddingResponse(data=data, usage=usage)


# --- scraper ---
# Use a thread pool to handle scraping tasks concurrently
executor = ThreadPoolExecutor(max_workers=5)  # Adjust max_workers as needed

def scrape_website(url: str):
    """Scrapes content from a given URL using Selenium."""
    try:
        driver = initialize_driver()  # Ensure driver is initialized
        # print('getting page from url')
        # Navigate to the URL
        driver.get(url)
        time.sleep(1)  # Wait for the page to load

        # print('getting title')
        # Extract title
        try:
            title = driver.find_element(By.TAG_NAME, "title").text
        except:
            title = ""
        # print(title)

        # print('getting description')
        # Extract description (from meta tag)
        try:
            description_element = driver.find_element(
                By.XPATH, '//meta[@name="description"]'
            )
            description = description_element.get_attribute("content")
        except:
            description = ""
        # print(description)

        # print('getting content')
        # Extract text content
        content = driver.find_element(By.TAG_NAME, "body").text
        # print(type(content), len(content))

        # print('getting links')
        # Extract links
        links: List[Tuple[str, str]] = []
        a_tags = driver.find_elements(By.TAG_NAME, "a")
        for a_tag in a_tags:
            try:
                href = a_tag.get_attribute("href")
                text = a_tag.text
                if href:
                    links.append((text, href))
            except:
                pass
        # print(links)

        # Count tokens
        # tokens = count_tokens(content)
        tokens = 0

        data = {
            "title": title,
            "description": description,
            "url": url,
            "content": content,
            "usage": {"tokens": tokens},
            "links": links,
        }
        # print('scrape finished')
        return data

    except Exception as e:
        # No need to quit the driver here; it's handled in shutdown_event
        raise Exception(f"Scraping failed: {str(e)}")


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape(request: ScrapeRequest):
    print(f"scrape input: url={request.url}")
    try:
        # Run the scraping task in a separate thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, scrape_website, request.url)

        response = ScrapeResponse(
            code=200,
            status=200,
            data=result,
            message='',
            readableMessage=''
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1000)


# from fastapi import FastAPI, Request, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional, Dict
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
# import torch
# import numpy as np
# from sentence_transformers import util
# import uvicorn
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# import time
# import json


# app = FastAPI()

# # --- Reranker Model Loading ---
# reranker_model_name = "jinaai/jina-reranker-v2-base-multilingual"
# reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
# reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name, trust_remote_code=True)

# # --- Embeddings Model Loading ---
# embeddings_model_name = "jinaai/jina-embeddings-v3"
# embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name)
# embeddings_model = AutoModel.from_pretrained(embeddings_model_name, trust_remote_code=True)


# # --- Device Selection ---
# device = "cuda" if torch.cuda.is_available() else "cpu"

# reranker_model.to(device)
# embeddings_model.to(device)
# embeddings_model.eval()

# # --- scraper ---
# path_to_chromedriver = "C:\\Users\\bhunp\\chrome-win64\\chromedriver.exe" # Укажите свой путь
# use_cutting = False

# # Set up Chrome options
# # chrome_options = Options()
# # chrome_options.add_argument("--headless")  # Run in headless mode
# # chrome_options.add_argument("--disable-gpu")
# # chrome_options.add_argument("--no-sandbox")
# # chrome_options.add_argument("--disable-dev-shm-usage")
# # chrome_options.add_argument("--remote-debugging-port=9222")
# # chrome_options.add_argument("--disable-blink-features=AutomationControlled")
# # chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36")
# chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--disable-gpu")
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--disable-dev-shm-usage")
# chrome_options.add_argument("--remote-debugging-port=9222")
# chrome_options.add_argument("--disable-blink-features=AutomationControlled")
# chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36")
# chrome_options.add_argument("--disable-extensions")
# chrome_options.add_argument("--disable-infobars")
# chrome_options.add_argument("--mute-audio")
# chrome_options.add_argument("--disable-background-networking")
# chrome_options.add_argument("--disable-background-timer-throttling")
# #chrome_options.add_argument("--disable-features=NetworkService") # Попробуйте это в последнюю очередь


# # Set up the WebDriver
# service = Service(executable_path=path_to_chromedriver)
# driver = webdriver.Chrome(service=service, options=chrome_options)



# # --- Request and Response models ---
# class RerankRequest(BaseModel):
#     model: str
#     query: str
#     top_n: int
#     documents: list[str]


# class RerankResult(BaseModel):
#     index: int
#     document: dict
#     relevance_score: float


# class RerankResponse(BaseModel):
#     model: str
#     results: list[RerankResult]
#     usage: dict  # Simplified usage tracking


# class EmbeddingInput(BaseModel):
#     model: str = "jina-embeddings-v3"  # Enforce the model name
#     input: List[str]
#     task: Optional[str] = "text-matching"
#     truncate: Optional[bool] = True
#     dimensions: Optional[int] = 128  # Adjust as needed
#     late_chunking: Optional[bool] = None
#     embedding_type: Optional[str] = None


# class EmbeddingData(BaseModel):
#     index: int
#     object: str = "embedding"
#     embedding: List[float]


# class EmbeddingUsage(BaseModel):
#     prompt_tokens: int
#     total_tokens: int


# class EmbeddingResponse(BaseModel):
#     data: List[EmbeddingData]
#     model: str = "jina-embeddings-v3"
#     object: str = "list"
#     usage: EmbeddingUsage


# # Define the data model for the request
# class ScrapeRequest(BaseModel):
#     url: str
#     # Дополнительные параметры, если нужны
#     # Например:  use_cutting: Optional[bool] = False


# # Define the data model for the response
# class ScrapeResponse(BaseModel):
#     code: int
#     status: int
#     data: dict
#     message: str
#     readableMessage: str



# # --- Reranking Endpoint ---
# @app.post("/rerank", response_model=RerankResponse)
# async def rerank(request: RerankRequest):
#     try:
#         query = request.query
#         documents = request.documents

#         try:  # Wrap model inference in a try...except block
#             features = reranker_tokenizer(
#                 [query] * len(documents),
#                 documents,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#             )

#             features = {k: v.to(device) for k, v in features.items()} #Move tensors to device

#             with torch.no_grad():
#                 reranker_model.eval()
#                 scores = reranker_model(**features).logits.flatten().float().tolist()

#         except Exception as e:
#             print(f"Error during model inference: {e}") # Log the error
#             raise HTTPException(status_code=500, detail=f"Error during model inference: {e}")


#         results = []
#         for i, score in enumerate(scores):
#             results.append(
#                 RerankResult(
#                     index=i,
#                     document={"text": documents[i]},
#                     relevance_score=score,
#                 )
#             )

#         results.sort(key=lambda x: x.relevance_score, reverse=True)


#         response = RerankResponse(
#             model=request.model,
#             results=results,
#             usage={"total_tokens": 0},
#         )

#         return response

#     except Exception as e:
#         print(f"General error in /rerank endpoint: {e}") # Log the outer error
#         raise HTTPException(status_code=500, detail=str(e))


# # --- Embedding Generation Function ---
# def get_embeddings(texts: List[str], dimensions: int) -> List[List[float]]:
#     """Generates embeddings for a list of texts using the loaded model."""
#     # Tokenize the input texts
#     encoded_input = embeddings_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)

#     # Compute token usage
#     prompt_tokens = torch.sum(encoded_input['attention_mask']).item()
#     total_tokens = prompt_tokens

#     # Disable gradient calculation for faster inference
#     with torch.no_grad():
#         model_output = embeddings_model(**encoded_input, return_dict=True)
#         sentence_embeddings = model_output[0][:, 0]
#         # normalize embeddings
#         sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
#         sentence_embeddings = sentence_embeddings.cpu().tolist()

#     return sentence_embeddings, prompt_tokens, total_tokens


# # --- Embeddings Endpoint ---
# @app.post("/embeddings", response_model=EmbeddingResponse)
# async def embeddings_endpoint(request: Request, item: EmbeddingInput):
#     """Endpoint for generating embeddings."""

#     if item.model != "jina-embeddings-v3":
#         raise HTTPException(status_code=400, detail="Only 'jina-embeddings-v3' model is supported.")

#     embeddings, prompt_tokens, total_tokens = get_embeddings(item.input, item.dimensions)

#     data = []
#     for index, embedding in enumerate(embeddings):
#         data.append(EmbeddingData(index=index, embedding=embedding))

#     usage = EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)

#     return EmbeddingResponse(data=data, usage=usage)


# # --- scraper ---
# def scrape_website(url: str):
#     try:
#         # print('getting page from url')
#         # Navigate to the URL
#         driver.get(url)
#         time.sleep(3)  # Wait for the page to load

#         # print('getting title')
#         # Extract title
#         try:
#             title = driver.find_element(By.TAG_NAME, "title").text
#         except:
#             title = ""
#         # print(title)

#         # print('getting description')
#         # Extract description (from meta tag)
#         try:
#             description_element = driver.find_element(
#                 By.XPATH, '//meta[@name="description"]'
#             )
#             description = description_element.get_attribute("content")
#         except:
#             description = ""
#         # print(description)

#         # print('getting content')
#         # Extract text content
#         content = driver.find_element(By.TAG_NAME, "body").text
#         # print(type(content), len(content))

#         # print('getting links')
#         # Extract links
#         links: List[Tuple[str, str]] = []
#         a_tags = driver.find_elements(By.TAG_NAME, "a")
#         for a_tag in a_tags:
#             try:
#                 href = a_tag.get_attribute("href")
#                 text = a_tag.text
#                 if href:
#                     links.append((text, href))
#             except:
#                 pass
#         # print(links)

#         # Count tokens
#         # tokens = count_tokens(content)
#         tokens = 0

#         data = {
#             "title": title,
#             "description": description,
#             "url": url,
#             "content": content,
#             "usage": {"tokens": tokens},
#             "links": links,
#         }
#         # print('scrape finished')
#         return data

#     except Exception as e:
#         if driver:
#             driver.quit()
#         raise HTTPException(
#             status_code=500, detail=f"Scraping failed: {str(e)}"
#         )  # Proper exception handling


# @app.post("/scrape", response_model=ScrapeResponse)
# async def scrape(request: ScrapeRequest):
#     print(f"scrape input: url={request.url}")
#     try:
#         result = scrape_website(request.url)
#         # print(f"scrap result: {result["title"]}, {result["url"]}")
#         response = ScrapeResponse(
#             code=200,
#             status=200,
#             data=result,
#             message='',
#             readableMessage=''
#         )
#         # print(f"response: {response.data.title}, {response.data.url}")
#         return response
#     except HTTPException as e:
#         raise e  # Re-raise HTTPExceptions
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=1000)