from typing import List, Tuple, Dict
from fastapi import HTTPException
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
from concurrent.futures import ThreadPoolExecutor
import asyncio
from selenium_stealth import stealth
from fake_useragent import UserAgent
import random
import time
import httpx
import re

from models import ScrapeRequest, ScrapeResponse


class Scraper:
    def __init__(self, proxy_file="proxies.txt", max_workers=5):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.path_to_chromedriver = "C:\\Users\\bhunp\\chrome-win64\\chromedriver.exe"
        self.service = Service(executable_path=self.path_to_chromedriver)

        # --- PROXY LOADING ---
        self.proxies = self._load_proxies(proxy_file)
        if not self.proxies:
            print("Warning: No proxies loaded. Running without proxies.")
        else:
            print(f"Successfully loaded {len(self.proxies)} proxies.")
        
    def _load_proxies(self, filename: str) -> List[str]:
        """Loads proxies from a file, one proxy per line."""
        try:
            with open(filename, 'r') as f:
                # Read lines and strip any whitespace
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def _get_random_proxy(self) -> str | None:
        """Returns a random proxy from the list."""
        return random.choice(self.proxies) if self.proxies else None

    async def scrape(
            self, 
            scrap_request: ScrapeRequest
        ) -> ScrapeResponse:
        """Asynchronously scrapes a website using a dedicated, proxy-enabled driver."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._scrape_website_with_proxy, # We call the new proxy-aware method
                scrap_request.url,
                scrap_request.download_images,
                scrap_request.get_html
            )
        except Exception as e:
            # Catch exceptions bubbled up from the thread
            raise HTTPException(status_code=500, detail=f"Error during scraping: {str(e)}")

        return ScrapeResponse(data=result)
    
    def _scrape_website_with_proxy(self, url: str, download_images: bool = False, get_html: bool = False) -> Dict:
        """
        Creates a new WebDriver instance with a random proxy for a single scrape job,
        then closes it.
        """
        proxy = self._get_random_proxy()
        
        ua = UserAgent()
        chrome_options = Options()
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument(f"user-agent={ua.random}")
        
        if proxy:
            # Selenium expects the proxy server without the scheme (http://)
            # We use a simple regex to strip it.
            proxy_server = re.sub(r'^(http|https|socks5)://', '', proxy)
            chrome_options.add_argument(f'--proxy-server={proxy_server}')
            print(f"Using proxy: {proxy_server} for URL: {url}")

        driver = None  # Initialize driver to None
        try:
            # Each thread gets its own driver
            driver = webdriver.Chrome(service=self.service, options=chrome_options)

            # Apply stealth to the newly created driver
            stealth(driver,
                    languages=["ru-Ru", "ru"],
                    vendor="Google Inc.",
                    platform="Win32",
                    webgl_vendor="Intel Inc.",
                    renderer="Intel Iris OpenGL Engine",
                    fix_hairline=True
                )
            
            self.driver = driver
            
            # --- Start of actual scraping logic (same as before) ---
            self.driver.get(url)
            time.sleep(random.uniform(2, 5))

            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

            # Simulate scrolling down the page like a human
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/4);")
            time.sleep(random.uniform(0.5, 1.5))
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(random.uniform(0.5, 1.5))
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            try:
                title = self.driver.find_element(By.TAG_NAME, "title").text
            except:
                title = ""

            try:
                description_element = self.driver.find_element(
                    By.XPATH, '//meta[@name="description"]'
                )
                description = description_element.get_attribute("content")
            except:
                description = ""

            content = self.driver.find_element(By.TAG_NAME, "body").text
            html = self.driver.page_source if get_html else None

            links: List[Tuple[str, str]] = []
            a_tags = self.driver.find_elements(By.TAG_NAME, "a")
            for a_tag in a_tags:
                try:
                    href = a_tag.get_attribute("href")
                    text = a_tag.text
                    if href:
                        links.append((text, href))
                except:
                    pass

            images_metadata = []
            images_data = {}

            if download_images:
                img_elements = driver.find_elements(By.TAG_NAME, "img")
                img_urls = {img.get_attribute("src") for img in img_elements if img.get_attribute("src")}
                
                # We need to pass the proxy to httpx as well!
                proxy_dict = {"http://": proxy, "https://": proxy} if proxy else None
                with httpx.Client(proxies=proxy_dict) as client:
                    for img_url in img_urls:
                        try:
                            response = client.get(img_url, timeout=10)
                            response.raise_for_status()
                            img_binary_data = response.content
                            format_type = img_url.split('.')[-1].lower() if '.' in img_url else 'unknown'
                            images_metadata.append({
                                "url": img_url, 
                                "format": format_type, 
                                "size_bytes": len(img_binary_data)
                            })
                            images_data[img_url] = base64.b64encode(img_binary_data).decode('utf-8')
                        except Exception as e:
                            print(f"Failed to download image {img_url} via proxy: {str(e)}")

            return {
                "title": title, 
                "description": description, 
                "url": url, 
                "content": content,
                "html": html,
                "usage": {"tokens": 0}, 
                "links": links, 
                "images_metadata": images_metadata,
                "images_data": images_data if download_images else None
            }
            # --- End of actual scraping logic ---

        except Exception as e:
            # Re-raise the exception so it can be caught by the main `scrape` method
            raise Exception(f"Scraping failed for {url} with proxy {proxy}: {str(e)}")
        
        finally:
            if driver:
                driver.quit() # CRUCIAL: Always close the driver to free up resources

    async def close(self):
        # self.driver.quit()
        self.executor.shutdown(wait=True)
