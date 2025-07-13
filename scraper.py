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

from models import ScrapeRequest, ScrapeResponse


class Scraper:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
        path_to_chromedriver = "C:\\Users\\bhunp\\chrome-win64\\chromedriver.exe"
        
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
        chrome_options.add_argument("--enable-unsafe-swiftshader")
        
        self.service = Service(executable_path=path_to_chromedriver)
        self.driver = webdriver.Chrome(service=self.service, options=chrome_options)

    async def scrape(
        self,
        scrap_request: ScrapeRequest
    ) -> ScrapeResponse:
        url = scrap_request.url
        timeout: int = 10,
        max_content_length: int = 1000000,
        extract_text: bool = True,
        download_images = scrap_request.download_images
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._scrape_website,
                url,
                download_images
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during scraping: {str(e)}"
            )

        return ScrapeResponse(data=result)

    def _scrape_website(self, url: str, download_images: bool = False) -> Dict:
        try:
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 10)
            
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            
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
                wait.until(EC.presence_of_element_located((By.TAG_NAME, 'img')))
                
                img_elements = self.driver.find_elements(By.TAG_NAME, "img")
                img_urls = [img.get_attribute("src") for img in img_elements if img.get_attribute("src")]
                
                for img_url in img_urls:
                    try:
                        self.driver.get(img_url)
                        img = self.driver.find_element(By.TAG_NAME, "img")
                        width = img.get_attribute("width") or 0
                        height = img.get_attribute("height") or 0
                        format = img_url.split('.')[-1].lower() if '.' in img_url else 'unknown'
                        img_data = self.driver.get_screenshot_as_png()
                        
                        images_metadata.append({
                            "url": img_url,
                            "width": int(width),
                            "height": int(height),
                            "format": format,
                            "size_bytes": len(img_data)
                        })
                        images_data[img_url] = base64.b64encode(img_data).decode('utf-8')
                    except Exception as e:
                        print(f"Failed to download image {img_url}: {str(e)}")
                
                self.driver.get(url)
            
            return {
                "title": title,
                "description": description,
                "url": url,
                "content": content,
                "usage": {"tokens": 0},
                "links": links,
                "images_metadata": images_metadata,
                "images_data": images_data if download_images else None
            }

        except Exception as e:
            raise Exception(f"Scraping failed: {str(e)}")

    async def close(self):
        self.driver.quit()
        self.executor.shutdown()
