from typing import List, Tuple, Dict, Any, Optional
from fastapi import HTTPException
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
import base64
import asyncio
from fake_useragent import UserAgent
import random
import httpx
import re

from models import ScrapeRequest, ScrapeResponse


class Scraper:
    def __init__(self, proxy_file="proxies.txt", max_workers=5):
        self.max_workers = max_workers
        self.ua = UserAgent()

        # --- PROXY LOADING ---
        self.proxies = self._load_proxies(proxy_file)
        if not self.proxies:
            print("Warning: No proxies loaded. Running without proxies.")
        else:
            print(f"Successfully loaded {len(self.proxies)} proxies.")

    def _load_proxies(self, filename: str) -> List[str]:
        """Loads proxies from a file, one proxy per line."""
        try:
            with open(filename, "r") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def _get_random_proxy(self) -> Optional[str]:
        """Returns a random proxy from the list."""
        return random.choice(self.proxies) if self.proxies else None

    def _parse_proxy(self, proxy_str: str) -> Dict[str, str]:
        """Parses a proxy string into Playwright proxy format."""
        # Simple parsing for http://user:pass@host:port or http://host:port
        # Playwright expects: { server: 'host:port', username: '...', password: '...' }
        match = re.match(
            r"(?:https?://)?(?:([^:]+):([^@]+)@)?([^:/]+):(\d+)", proxy_str
        )
        if match:
            user, password, host, port = match.groups()
            proxy_dict = {"server": f"{host}:{port}"}
            if user and password:
                proxy_dict["username"] = user
                proxy_dict["password"] = password
            return proxy_dict
        return {"server": proxy_str.replace("http://", "").replace("https://", "")}

    async def scrape(self, scrape_request: ScrapeRequest) -> ScrapeResponse:
        """Asynchronously scrapes a website using Playwright."""
        try:
            result = await self._scrape_website(
                scrape_request.url,
                scrape_request.download_images,
                scrape_request.get_html,
            )
            return ScrapeResponse(data=result)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error during scraping: {str(e)}"
            )

    async def _scrape_website(
        self, url: str, download_images: bool = False, get_html: bool = False
    ) -> Dict[str, Any]:
        """Internal method to perform the actual scraping with Playwright."""
        proxy_str = self._get_random_proxy()
        proxy = self._parse_proxy(proxy_str) if proxy_str else None

        async with async_playwright() as p:
            # Launch the browser
            browser = await p.chromium.launch(headless=True, proxy=proxy)

            # Create a context with a random user agent
            context = await browser.new_context(
                user_agent=self.ua.random, viewport={"width": 1920, "height": 1080}
            )

            page = await context.new_page()

            if proxy_str and proxy:
                print(f"Using proxy: {proxy.get('server')} for URL: {url}")

            try:
                # Navigate to the URL
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)

                # Human-like delay
                await asyncio.sleep(random.uniform(2, 5))

                # Simulate scrolling
                for _ in range(3):
                    await page.mouse.wheel(0, 500)
                    await asyncio.sleep(random.uniform(0.5, 1.5))

                # Scroll to bottom to trigger lazy loading
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(random.uniform(1, 2))

                # Extract data
                title = await page.title()

                description = ""
                description_meta = await page.query_selector('meta[name="description"]')
                if description_meta:
                    description = await description_meta.get_attribute("content") or ""

                content = await page.inner_text("body")
                html = await page.content() if get_html else None

                links: List[Tuple[str, str]] = []
                a_tags = await page.query_selector_all("a")
                for a_tag in a_tags:
                    try:
                        href = await a_tag.get_attribute("href")
                        text = await a_tag.inner_text()
                        if href:
                            links.append((text.strip(), href))
                    except:
                        pass

                images_metadata = []
                images_data = {}

                if download_images:
                    img_elements = await page.query_selector_all("img")
                    img_urls = set()
                    for img in img_elements:
                        src = await img.get_attribute("src")
                        if src:
                            if src.startswith("http"):
                                img_urls.add(src)

                    # In newer httpx versions, use 'proxy' instead of 'proxies'
                    async with httpx.AsyncClient(proxy=proxy_str) as client:
                        for img_url in img_urls:
                            try:
                                response = await client.get(img_url, timeout=10)
                                response.raise_for_status()
                                img_binary_data = response.content
                                format_type = (
                                    img_url.split(".")[-1].lower()
                                    if "." in img_url
                                    else "unknown"
                                )
                                format_type = format_type.split("?")[0]

                                images_metadata.append(
                                    {
                                        "url": img_url,
                                        "format": format_type,
                                        "size_bytes": len(img_binary_data),
                                    }
                                )
                                images_data[img_url] = base64.b64encode(
                                    img_binary_data
                                ).decode("utf-8")
                            except Exception as e:
                                print(
                                    f"Failed to download image {img_url} via proxy: {str(e)}"
                                )

                return {
                    "title": title,
                    "description": description,
                    "url": url,
                    "content": content,
                    "html": html,
                    "usage": {"tokens": 0},
                    "links": links,
                    "images_metadata": images_metadata,
                    "images_data": images_data if download_images else None,
                }

            except Exception as e:
                raise Exception(
                    f"Scraping failed for {url} with proxy {proxy_str}: {str(e)}"
                )

            finally:
                await browser.close()

    async def close(self):
        """No-op for compatibility with previous interface."""
        pass
