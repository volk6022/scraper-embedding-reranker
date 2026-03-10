import re
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import HTTPException
from models import ReaderRequest, ReaderResponse
from config import args

# Patterns for HTML cleaning
SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"


class Reader:
    def __init__(self):
        self.model_name = "jinaai/ReaderLM-v2"
        self.tokenizer = None
        self.model = None
        self.device = args.device if torch.cuda.is_available() else "cpu"
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)

        if not args.lazy_load:
            self._load_model()

    def _load_model(self):
        if self.model is None:
            print(f"Loading Reader model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
                self.device
            )
            self.model.eval()

    def clean_html(
        self, html: str, clean_svg: bool = False, clean_base64: bool = False
    ):
        html = re.sub(
            SCRIPT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            META_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            COMMENT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            LINK_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )

        if clean_svg:
            html = re.sub(
                SVG_PATTERN,
                lambda match: f"{match.group(1)}this is a placeholder{match.group(3)}",
                html,
                flags=re.DOTALL,
            )
        if clean_base64:
            html = re.sub(BASE64_IMG_PATTERN, '<img src="#"/>', html)
        return html

    def create_prompt(
        self, text: str, instruction: str = None, schema: str = None
    ) -> str:
        if not instruction:
            instruction = "Extract the main content from the given HTML and convert it to Markdown format."
        if schema:
            instruction = "Extract the specified information from a list of news threads and present it in a structured JSON format."
            prompt = f"{instruction}\n```html\n{text}\n```\nThe JSON schema is as follows:```json\n{schema}\n```"
        else:
            prompt = f"{instruction}\n```html\n{text}\n```"

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _sync_generate(self, request: ReaderRequest) -> str:
        """Synchronous blocking inference logic."""
        self._load_model()

        if args.low_vram and self.device.startswith("cuda"):
            self.model.to(self.device)

        html = self.clean_html(request.html, request.clean_svg, request.clean_base64)
        input_prompt = self.create_prompt(
            html, request.instruction, request.extraction_schema
        )

        inputs = self.tokenizer.encode(input_prompt, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=1024,
                temperature=0,
                do_sample=False,
                repetition_penalty=1.08,
            )

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "assistant\n" in decoded_output:
            decoded_output = decoded_output.split("assistant\n")[-1]

        if args.low_vram and self.device.startswith("cuda"):
            self.model.to("cpu")
            torch.cuda.empty_cache()

        return decoded_output

    async def read(self, request: ReaderRequest) -> ReaderResponse:
        async with self.lock:
            try:
                loop = asyncio.get_event_loop()
                content = await loop.run_in_executor(
                    self.executor, self._sync_generate, request
                )
                return ReaderResponse(content=content)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error in Reader: {str(e)}"
                )

    async def close(self):
        self.executor.shutdown()
