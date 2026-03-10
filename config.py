import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scraper, Embedding, Reranker, and Reader API"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=1000, help="API port")
    parser.add_argument(
        "--lazy-load", action="store_true", help="Load models only when needed"
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Offload models to CPU/RAM when not in use",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Default device (cuda/cpu)"
    )

    return parser.parse_args()


args = parse_args()
