import base64
import os
from typing import Dict, Union
from pathlib import Path

from all_in_one import ScrapeResponse, ImageData


def read_scraped_images(
    response_data: Union[dict, ScrapeResponse], 
    save_to_disk: bool = False, 
    output_dir: str = "images"
) -> Dict[str, Union[bytes, str]]:
    """
    Processes base64 encoded images from scrape_website response.
    
    Args:
        response_data: Either ScrapeResponse object or raw response dict containing images
        save_to_disk: If True, saves images to output_dir. If False, returns decoded bytes
        output_dir: Directory to save images (created if doesn't exist)
    
    Returns:
        Dict mapping image URLs to:
        - image bytes (if save_to_disk=False)
        - file paths (if save_to_disk=True)
    
    Raises:
        ValueError: If response_data doesn't contain valid image data
        OSError: If unable to create output directory or write files
    """
    # Extract images data from response
    if isinstance(response_data, ScrapeResponse):
        images_data = response_data.data.images_data
    else:
        images_data = response_data.get('data').get('images_data')

    if not images_data:
        raise ValueError("Response data contains no images")
    
    # Prepare output directory if saving to disk
    if save_to_disk:
        os.makedirs(output_dir, exist_ok=True)
    
    result = {}
    for img_url, base64_data in images_data.items():
        try:
            # Decode base64 image data
            img_bytes = base64.b64decode(base64_data)
            
            if save_to_disk:
                # Determine file extension from URL or default to .png
                ext = Path(img_url).suffix.lower()
                if not ext or ext not in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}:
                    ext = '.png'
                
                # Create safe filename from URL
                filename = f"{hash(img_url)}{ext}"
                filepath = os.path.join(output_dir, filename)
                
                # Save to file
                with open(filepath, 'wb') as f:
                    f.write(img_bytes)
                
                result[img_url] = filepath
            else:
                result[img_url] = img_bytes
                
        except (base64.binascii.Error, OSError) as e:
            print(f"Failed to process image {img_url}: {str(e)}")
            continue
    
    return result

def validate_image_data(base64_str: str) -> bool:
    """
    Validates if a string is valid base64 encoded image data.
    
    Args:
        base64_str: String to validate
    
    Returns:
        True if valid base64 image data, False otherwise
    """
    try:
        base64.b64decode(base64_str)
        return True
    except (base64.binascii.Error, ValueError):
        return False
