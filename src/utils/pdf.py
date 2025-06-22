from pdf2image import convert_from_path
from typing import List, Dict
from PIL import Image
import os
import base64
import io

import cv2
import numpy as np

def extract_pdf_pages_as_images(pdf_path: str, dpi: int = 200) -> List[str]:
    """
    Extract pages from a PDF file and convert them to a list of base64-encoded images.

    Args:
        pdf_path (str): Path to the PDF file
        dpi (int): DPI for image conversion (higher = better quality, larger file)

    Returns:
        List[str]: List of base64-encoded PNG images, one for each page

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If there's an error during PDF processing
    """
    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # Convert PDF pages to images
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt='ppm',  # Internal format for processing
            thread_count=1
        )

        # Convert each image to base64-encoded PNG
        base64_images = []
        for img in images:
            with io.BytesIO() as output:
                img.save(output, format='PNG')
                base64_str = base64.b64encode(output.getvalue()).decode('utf-8')
                base64_images.append(base64_str)

        return base64_images

    except Exception as e:
        raise Exception(f"Error processing PDF '{pdf_path}': {str(e)}")


def save_pdf_pages_as_images(pdf_path: str, output_dir: str, 
                            image_format: str = 'PNG', dpi: int = 200) -> List[str]:
    """
    Extract PDF pages and save them as image files.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save the images
        image_format (str): Image file format ('PNG', 'JPEG', 'TIFF')
        dpi (int): DPI for image conversion
    
    Returns:
        List[str]: List of saved image file paths
    
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If there's an error during processing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract images
    images = extract_pdf_pages_as_images(pdf_path, dpi=dpi)
    
    # Save images
    saved_paths = []
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    for i, image_base64 in enumerate(images, 1):
        filename = f"{pdf_name}_page_{i:03d}.{image_format.lower()}"
        filepath = os.path.join(output_dir, filename)

        # Decode base64 string to bytes and open as image
        image_bytes = base64.b64decode(image_base64)
        with io.BytesIO(image_bytes) as img_buffer:
            with Image.open(img_buffer) as img:
                # Save with appropriate settings for different formats
                if image_format.upper() == 'JPEG':
                    img.save(filepath, format=image_format, quality=95, optimize=True)
                else:
                    img.save(filepath, format=image_format)

        saved_paths.append(filepath)
    
    return saved_paths

def extract_images_from_page_base64(page_base64: str) -> List[bytes]:
    """
    Detect and extract images embedded within a single PDF page image (base64-encoded).

    Args:
        page_base64 (str): Base64-encoded image of a PDF page.

    Returns:
        List[bytes]: List of extracted image regions as bytes (JPEG format).
    """
    # Decode base64 to bytes and load as OpenCV image
    image_bytes = base64.b64decode(page_base64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return []

    # Convert to grayscale and threshold to find contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use adaptive threshold to handle varying backgrounds
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 10
    )

    # Find contours (external only)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_images = []
    h_img, w_img = img.shape[:2]
    min_area = 5000  # Minimum area to consider as an image (tune as needed)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Heuristic: ignore very small or very large regions (likely not images)
        if area < min_area or w > 0.95 * w_img or h > 0.95 * h_img:
            continue
        # Crop the region and encode as JPEG
        crop = img[y:y+h, x:x+w]
        success, buf = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if success:
            extracted_images.append(buf.tobytes())

    return extracted_images

def extract_images_from_pdf(pdf_path: str, dpi: int = 200) -> Dict[int, List[bytes]]:
    """
    For each page in a PDF, detect and extract images embedded within the page.

    Args:
        pdf_path (str): Path to the PDF file.
        dpi (int): DPI for page rendering.

    Returns:
        Dict[int, List[bytes]]: Mapping from page index (0-based) to list of extracted image bytes.
    """
    # Get base64 images for each page
    page_base64s = extract_pdf_pages_as_images(pdf_path, dpi=dpi)
    page_images_map = {}

    for idx, page_b64 in enumerate(page_base64s):
        images = extract_images_from_page_base64(page_b64)
        page_images_map[idx] = images

    return page_images_map

