"""
PDF Processing Pipeline for Visual-Graph RAG
Converts PDF to images and processes with DeepSeek-OCR
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional

from PIL import Image
from tqdm import tqdm

try:
    import fitz  # PyMuPDF

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from pdf2image import convert_from_path

    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

from .deepseek_ocr import DeepSeekOCR, PageOCRResult


@dataclass
class PDFInfo:
    """PDF document information"""

    path: str
    num_pages: int
    title: str
    author: str
    creation_date: str
    file_size: int


class PDFProcessor:
    """
    PDF Processing Pipeline
    Handles PDF to image conversion and OCR processing
    """

    def __init__(
        self,
        ocr_model: Optional[DeepSeekOCR] = None,
        dpi: int = 150,
        batch_size: int = 4,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize PDF Processor

        Args:
            ocr_model: DeepSeekOCR instance
            dpi: DPI for PDF to image conversion
            batch_size: Number of pages to process in parallel
            cache_dir: Directory for caching intermediate results
        """
        self.ocr_model = ocr_model or DeepSeekOCR()
        self.dpi = dpi
        self.batch_size = batch_size
        self.cache_dir = cache_dir or tempfile.mkdtemp()

        if not HAS_PYMUPDF and not HAS_PDF2IMAGE:
            raise ImportError("Please install either PyMuPDF or pdf2image for PDF processing")

    def get_pdf_info(self, pdf_path: str) -> PDFInfo:
        """Get PDF metadata"""
        try:
            if HAS_PYMUPDF:
                doc = fitz.open(pdf_path)
                metadata = doc.metadata or {}
                info = PDFInfo(
                    path=pdf_path,
                    num_pages=len(doc),
                    title=metadata.get("title", "Unknown"),
                    author=metadata.get("author", "Unknown"),
                    creation_date=metadata.get("creationDate", "Unknown"),
                    file_size=os.path.getsize(pdf_path),
                )
                doc.close()
                return info
            else:
                # Fallback
                return PDFInfo(
                    path=pdf_path,
                    num_pages=0,
                    title="Unknown",
                    author="Unknown",
                    creation_date="Unknown",
                    file_size=os.path.getsize(pdf_path),
                )
        except Exception as e:
            print(f"Warning: Failed to get PDF metadata: {e}")
            return PDFInfo(
                path=pdf_path,
                num_pages=0,
                title="Unknown",
                author="Unknown",
                creation_date="Unknown",
                file_size=os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0,
            )

    def pdf_to_images(
        self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None
    ) -> Generator[Image.Image, None, None]:
        """
        Convert PDF pages to images

        Args:
            pdf_path: Path to PDF file
            start_page: First page to convert (0-indexed)
            end_page: Last page to convert (exclusive)

        Yields:
            PIL Image for each page
        """
        if HAS_PYMUPDF:
            yield from self._pymupdf_to_images(pdf_path, start_page, end_page)
        elif HAS_PDF2IMAGE:
            yield from self._pdf2image_to_images(pdf_path, start_page, end_page)
        else:
            raise RuntimeError("No PDF backend available")

    def _pymupdf_to_images(
        self, pdf_path: str, start_page: int, end_page: Optional[int]
    ) -> Generator[Image.Image, None, None]:
        """Convert using PyMuPDF (faster, recommended)"""
        doc = fitz.open(pdf_path)

        if end_page is None:
            end_page = len(doc)

        for page_num in range(start_page, min(end_page, len(doc))):
            page = doc[page_num]
            # Render page to image
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield img

        doc.close()

    def _pdf2image_to_images(
        self, pdf_path: str, start_page: int, end_page: Optional[int]
    ) -> Generator[Image.Image, None, None]:
        """Convert using pdf2image"""
        images = convert_from_path(
            pdf_path,
            dpi=self.dpi,
            first_page=start_page + 1,  # pdf2image uses 1-based indexing
            last_page=end_page if end_page else None,
        )

        for img in images:
            yield img

    def process_pdf(
        self,
        pdf_path: str,
        start_page: int = 0,
        end_page: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[PageOCRResult]:
        """
        Process entire PDF with DeepSeek-OCR

        Args:
            pdf_path: Path to PDF file
            start_page: First page to process
            end_page: Last page to process
            show_progress: Show progress bar

        Returns:
            List of PageOCRResult for each page

        Raises:
            ValueError: If PDF has no pages or is invalid
        """
        # Get PDF info
        pdf_info = self.get_pdf_info(pdf_path)
        total_pages = pdf_info.num_pages

        if total_pages == 0:
            raise ValueError(f"PDF has no pages or could not be read: {pdf_path}")

        if end_page is None:
            end_page = total_pages
        else:
            end_page = min(end_page, total_pages)

        if start_page >= end_page:
            raise ValueError(f"Invalid page range: {start_page} to {end_page}")

        print(f"Processing PDF: {pdf_path}")
        print(f"Total pages: {total_pages}")
        print(f"Processing pages {start_page} to {end_page}")

        results = []
        batch_images = []
        batch_start_page = start_page
        processing_errors = 0

        # Create progress bar
        pages_gen = self.pdf_to_images(pdf_path, start_page, end_page)
        if show_progress:
            pages_gen = tqdm(pages_gen, total=end_page - start_page, desc="Processing pages")

        page_num = start_page
        try:
            for image in pages_gen:
                try:
                    batch_images.append(image)

                    # Process batch when full
                    if len(batch_images) >= self.batch_size:
                        batch_results = self.ocr_model.batch_process(
                            batch_images, batch_start_page, show_progress=False
                        )
                        results.extend(batch_results)

                        # Clear batch
                        batch_images = []
                        batch_start_page = page_num + 1

                    page_num += 1
                except Exception as e:
                    processing_errors += 1
                    print(f"Warning: Failed to process page {page_num}: {e}")
                    page_num += 1
                    continue

            # Process remaining pages
            if batch_images:
                try:
                    batch_results = self.ocr_model.batch_process(
                        batch_images, batch_start_page, show_progress=False
                    )
                    results.extend(batch_results)
                except Exception as e:
                    processing_errors += 1
                    print(f"Warning: Failed to process final batch: {e}")

        except Exception as e:
            print(f"Error during PDF processing: {e}")
            if not results:
                raise ValueError(f"Failed to process any pages from {pdf_path}: {e}")

        if not results:
            raise ValueError(f"No pages were successfully processed from {pdf_path}")

        if processing_errors > 0:
            print(f"Completed with {processing_errors} processing errors")

        # Print compression statistics
        stats = self.ocr_model.get_compression_stats(results)
        print("\n--- Compression Statistics ---")
        print(f"Total pages processed: {stats['total_pages']}")
        print(f"Total compressed tokens: {stats['total_compressed_tokens']}")
        print(f"Compression ratio: {stats['compression_ratio']}")
        print(f"Tokens per page: {stats['tokens_per_page']:.1f}")
        print(f"Space savings: {stats['space_savings']}")

        return results

    def process_pdf_streaming(
        self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None
    ) -> Generator[PageOCRResult, None, None]:
        """
        Process PDF pages one at a time (memory efficient)

        Yields:
            PageOCRResult for each page
        """
        page_num = start_page
        for image in self.pdf_to_images(pdf_path, start_page, end_page):
            result = self.ocr_model.process_image(image, page_num)
            yield result
            page_num += 1

    def save_results(self, results: List[PageOCRResult], output_path: str):
        """Save OCR results to JSON"""
        import json

        data = {
            "num_pages": len(results),
            "pages": [r.to_dict() for r in results],
            "statistics": self.ocr_model.get_compression_stats(results),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {output_path}")

    def load_results(self, input_path: str) -> List[PageOCRResult]:
        """Load OCR results from JSON"""
        import json

        from .deepseek_ocr import BoundingBox, VisualRegion, VisualToken

        with open(input_path, "r") as f:
            data = json.load(f)

        results = []
        for page_data in data["pages"]:
            regions = []
            for r_data in page_data["regions"]:
                region = VisualRegion(
                    region_id=r_data["region_id"],
                    page_num=r_data["page_num"],
                    block_type=r_data["block_type"],
                    bbox=BoundingBox.from_list(r_data["bbox"]),
                    compressed_tokens=[],  # Tokens not saved to JSON
                    text_content=r_data["text_content"],
                    markdown_content=r_data["markdown_content"],
                    token_count=r_data["token_count"],
                    confidence=r_data["confidence"],
                    metadata=r_data.get("metadata", {}),
                )
                regions.append(region)

            result = PageOCRResult(
                page_num=page_data["page_num"],
                width=page_data["width"],
                height=page_data["height"],
                regions=regions,
                total_tokens=page_data["total_tokens"],
                processing_time=page_data["processing_time"],
            )
            results.append(result)

        return results

    def simulate_pdf_from_text(self, text_content: str) -> List[PageOCRResult]:
        """
        Simulate PDF processing from plain text content (for testing)

        Args:
            text_content: Plain text content to simulate

        Returns:
            List of simulated PageOCRResult objects
        """
        from .deepseek_ocr import VisualRegion, BoundingBox, VisualToken, PageOCRResult
        import numpy as np

        # Split text into sections (simulate regions)
        sections = text_content.split("\n\n")
        regions = []

        y_position = 50
        for i, section in enumerate(sections):
            if section.strip():
                # Create visual tokens (simulate compression)
                tokens = []
                words = section.split()
                for j, word in enumerate(words):
                    token = VisualToken(
                        token_id=j,
                        embedding=np.random.rand(256),  # Random embedding
                        confidence=0.95,
                        region_type="text",
                        spatial_position=(j * 0.1, 0),
                        compression_method="simulated",
                    )
                    tokens.append(token)

                # Determine block type
                block_type = "paragraph"
                if section.startswith("#"):
                    block_type = "header"
                elif "|" in section or "Model" in section and "Score" in section:
                    block_type = "table"
                elif section.startswith("Abstract:"):
                    block_type = "abstract"

                # Create visual region
                region = VisualRegion(
                    region_id=f"region_{i}",
                    page_num=1,
                    block_type=block_type,
                    bbox=BoundingBox(50, y_position, 550, y_position + 100),
                    compressed_tokens=tokens,
                    text_content=section.strip(),
                    markdown_content=section.strip(),
                    token_count=len(words),
                    confidence=0.95,
                    metadata={"simulated": True, "section_index": i},
                )
                regions.append(region)
                y_position += 120

        # Create page result
        total_tokens = sum(len(r.compressed_tokens) for r in regions)

        page_result = PageOCRResult(
            page_num=1,
            width=600,
            height=800,
            regions=regions,
            total_tokens=total_tokens,
            processing_time=0.5,  # Simulated processing time
        )

        print(f"Simulated OCR result: {len(regions)} regions, {total_tokens} tokens")
        return [page_result]
