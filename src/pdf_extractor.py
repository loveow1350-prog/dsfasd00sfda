"""
PyMuPDF Component: PDF text and metadata extraction
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import uuid
import logging

from src.models import RawDocument, PageData, TextBlock, ExtractionQuality
from src.utils import setup_logger

logger = setup_logger(__name__)

class PDFExtractor:
    """Extract structured text and metadata from PDF documents"""

    def __init__(self):
        self.quality_threshold = 0.6

    def extract(self, pdf_path: str) -> RawDocument:
        """
        Extract text blocks with metadata from PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            RawDocument object with structured content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)

        # Extract metadata
        metadata = self._extract_metadata(doc)

        # Extract pages
        pages = []
        missing_pages = []
        encoding_errors = 0

        for page_num in range(len(doc)):
            try:
                page_data = self._extract_page(doc[page_num], page_num)
                pages.append(page_data)

                # Count encoding errors
                for block in page_data.blocks:
                    if self._has_encoding_error(block.text):
                        encoding_errors += 1

            except Exception as e:
                logger.error(f"Failed to extract page {page_num}: {e}")
                missing_pages.append(page_num)

        # Calculate quality score
        quality = self._calculate_quality(
            total_pages=len(doc),
            missing_pages=missing_pages,
            encoding_errors=encoding_errors,
            total_blocks=sum(len(p.blocks) for p in pages)
        )

        extraction_quality = ExtractionQuality(
            missing_pages=missing_pages,
            encoding_errors=encoding_errors,
            quality_score=quality
        )

        doc.close()

        raw_doc = RawDocument(
            document_id=str(uuid.uuid4()),
            metadata=metadata,
            pages=pages,
            extraction_quality=extraction_quality
        )

        logger.info(f"Extraction complete. Quality: {quality:.2f}, Pages: {len(pages)}")

        if quality < self.quality_threshold:
            logger.warning(f"Quality score {quality:.2f} below threshold {self.quality_threshold}")

        return raw_doc

    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract document metadata"""
        metadata = doc.metadata or {}
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "pages": len(doc),
            "format": metadata.get("format", "PDF")
        }

    def _extract_page(self, page: fitz.Page, page_num: int) -> PageData:
        """Extract text blocks and page image from a single page"""
        blocks = []

        # Get text blocks with formatting info (block 단위로 병합)
        text_dict = page.get_text("dict")

        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block (not image)
                # Merge all spans in a block into one TextBlock
                block_texts = []
                block_bbox = block.get("bbox", [0, 0, 0, 0])
                avg_size = 0
                font = ""
                flags = 0
                span_count = 0

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            block_texts.append(text)
                            avg_size += span.get("size", 11.0)
                            font = span.get("font", font)
                            flags |= span.get("flags", 0)
                            span_count += 1

                if block_texts:
                    merged_text = " ".join(block_texts)
                    avg_size = avg_size / span_count if span_count > 0 else 11.0

                    text_block = TextBlock(
                        bbox=list(block_bbox),
                        text=merged_text,
                        font=font,
                        size=avg_size,
                        flags=flags
                    )
                    blocks.append(text_block)

        # Sort blocks by vertical position (top to bottom)
        blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))

        # Render page as image (for VLM)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better quality
        img_bytes = pix.tobytes("png")

        return PageData(
            page_num=page_num,
            blocks=blocks,
            image_data=img_bytes  # Add image data
        )

    def _has_encoding_error(self, text: str) -> bool:
        """Check if text has encoding errors"""
        # Check for common encoding error patterns
        error_patterns = ['�', '\ufffd', '\x00']
        return any(pattern in text for pattern in error_patterns)

    def _calculate_quality(
        self,
        total_pages: int,
        missing_pages: List[int],
        encoding_errors: int,
        total_blocks: int
    ) -> float:
        """
        Calculate extraction quality score

        Formula: 0.4 * (1 - missing_ratio) + 0.3 * (1 - error_ratio) + 0.3 * layout_score
        """
        if total_pages == 0:
            return 0.0

        missing_ratio = len(missing_pages) / total_pages

        error_ratio = 0.0
        if total_blocks > 0:
            error_ratio = min(encoding_errors / total_blocks, 1.0)

        # Layout coherence: assume 1.0 if no missing pages
        layout_score = 1.0 if not missing_pages else 0.7

        quality_score = (
            0.4 * (1 - missing_ratio) +
            0.3 * (1 - error_ratio) +
            0.3 * layout_score
        )

        return max(0.0, min(1.0, quality_score))


if __name__ == "__main__":
    # Add parent directory to path for imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Re-import after path is set
    from src.models import RawDocument, PageData, TextBlock, ExtractionQuality

    # Test extraction
    extractor = PDFExtractor()

    # Try different possible PDF locations
    pdf_paths = [
        "중간보고서_자연어처리.pdf",  # Current directory
        "../중간보고서_자연어처리.pdf",  # Parent directory
        "sample/중간보고서_자연어처리.pdf",  # Sample folder
        "../sample/중간보고서_자연어처리.pdf"  # Parent's sample folder
    ]

    pdf_path = None
    for path in pdf_paths:
        if Path(path).exists():
            pdf_path = path
            break

    if not pdf_path:
        print("❌ PDF 파일을 찾을 수 없습니다.")
        print("다음 위치 중 하나에 '중간보고서_자연어처리.pdf' 파일을 배치하세요:")
        print("  - 프로젝트 루트: /mnt/c/Users/pegoo/Desktop/nlp_project_2/")
        print("  - sample 폴더: /mnt/c/Users/pegoo/Desktop/nlp_project_2/sample/")
        sys.exit(1)

    print(f"📄 PDF 파일 발견: {pdf_path}")
    result = extractor.extract(pdf_path)

    print(f"✅ Document ID: {result.document_id}")
    print(f"✅ Pages: {len(result.pages)}")
    print(f"✅ Quality: {result.extraction_quality.quality_score:.2f}")
    print(f"✅ First page blocks: {len(result.pages[0].blocks) if result.pages else 0}")
