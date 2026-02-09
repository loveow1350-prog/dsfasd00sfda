"""
Structure & Section Parsing Agent
Uses VLM to create meaningful text chunks and LLM to classify them into 5 categories
"""
import re
import json
import hashlib
from typing import Dict, Optional, List
from tqdm import tqdm
import logging, colorlog
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models import RawDocument, StructuredDocument
from src.utils import Config, LLMRouter, CacheManager, load_prompts, setup_logger, extract_json_array

logger = setup_logger(__name__)

class StructureParser:
    """Parse document structure using VLM + LLM approach"""

    def __init__(self, config: Config, llm_client: LLMRouter, cache_manager: Optional[CacheManager] = None):
        self.config = config
        self.llm = llm_client
        self.cache = cache_manager
        self.prompts = load_prompts()

    def parse(self, raw_doc: RawDocument) -> StructuredDocument:
        """
        Parse document into 6 sections using VLM + LLM

        Workflow:
        1. Extract blocks and images from pages
        2. Use VLM to create meaningful text chunks from image + blocks
        3. Use LLM to classify each chunk into 5 categories
        4. Merge chunks by category

        Args:
            raw_doc: RawDocument from PDF extractor

        Returns:
            StructuredDocument with 6-key dictionary
        """
        logger.info(f"Parsing document {raw_doc.document_id}")

        # Step 1: Create text chunks using VLM
        chunks = self._create_chunks_with_vlm(raw_doc)
        logger.info(f"Created {len(chunks)} text chunks")

        # Step 2: Classify each chunk using LLM
        chunk_classifications = self._classify_chunks_with_llm(chunks)
        logger.info(f"Classified {len(chunk_classifications)} chunks")

        # Step 3: Merge chunks by category
        sections = self._merge_chunks_by_category(chunks, chunk_classifications)

        # Handle missing sections
        missing_sections = [k for k, v in sections.items() if v is None or v == ""]

        topic_generation = False
        if 'topic' in missing_sections:
            logger.warning("Topic section is missing or empty -> using LLM fallback")
            sections['topic'] = self.topic_classification_fallback(chunks)
            topic_generation = True

        structured_doc = StructuredDocument(
            document_id=raw_doc.document_id,
            sections=sections,
            metadata={
                "missing_sections": missing_sections,
                "total_chunks": len(chunks),
                "chunk_classifications": chunk_classifications,
                "generated_topic": topic_generation
            }
        )

        logger.info(f"Parsing complete. Missing sections: {missing_sections}")
        return structured_doc

    def _create_chunks_with_vlm(self, raw_doc: RawDocument) -> List[Dict]:
        """
        Create meaningful text chunks using VLM (Vision Language Model)

        For each page (processed in parallel):
        1. Send page image + extracted blocks to VLM
        2. Ask VLM to merge blocks into coherent chunks
        3. VLM returns structured chunks with merged text

        Args:
            raw_doc: Raw document with pages

        Returns:
            List of chunks: [{"text": str, "page": int}, ...]
        """
        chunks = []
        max_workers = self.config.get('parser.max_workers', 4)

        # Filter pages with blocks
        valid_pages = [page for page in raw_doc.pages if page.blocks]

        if not valid_pages:
            return chunks

        logger.info(f"Processing {len(valid_pages)} pages in parallel (max_workers={max_workers})")
        logger.debug(f"Pages to process: {[p.page_num for p in valid_pages]}")

        def process_page(page):
            """Process a single page and return chunks"""
            logger.debug(f"[Thread] Started processing page {page.page_num}")
            # Prepare blocks text with bbox coordinates
            blocks_text = "\n".join([
                f"Block {i+1} [x:{block.bbox[0]:.0f}, y:{block.bbox[1]:.0f}, w:{block.bbox[2]-block.bbox[0]:.0f}, h:{block.bbox[3]-block.bbox[1]:.0f}]: {block.text}"
                for i, block in enumerate(page.blocks)
            ])

            # Check if VLM is available (GPT-4V, Claude with vision, Gemini)
            if page.image_data and self._supports_vision():
                logger.info(f"Using VLM for page {page.page_num}")
                # Use VLM with image
                result = self._vlm_create_chunks(page.image_data, blocks_text, page.page_num)
                logger.debug(f"[Thread] Page {page.page_num} completed: {len(result)} chunks created")
                return result
            else:
                logger.info(f"Using LLM for page {page.page_num}")
                # Fallback: Use LLM with text only
                result = self._llm_create_chunks(blocks_text, page.page_num)
                logger.debug(f"[Thread] Page {page.page_num} completed: {len(result)} chunks created")
                return result

        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_page, page): page for page in valid_pages}

            # Use tqdm for progress tracking
            with tqdm(total=len(valid_pages), desc="Creating chunks with VLM/LLM") as pbar:
                for future in as_completed(futures):
                    page = futures[future]
                    try:
                        page_chunks = future.result()
                        chunks.extend(page_chunks)
                    except Exception as e:
                        logger.error(f"Failed to process page {page.page_num}: {e}")
                    finally:
                        pbar.update(1)

        # Sort chunks by page number to maintain order
        chunks.sort(key=lambda x: x.get('page', 0))

        return chunks

    def _supports_vision(self) -> bool:
        """Check if current LLM provider supports vision"""
        return self.llm.supports_vision()

    def _vlm_create_chunks(self, image_data: bytes, blocks_text: str, page_num: int) -> List[Dict]:
        """
        Use VLM to create chunks from image + text

        Args:
            image_data: PNG image bytes
            blocks_text: Extracted text blocks
            page_num: Page number

        Returns:
            List of chunks
        """
        try:
            # Get prompt from config
            prompt_template = self.prompts.get('vlm_chunk_creator', {}).get('prompt', '')
            system = self.prompts.get('vlm_chunk_creator', {}).get('system', '')

            if not prompt_template:
                logger.warning("No vlm_chunk_creator prompt found, using default")
                prompt_template = \
                    """
                        Merge these text blocks into coherent chunks:
                        
                        {blocks_text}
                        
                        Return JSON array:
                        [
                          {{"text": "merged text", "page": {page_num}}}
                        ]
                    """

            prompt = prompt_template.format(
                blocks_text=blocks_text,
                page_num=page_num
            )

            # Add explicit JSON instruction
            json_instruction = "\n\nIMPORTANT: Respond with ONLY a valid JSON array. No explanation, no markdown, just the JSON array starting with [ and ending with ]."
            prompt = prompt + json_instruction

            # Use LLMClient's generate_with_image method
            response = self.llm.generate_with_image(prompt, image_data, system=system)

            # Log full response for debugging
            logger.info(f"VLM response for page {page_num} (length: {len(response) if response else 0}):")
            logger.info(f"Response preview: {response[:300] if response else 'EMPTY'}...")
            logger.info(f"Response end: ...{response[-100:] if response and len(response) > 100 else response}")

            # Extract JSON array from response (handling markdown code blocks)
            chunks = extract_json_array(response)
            if chunks and isinstance(chunks, list):
                logger.info(f"✅ VLM parsed {len(chunks)} chunks from page {page_num}")
                return chunks

            logger.warning(f"❌ No JSON array found in VLM response for page {page_num}")
            logger.warning(f"Full response:\n{response}")

            # Fallback: create single chunk from all blocks
            logger.info(f"Using fallback: creating single chunk from blocks (page {page_num})")
            return [{"text": blocks_text, "page": page_num}]

        except Exception as e:
            logger.warning(f"VLM chunk creation failed: {e}. Falling back to LLM.")
            return self._llm_create_chunks(blocks_text, page_num)

    def _llm_create_chunks(self, blocks_text: str, page_num: int) -> List[Dict]:
        """
        Fallback: Use text-only LLM to create chunks

        Args:
            blocks_text: Extracted text blocks
            page_num: Page number

        Returns:
            List of chunks
        """
        try:
            # Get prompt from config
            prompt_template = self.prompts.get('llm_chunk_creator', {}).get('prompt', '')
            system = self.prompts.get('llm_chunk_creator', {}).get('system', '')

            if not prompt_template:
                logger.warning("No llm_chunk_creator prompt found, using default")
                prompt_template = \
                    """
                        Merge these text blocks into coherent chunks:
                        
                        {blocks_text}
                        
                        Return JSON array:
                        [
                          {{"text": "merged text", "page": {page_num}}}
                        ]
                    """

            prompt = prompt_template.format(
                blocks_text=blocks_text,
                page_num=page_num
            )

            # Add explicit JSON instruction
            json_instruction = "\n\nIMPORTANT: Respond with ONLY a valid JSON array. No explanation, no markdown, just the JSON array starting with [ and ending with ]."
            prompt = prompt + json_instruction

            response = self.llm.generate(prompt, system=system, complexity="high")

            # Log full response for debugging
            logger.info(f"LLM response for page {page_num} (length: {len(response) if response else 0}):")
            logger.info(f"Response preview: {response[:300] if response else 'EMPTY'}...")
            logger.info(f"Response end: ...{response[-100:] if response and len(response) > 100 else response}")

            # Extract JSON array from response (handling markdown code blocks)
            chunks = extract_json_array(response)
            if chunks and isinstance(chunks, list):
                logger.info(f"✅ LLM parsed {len(chunks)} chunks from page {page_num}")
                return chunks

            logger.warning(f"❌ No JSON array found in LLM response for page {page_num}")
            logger.warning(f"Full response:\n{response}")

            # Fallback: create single chunk from all blocks
            logger.info(f"Using fallback: creating single chunk from blocks (page {page_num})")
            return [{"text": blocks_text, "page": page_num}]

        except Exception as e:
            logger.error(f"LLM chunk creation failed: {e}")
            # Final fallback: just use blocks as-is
            return [{"text": blocks_text, "page": page_num}]

    def _classify_chunks_with_llm(self, chunks: List[Dict]) -> Dict[int, str]:
        """
        Classify each chunk into one of 5 categories using LLM

        Args:
            chunks: List of text chunks

        Returns:
            Dict mapping chunk_index -> category
        """
        classifications = {}

        if not chunks:
            return classifications

        batch_template = self.prompts.get('chunk_classifier_batch', {}).get('prompt', '')
        batch_system = self.prompts.get('chunk_classifier_batch', {}).get('system', '')

        # Fallback to single-chunk template
        single_template = self.prompts.get('chunk_classifier', {}).get('prompt', '')
        single_system = self.prompts.get('chunk_classifier', {}).get('system', '')

        if not batch_template:
            batch_template = (
                "You are classifying text chunks into one of six categories: "
                "topic, purpose, background, data, pipeline, plan. Given the JSON array 'chunks', "
                "return JSON array of objects: [{{\"idx\": <int>, \"category\": <one of five>}}].\n"
                "chunks: {chunks}"
            )

        batch_size = self.config.get('parser.chunk_batch_size', 20)
        valid_categories = {'topic', 'purpose', 'background', 'data', 'pipeline', 'plan'}

        uncached = []
        for idx, chunk in enumerate(chunks):
            text = chunk.get('text', '') or ''
            if len(text) < 10:
                continue

            chunk_hash = self._hash_text(text)
            cached = self.cache.get_chunk_category(chunk_hash) if self.cache else None
            if cached in valid_categories:
                classifications[idx] = cached
                continue

            uncached.append((idx, text, chunk_hash))

        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            payload = [{"idx": idx, "text": text[:500]} for idx, text, _ in batch]
            index_to_hash = {idx: chunk_hash for idx, _, chunk_hash in batch}

            prompt = batch_template.format(chunks=json.dumps(payload, ensure_ascii=False))

            try:
                response = self.llm.generate(prompt, system=batch_system, complexity="low")
                parsed = self._extract_json(response)
            except Exception as e:
                logger.error(f"Batch classification failed: {e}")
                parsed = None

            if not parsed:
                logger.warning("Falling back to per-chunk classification for this batch")
                for idx, text, chunk_hash in batch:
                    category = self._classify_single_chunk(text, single_template, single_system, valid_categories)
                    if category:
                        classifications[idx] = category
                        if self.cache:
                            self.cache.set_chunk_category(chunk_hash, category)
                continue

            if isinstance(parsed, dict):
                parsed = parsed.get('results') or parsed.get('chunks') or []

            for item in parsed or []:
                idx = item.get('idx')
                category = str(item.get('category', '')).strip().lower()
                if idx is None:
                    continue
                if category in valid_categories:
                    classifications[idx] = category
                    if self.cache:
                        chunk_hash = index_to_hash.get(idx)
                        if chunk_hash:
                            self.cache.set_chunk_category(chunk_hash, category)
                else:
                    logger.warning(f"Invalid category '{category}' for chunk {idx}")

        return classifications

    def _classify_single_chunk(
        self,
        text: str,
        template: str,
        system: str,
        valid_categories: set
    ) -> Optional[str]:
        """Classify a single chunk (fallback)"""
        if not template:
            return "pipeline"  # Default fallback

        preview = text[:500]
        prompt = template.format(
            chunk_preview=preview,
            header="[No explicit header]",
            content_preview=preview
        )

        try:
            response = self.llm.generate(prompt, system=system, complexity="low")

            # Handle empty response
            if not response or not response.strip():
                logger.warning("Empty response from chunk classifier, using 'pipeline' as default")
                return "pipeline"

            category = response.strip().lower()

            # Check if valid category
            if category in valid_categories:
                return category

            # Try to extract category from response (in case LLM added explanation)
            for valid_cat in valid_categories:
                if valid_cat in category:
                    logger.debug(f"Extracted '{valid_cat}' from response: {response[:50]}")
                    return valid_cat

            logger.warning(f"Invalid category '{category}', using 'pipeline' as default")
            return "pipeline"  # Default fallback

        except Exception as e:
            logger.error(f"Failed to classify chunk: {e}")
            return "pipeline"  # Default fallback

    def _hash_text(self, text: str) -> str:
        """Create stable hash for caching chunk categories"""
        return hashlib.md5(text.encode()).hexdigest()

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON object/array from text"""
        match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def _merge_chunks_by_category(
        self,
        chunks: List[Dict],
        classifications: Dict[int, str]
    ) -> Dict[str, Optional[str]]:
        """
        Merge chunks by their classified category

        Args:
            chunks: List of chunks
            classifications: Mapping of chunk_index -> category

        Returns:
            Dict with 6 keys (topic, purpose, background, data, pipeline, plan)
        """
        sections = {
            'topic': [],
            'purpose': [],
            'background': [],
            'data': [],
            'pipeline': [],
            'plan': []
        }

        for idx, category in classifications.items():
            if idx < len(chunks):
                chunk_text = chunks[idx].get('text', '')
                if chunk_text:
                    sections[category].append(chunk_text)

        # Merge texts for each category
        merged_sections = {}
        for category, texts in sections.items():
            if texts:
                merged_sections[category] = '\n\n'.join(texts)
            else:
                merged_sections[category] = None

        return merged_sections

    def topic_classification_fallback(self, chunks: List[Dict[str, str]]) -> str:
        """
        Fallback method to classify topic if missing using LLM

        Args:
            chunks: Listed Dictionary of texts

        Returns:
            LLM generated topic string
        """
        topic_template = self.prompts.get('topic_fallback', {}).get('prompt', '')
        topic_system = self.prompts.get('topic_fallback', {}).get('system', '')

        if not topic_template:
            topic_template = (
                "Given the following text chunks from a document, provide a concise topic summary:\n\n"
                "{chunks}\n\n"
                "Return only the topic as a single sentence."
            )

        combined_text = "\n\n".join([chunk.get('text', '') for chunk in chunks])

        prompt = topic_template.format(text_content=combined_text)

        try:
            response = self.llm.generate(prompt, system=topic_system, complexity="high")
            topic = response.strip()
            logger.info(f"LLM fallback generated topic: {topic}")
            return topic
        except Exception as e:
            logger.error(f"Topic classification fallback failed: {e}")
            return "Unknown Topic"


if __name__ == "__main__":
    # Add parent directory to path for imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Test structure parser
    from src.pdf_extractor import PDFExtractor
    from src.utils import Config, LLMClient

    config = Config()
    llm = LLMClient(config)
    extractor = PDFExtractor()
    from src.utils import CacheManager

    cache = CacheManager(config)
    parser = StructureParser(config, llm, cache)

    raw_doc = extractor.extract("sample/중간보고서_자연어처리.pdf")
    structured = parser.parse(raw_doc)

    print(f"Document ID: {structured.document_id}")
    print(f"Missing sections: {structured.metadata['missing_sections']}")

    for section, content in structured.sections.items():
        if content:
            print(f"\n{section.upper()}: {content[:200]}...")
        else:
            print(f"\n{section.upper()}: [NOT FOUND]")
