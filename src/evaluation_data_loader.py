"""
Evaluation Data Loader
Loads and provides access to all pipeline output data for evaluation
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from src.models import (
    RawDocument,
    StructuredDocument,
    SequentialSteps,
    ProblemMapping,
    Step
)
from src.utils import setup_logger

logger = setup_logger(__name__)


class EvaluationDataLoader:
    """Load all pipeline outputs for evaluation"""

    def __init__(self, output_dir: str, document_id: str):
        self.output_dir = Path(output_dir)
        self.document_id = document_id
        self._data = {}

    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all necessary data for evaluation

        Returns:
            Dict with keys: 'structured', 'steps', 'problems', 'raw'
        """
        logger.info(f"Loading evaluation data for document {self.document_id}")

        try:
            self._data = {
                'structured': self._load_structured_document(),
                'steps': self._load_sequential_steps(),
                'problems': self._load_problem_mapping(),
                'raw': self._load_raw_document()
            }
            logger.info("All evaluation data loaded successfully")
            return self._data

        except Exception as e:
            logger.error(f"Failed to load evaluation data: {e}")
            raise

    def _load_structured_document(self) -> StructuredDocument:
        """Load structured_document.json"""
        file_path = self._get_file_path("structured_document")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return StructuredDocument(**data)

    def _load_sequential_steps(self) -> SequentialSteps:
        """Load sequential_steps.json"""
        file_path = self._get_file_path("sequential_steps")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return SequentialSteps(**data)

    def _load_problem_mapping(self) -> ProblemMapping:
        """Load problem_mapping.json"""
        file_path = self._get_file_path("problem_mapping")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return ProblemMapping(**data)

    def _load_raw_document(self) -> RawDocument:
        """Load raw_document.json"""
        file_path = self._get_file_path("raw_document")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return RawDocument(**data)

    def _get_file_path(self, file_type: str) -> Path:
        """Get file path for given type"""
        # Find file with pattern: *_{document_id}_{file_type}.json
        pattern = f"*{self.document_id}_{file_type}.json"
        matches = list(self.output_dir.glob(pattern))

        if not matches:
            raise FileNotFoundError(f"No file found matching pattern: {pattern}")

        if len(matches) > 1:
            logger.warning(f"Multiple files found for {file_type}, using first one")

        return matches[0]

    # Helper methods for extracting specific sections
    def extract_purpose_text(self) -> Optional[str]:
        """Extract purpose section text"""
        if 'structured' not in self._data:
            self.load_all_data()
        return self._data['structured'].sections.get('purpose')

    def extract_background_text(self) -> Optional[str]:
        """Extract background section text"""
        if 'structured' not in self._data:
            self.load_all_data()
        return self._data['structured'].sections.get('background')

    def extract_data_text(self) -> Optional[str]:
        """Extract data section text"""
        if 'structured' not in self._data:
            self.load_all_data()
        return self._data['structured'].sections.get('data')

    def extract_pipeline_text(self) -> Optional[str]:
        """Extract pipeline section text"""
        if 'structured' not in self._data:
            self.load_all_data()
        return self._data['structured'].sections.get('pipeline')

    def extract_plan_text(self) -> Optional[str]:
        """Extract plan section text"""
        if 'structured' not in self._data:
            self.load_all_data()
        return self._data['structured'].sections.get('plan')

    def get_step_related_chunks(self, step_id: str) -> List[str]:
        """
        Get text chunks related to a specific step

        Simple approach: collect all text blocks from raw_document
        that contain keywords from the step's action/techniques

        Args:
            step_id: Step ID to find related chunks for

        Returns:
            List of text chunks
        """
        if 'steps' not in self._data or 'raw' not in self._data:
            self.load_all_data()

        # Find the step
        step = None
        for s in self._data['steps'].steps:
            if s.step_id == step_id:
                step = s
                break

        if not step:
            logger.warning(f"Step {step_id} not found")
            return []

        # Extract keywords from step
        keywords = []
        keywords.extend(step.techniques)

        # Add important words from action (simple split)
        action_words = step.action.split()
        keywords.extend([w for w in action_words if len(w) > 3])

        # Collect matching blocks
        related_chunks = []
        for page in self._data['raw'].pages:
            for block in page.blocks:
                text = block.text.lower()
                # Check if any keyword appears in text
                for keyword in keywords:
                    if keyword.lower() in text:
                        related_chunks.append(block.text)
                        break

        return related_chunks

    def get_all_steps(self) -> List[Step]:
        """Get all steps"""
        if 'steps' not in self._data:
            self.load_all_data()
        return self._data['steps'].steps
