"""
Step Abstraction Agent
Decomposes data and pipeline sections into sequential steps
"""
import json
import re
from typing import List, Dict, Optional
import logging

from src.models import StructuredDocument, SequentialSteps, Step
from src.utils import Config, LLMRouter, load_prompts, setup_logger

logger = setup_logger(__name__)

class StepDecomposer:
    """Decompose pipeline into sequential steps"""

    def __init__(self, config: Config, llm_client: LLMRouter):
        self.config = config
        self.llm = llm_client
        self.prompts = load_prompts()

    def decompose(self, structured_doc: StructuredDocument) -> SequentialSteps:
        """
        Decompose data and pipeline sections into steps

        Args:
            structured_doc: StructuredDocument with 6 sections

        Returns:
            SequentialSteps with ordered list of steps
        """
        logger.info(f"Decomposing steps for document {structured_doc.document_id}")

        # Extract data and pipeline sections
        data_content = structured_doc.sections.get('data', '') or ''
        pipeline_content = structured_doc.sections.get('pipeline', '') or ''

        if not data_content and not pipeline_content:
            logger.warning("Both data and pipeline sections are empty")
            return SequentialSteps(
                document_id=structured_doc.document_id,
                steps=[],
                metadata={
                    "total_steps": 0,
                    "data_steps_count": 0,
                    "pipeline_steps_count": 0
                }
            )

        # Use LLM to extract steps
        steps = self._extract_steps_with_llm(data_content, pipeline_content)

        # Assign step IDs
        for i, step in enumerate(steps, 1):
            step.step_id = f"STEP_{i:03d}"

        # Count steps by category (free-form categories now)
        category_counts = {}
        for s in steps:
            cat = s.category.lower()
            category_counts[cat] = category_counts.get(cat, 0) + 1

        sequential_steps = SequentialSteps(
            document_id=structured_doc.document_id,
            steps=steps,
            metadata={
                "total_steps": len(steps),
                "category_counts": category_counts,
                "unique_categories": list(category_counts.keys())
            }
        )

        # Log category distribution
        cat_summary = ", ".join([f"{k}: {v}" for k, v in category_counts.items()])
        logger.info(f"Extracted {len(steps)} steps ({cat_summary})")
        return sequential_steps

    def _extract_steps_with_llm(
        self,
        data_content: str,
        pipeline_content: str
    ) -> List[Step]:
        """
        Use LLM to extract steps from content

        Returns:
            List of Step objects
        """
        prompt_template = self.prompts.get('step_extractor', {}).get('prompt', '')
        system = self.prompts.get('step_extractor', {}).get('system', '')

        if not prompt_template:
            logger.error("Step extractor prompt not found")
            return []

        prompt = prompt_template.format(
            data_content=data_content[:2000],  # Limit length
            pipeline_content=pipeline_content[:3000]
        )

        try:
            response = self.llm.generate(prompt, system=system, complexity="high")

            # Extract JSON from response
            steps_data = self._extract_json(response)

            if not steps_data:
                logger.error("Failed to parse JSON from LLM response")
                return []

            # Convert to Step objects
            steps_list = []
            for step_dict in steps_data:
                try:
                    # Map order to dependencies
                    deps = step_dict.get('dependencies', [])
                    if isinstance(deps, list) and deps:
                        # Convert order indices to step_ids (will be assigned later)
                        step_ids = []
                        for d in deps:
                            if isinstance(d, int):
                                step_ids.append(f"STEP_{d:03d}")
                            elif isinstance(d, str):
                                # Already in STEP_XXX format
                                step_ids.append(d)
                        deps = step_ids

                    step_instance = Step(
                        step_id="",  # Will be assigned later
                        order=step_dict.get('order', 0),
                        category=str(step_dict.get('category', 'uncategorized')),  # Free-form string
                        action=step_dict.get('action', ''),
                        input=step_dict.get('input', ''),
                        output=step_dict.get('output', ''),
                        techniques=step_dict.get('techniques', []),
                        dependencies=deps
                    )
                    steps_list.append(step_instance)

                except Exception as e:
                    logger.warning(f"Failed to parse step: {e}")
                    continue

            # Sort by order
            steps_list.sort(key=lambda s: s.order)

            return steps_list

        except Exception as e:
            logger.error(f"Step extraction failed: {e}")
            return []

    def _extract_json(self, text: str) -> Optional[List[Dict]]:
        """Extract JSON array from text"""
        # Try to find JSON array
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")

        return None


if __name__ == "__main__":
    # Add parent directory to path for imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Test step decomposer
    from src.pdf_extractor import PDFExtractor
    from src.structure_parser import StructureParser
    from src.utils import Config, LLMRouter

    config = Config()
    llm = LLMRouter(config)

    extractor = PDFExtractor()
    parser = StructureParser(config, llm)
    decomposer = StepDecomposer(config, llm)

    raw_doc = extractor.extract("sample/중간보고서_자연어처리.pdf")
    structured = parser.parse(raw_doc)
    steps = decomposer.decompose(structured)

    print(f"Document ID: {steps.document_id}")
    print(f"Total steps: {steps.metadata['total_steps']}")

    for step in steps.steps:
        print(f"\n{step.step_id} (Order {step.order}): {step.action}")
        print(f"  Category: {step.category}")
        print(f"  Techniques: {step.techniques}")
        print(f"  Dependencies: {step.dependencies}")
