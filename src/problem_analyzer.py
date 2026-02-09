"""
Problem Definition Agent
Identifies technical challenges each step addresses using search API
"""
import json
import re
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from threading import Lock

from src.models import (
    SequentialSteps, ProblemMapping, StepProblems,
    Problem, Severity
)
from src.utils import (
    Config, LLMRouter, CacheManager, load_prompts, setup_logger,
    extract_json_array, extract_json_object
)
from src.search_client import SearchClient

logger = setup_logger(__name__)


class ProblemAnalyzer:
    """Analyze problems that each pipeline step solves"""

    def __init__(
        self,
        config: Config,
        llm_client: LLMRouter,
        cache_manager: CacheManager,
        search_client: SearchClient
    ):
        self.config = config
        self.llm = llm_client
        self.cache = cache_manager
        self.search = search_client
        self.prompts = load_prompts()

    def analyze(self, sequential_steps: SequentialSteps) -> ProblemMapping:
        """
        Analyze problems for each step

        Args:
            sequential_steps: SequentialSteps from decomposer

        Returns:
            ProblemMapping with problems indexed by step_id
        """
        logger.info(f"Analyzing problems for {len(sequential_steps.steps)} steps")

        problem_mapping = {}
        problem_index = {}
        category_counts = {}  # Dynamic counting of categories
        critical_steps = []
        total_problems = 0
        counter = count(1)
        lock = Lock()

        def allocate_problem_id() -> str:
            with lock:
                return f"PROB_{next(counter):03d}"

        max_workers = self.config.get('problem.max_workers', 4)
        logger.debug(f"Starting parallel problem analysis with {max_workers} workers for {len(sequential_steps.steps)} steps")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._analyze_step, step, allocate_problem_id): step
                for step in sequential_steps.steps
            }
            logger.debug(f"Submitted {len(futures)} step analysis tasks")

            for future in as_completed(futures):
                step = futures[future]
                try:
                    problems = future.result()
                    logger.debug(f"[Thread] Completed {step.step_id}: {len(problems)} problems found")
                except Exception as e:
                    logger.error(f"Step analysis failed for {step.step_id}: {e}")
                    continue

                if not problems:
                    logger.debug(f"[Thread] {step.step_id}: No problems identified")
                    continue

                problem_mapping[step.step_id] = StepProblems(problems=problems)
                total_problems += len(problems)

                for problem in problems:
                    problem_index.setdefault(problem.problem_id, []).append(step.step_id)
                    cat = problem.category.lower() if isinstance(problem.category, str) else str(problem.category)
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    if problem.severity == Severity.HIGH:
                        critical_steps.append(step.step_id)

        mapping = ProblemMapping(
            document_id=sequential_steps.document_id,
            problem_mapping=problem_mapping,
            summary={
                "total_problems": total_problems,
                "by_category": category_counts,
                "critical_steps": list(set(critical_steps))
            },
            problem_index=problem_index
        )

        logger.info(f"Identified {total_problems} problems across {len(problem_mapping)} steps")
        return mapping

    def _analyze_step(self, step, allocate_problem_id) -> List[Problem]:
        """
        Analyze problems for a single step (based on INPUT → OUTPUT transformation)

        Args:
            step: Step object
            allocate_problem_id: Function to generate unique problem IDs

        Returns:
            List of Problem objects
        """
        logger.debug(f"[Thread] Analyzing {step.step_id}: {step.input} → {step.output}")
        problems = []

        # Generate cache key based on input/output transformation
        transformation_key = f"{step.input}→{step.output}"

        # Check cache first
        cached_problems = self.cache.get_technique_problems(transformation_key) if self.cache else None

        if cached_problems:
            logger.info(f"Cache hit for transformation: {transformation_key}")
            problem_desc = cached_problems[0]  # Use first cached result
        else:
            # Search for problems in this INPUT → OUTPUT transformation
            problem_desc = self._search_technique_problem(step)

            if problem_desc and self.cache:
                # Cache the result
                self.cache.set_technique_problems(transformation_key, [problem_desc])

        if problem_desc:
            # Use LLM to analyze and categorize the problem
            problem = self._create_problem(
                problem_id=allocate_problem_id(),
                description=problem_desc,
                technique=f"{step.input} → {step.output}",  # Use transformation as technique
                step=step
            )

            if problem:
                problems.append(problem)

        # Fallback: If no search results, infer from INPUT → OUTPUT directly
        if not problems:
            problem = self._infer_problem_from_action(
                problem_id=allocate_problem_id(),
                step=step
            )
            if problem:
                problems.append(problem)

        return problems

    def _search_technique_problem(self, step) -> Optional[str]:
        """
        Search for what problem exists in the INPUT → OUTPUT transformation

        Args:
            step: Step object with input/output information

        Returns:
            Problem description string or None
        """
        try:
            # Generate search queries using LLM (based on INPUT → OUTPUT)
            queries = self._generate_io_queries(step)

            if not queries:
                # Fallback to simple query based on input/output
                queries = [f"{step.input} to {step.output} transformation challenges"]

            # Search with the first query (can be extended to try multiple)
            results = self.search.search(queries[0])
            logger.debug(f"Search for '{step.input}→{step.output}': {len(results) if results else 0} results found")

            if not results:
                logger.warning(f"No search results for transformation: {step.input} → {step.output}")
                return None

            # Combine search results
            search_context = "\n\n".join([
                f"Title: {r['title']}\nContent: {r['content'][:300]}"
                for r in results[:3]
            ])
            logger.debug(f"Search context preview for '{step.input}→{step.output}': {search_context[:200]}...")

            # Use LLM to summarize (using io_result_summarizer or fallback)
            prompt_template = self.prompts.get('io_result_summarizer', {}).get('prompt', '') or \
                            self.prompts.get('search_result_summarizer', {}).get('prompt', '')

            if prompt_template:
                # Try to format with input/output first, fallback to technique
                try:
                    prompt = prompt_template.format(
                        input=step.input,
                        output=step.output,
                        search_results=search_context
                    )
                except KeyError:
                    # Fallback to technique-based format
                    prompt = prompt_template.format(
                        technique=f"{step.input} → {step.output}",
                        search_results=search_context
                    )

                summary = self.llm.generate(prompt, complexity="low")
                return summary.strip()

        except Exception as e:
            logger.error(f"Search failed for transformation {step.input}→{step.output}: {e}")

        return None

    def _generate_io_queries(self, step) -> List[str]:
        """
        Generate search queries based on INPUT → OUTPUT transformation using LLM

        Returns:
            List of search query strings
        """
        try:
            prompt_template = self.prompts.get('io_query_generator', {}).get('prompt', '')
            if not prompt_template:
                # Fallback to old search_query_generator
                return self._generate_search_queries_fallback(step)

            context = f"Action: {step.action}"
            prompt = prompt_template.format(
                input=step.input,
                output=step.output,
                context=context
            )

            response = self.llm.generate(prompt, complexity="low")

            # Extract JSON array from response
            queries = extract_json_array(response)
            if queries and isinstance(queries, list):
                logger.info(f"Generated {len(queries)} queries for {step.input}→{step.output}")
                return queries

        except Exception as e:
            logger.warning(f"IO query generation failed for {step.input}→{step.output}: {e}")

        return []

    def _generate_search_queries_fallback(self, step) -> List[str]:
        """
        Fallback: Generate search queries using old technique-based method

        Returns:
            List of search query strings
        """
        try:
            prompt_template = self.prompts.get('search_query_generator', {}).get('prompt', '')
            if not prompt_template:
                return []

            # Use first technique if available
            technique = step.techniques[0] if step.techniques else step.action
            context = f"Action: {step.action}, Input: {step.input}, Output: {step.output}"
            prompt = prompt_template.format(
                technique=technique,
                context=context
            )

            response = self.llm.generate(prompt, complexity="low")

            # Extract JSON array from response
            queries = extract_json_array(response)
            if queries and isinstance(queries, list):
                logger.info(f"Generated {len(queries)} queries (fallback) for {technique}")
                return queries

        except Exception as e:
            logger.warning(f"Fallback query generation failed: {e}")

        return []

    def _create_problem(
        self,
        problem_id: str,
        description: str,
        technique: str,
        step
    ) -> Optional[Problem]:
        """Create Problem object from description (technique param kept for compatibility)"""
        try:
            # Use LLM to categorize and assess severity (based on INPUT → OUTPUT)
            prompt_template = self.prompts.get('problem_analyzer', {}).get('prompt', '')
            system = self.prompts.get('problem_analyzer', {}).get('system', '')

            if not prompt_template:
                # Fallback: create problem with default values
                return Problem(
                    problem_id=problem_id,
                    category="uncategorized",
                    description=description,
                    severity=Severity.MEDIUM,
                    evidence=f"{step.input} → {step.output}",
                    addressed_by=technique,
                    confidence="inferred"
                )

            # Format prompt with INPUT/OUTPUT focus
            prompt = prompt_template.format(
                action=step.action,
                input=step.input,
                output=step.output
            )

            response = self.llm.generate(prompt, system=system, complexity="high")
            logger.debug(f"Problem analysis response for '{step.input}→{step.output}': {response[:150]}...")

            # Parse JSON response
            problem_data = extract_json_object(response)

            if problem_data:
                logger.debug(f"Parsed problem: {problem_id} | category={problem_data.get('category')} | severity={problem_data.get('severity')}")
                return Problem(
                    problem_id=problem_id,
                    category=str(problem_data.get('category', 'uncategorized')),
                    description=problem_data.get('description', description),
                    severity=Severity(problem_data.get('severity', 'medium')),
                    evidence=problem_data.get('evidence', f"{step.input} → {step.output}"),
                    addressed_by=technique,
                    confidence="derived"
                )

        except Exception as e:
            logger.warning(f"Problem creation failed: {e}")

        return None

    def _infer_problem_from_action(self, problem_id: str, step) -> Optional[Problem]:
        """Infer problem when no techniques are specified (based on INPUT → OUTPUT)"""
        try:
            prompt_template = self.prompts.get('problem_analyzer', {}).get('prompt', '')
            system = self.prompts.get('problem_analyzer', {}).get('system', '')

            # Use INPUT → OUTPUT for problem inference
            prompt = prompt_template.format(
                action=step.action,
                input=step.input,
                output=step.output
            )

            response = self.llm.generate(prompt, system=system, complexity="high")
            problem_data = extract_json_object(response)

            if problem_data:
                return Problem(
                    problem_id=problem_id,
                    category=str(problem_data.get('category', 'uncategorized')),
                    description=problem_data.get('description', ''),
                    severity=Severity(problem_data.get('severity', 'low')),
                    evidence=problem_data.get('evidence', f"{step.input} → {step.output}"),
                    addressed_by=step.action,
                    confidence="inferred"
                )

        except Exception as e:
            logger.warning(f"Problem inference failed: {e}")

        return None


if __name__ == "__main__":
    # Add parent directory to path for imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Test problem analyzer
    from src.pdf_extractor import PDFExtractor
    from src.structure_parser import StructureParser
    from src.step_decomposer import StepDecomposer
    from src.utils import Config, LLMRouter, CacheManager
    from src.search_client import SearchClient

    config = Config()
    llm = LLMRouter(config)
    cache = CacheManager(config)
    search = SearchClient(config)

    extractor = PDFExtractor()
    parser = StructureParser(config, llm)
    decomposer = StepDecomposer(config, llm)
    analyzer = ProblemAnalyzer(config, llm, cache, search)

    raw_doc = extractor.extract("sample/중간보고서_자연어처리.pdf")
    structured = parser.parse(raw_doc)
    steps = decomposer.decompose(structured)
    problems = analyzer.analyze(steps)

    print(f"Total problems: {problems.summary['total_problems']}")
    print(f"By category: {problems.summary['by_category']}")
    print(f"Critical steps: {problems.summary['critical_steps']}")

    for step_id, step_problems in problems.problem_mapping.items():
        print(f"\n{step_id}:")
        for problem in step_problems.problems:
            print(f"  - {problem.description} [{problem.severity.value}]")
