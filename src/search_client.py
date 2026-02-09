"""
Search API integration for technique problem discovery
"""
from typing import List, Optional, Dict
import logging
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils import Config, setup_logger

logger = setup_logger(__name__)

class SearchClient:
    """Unified search client supporting multiple APIs"""

    def __init__(self, config: Config):
        self.config = config
        self.primary_api = config.get('search.primary_api', 'tavily')
        self.fallback_api = config.get('search.fallback_api', 'duckduckgo')
        self.timeout = config.get('search.timeout', 10)
        self.max_results = config.get('search.max_results', 3)
        self.eval_max_results = config.get('search.eval_max_results', 10)

        import os
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Search for information using configured API

        Args:
            query: Search query
            max_results: Override default max_results (optional)

        Returns:
            List of search results with 'title', 'content', 'url'
        """
        # Use provided max_results or default
        if max_results is None:
            max_results = self.max_results

        try:
            # Try primary API
            if self.primary_api == 'tavily':
                results = self._search_tavily(query, max_results)
            elif self.primary_api == 'duckduckgo':
                results = self._search_duckduckgo(query, max_results)
            else:
                raise ValueError(f"Unknown search API: {self.primary_api}")

            if results:
                return results

        except Exception as e:
            logger.warning(f"Primary search API failed: {e}")

        # Fallback to secondary API
        try:
            if self.fallback_api == 'duckduckgo':
                return self._search_duckduckgo(query, max_results)
            elif self.fallback_api == 'tavily':
                return self._search_tavily(query, max_results)
        except Exception as e:
            logger.error(f"Fallback search API also failed: {e}")

        return []

    def _search_tavily(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search using Tavily API"""
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not set")

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.tavily_api_key)

            response = client.search(
                query=query,
                max_results=max_results
            )

            results = []
            for result in response.get('results', [])[:max_results]:
                results.append({
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'url': result.get('url', '')
                })

            logger.info(f"Tavily search returned {len(results)} results for: {query}")
            return results

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            raise

    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search using DuckDuckGo"""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    query,
                    max_results=max_results
                ))

            results = []
            for result in search_results[:max_results]:
                results.append({
                    'title': result.get('title', ''),
                    'content': result.get('body', ''),
                    'url': result.get('href', '')
                })

            logger.info(f"DuckDuckGo search returned {len(results)} results for: {query}")
            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            raise


class WikipediaClient:
    """Wikipedia search client as additional fallback"""

    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1"

    def search(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        """
        Search Wikipedia

        Returns:
            List of results with 'title', 'content', 'url'
        """
        try:
            # Search for pages
            search_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': limit,
                'format': 'json'
            }

            with httpx.Client(timeout=10) as client:
                response = client.get(search_url, params=params)
                response.raise_for_status()
                data = response.json()

            results = []
            if len(data) >= 4:
                titles = data[1]
                descriptions = data[2]
                urls = data[3]

                for title, desc, url in zip(titles, descriptions, urls):
                    results.append({
                        'title': title,
                        'content': desc,
                        'url': url
                    })

            logger.info(f"Wikipedia search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return []


if __name__ == "__main__":
    # Add parent directory to path for imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Test search
    from src.utils import Config

    config = Config()
    search_client = SearchClient(config)

    query = "batch normalization deep learning solves what problem"
    results = search_client.search(query)

    print(f"Query: {query}")
    print(f"Results: {len(results)}\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   {result['content'][:200]}...")
        print(f"   {result['url']}\n")
