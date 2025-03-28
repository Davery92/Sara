# modules/searxng_module.py

import requests
from bs4 import BeautifulSoup
import json
import time
import random
import re
import os
import asyncio
from urllib.parse import urljoin, urlparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logger = logging.getLogger("searxng-module")

class OllamaClient:
    def __init__(self, api_url="http://100.104.68.115:11434/api/chat", summary_model="qwen2.5:32b", briefing_model="qwen2.5:32b"):
        """
        Initialize the Ollama client
        
        Args:
            api_url (str): URL of the Ollama API
            summary_model (str): Model name to use for summaries
            briefing_model (str): Model name to use for the final briefing
        """
        self.api_url = api_url
        self.summary_model = summary_model
        self.briefing_model = briefing_model
        
    def generate_response(self, content, system_prompt=None, use_briefing_model=False):
        """
        Generate a response using the Ollama model
        
        Args:
            content (str): Content to analyze
            system_prompt (str): Optional system prompt
            use_briefing_model (bool): Whether to use the briefing model instead of the summary model
            
        Returns:
            str: Generated response
        """
        messages = []
        
        # Select which model to use
        model = self.briefing_model if use_briefing_model else self.summary_model
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add user content
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        try:
            logger.info(f"Using model: {model}")
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Error: Ollama API returned status code {response.status_code}")
                return f"Error generating response: {response.text}"
                
            result = response.json()
            return result.get("message", {}).get("content", "No response generated")
            
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error generating response: {str(e)}"
            
    def clean_response(self, response):
        """
        Clean response by removing <think> tags and their content
        
        Args:
            response (str): Original response
            
        Returns:
            str: Cleaned response
        """
        # Pattern to match <think>...</think> blocks, including nested tags
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', response, flags=re.DOTALL)
        return cleaned.strip()


class SearXNGScraper:
    def __init__(self, searxng_instance="http://10.185.1.8:4000", max_depth=2, delay=1.0):
        """
        Initialize the SearXNG scraper
        
        Args:
            searxng_instance (str): URL of the SearXNG instance to use
            max_depth (int): Maximum crawl depth for each search result
            delay (float): Base delay between requests (will add random jitter)
        """
        self.searxng_instance = searxng_instance
        self.max_depth = max_depth
        self.delay = delay
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.session = requests.Session()

    def search(self, query, num_results=5, categories=None):
        """
        Search SearXNG and return a list of result URLs
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return
            categories (list): List of SearXNG categories to search in
            
        Returns:
            list: List of result URLs
        """
        params = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        
        if categories:
            params["categories"] = categories
            
        try:
            response = self.session.get(
                f"{self.searxng_instance}/search", 
                params=params,
                headers=self.headers
            )
            
            if response.status_code != 200:
                logger.error(f"Error: SearXNG returned status code {response.status_code}")
                return []
                
            results = response.json().get("results", [])
            urls = [result.get("url") for result in results[:num_results]]
            return urls
            
        except Exception as e:
            logger.error(f"Error searching SearXNG: {e}")
            return []

    def scrape_page(self, url):
        """
        Scrape a single page and extract its content
        
        Args:
            url (str): URL to scrape
            
        Returns:
            dict: Page data including title, text content, and links
        """
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract page content (only title, url and text)
            page_data = {
                "url": url,
                "title": soup.title.text.strip() if soup.title else "No title",
                "text": self._extract_text(soup)
            }
            
            # Extract links for crawling but don't save them in the output
            links = self._extract_links(soup, url)
            
            return page_data, links
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def _extract_text(self, soup):
        """Extract main text content from page"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Break into lines and remove leading/trailing whitespace
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_links(self, soup, base_url):
        """Extract all links from page"""
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Skip anchor links and javascript
            if href.startswith('#') or href.startswith('javascript:'):
                continue
                
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Only include links to the same domain
            if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                links.append(absolute_url)
                
        return links
    
    def crawl(self, url, depth=0, visited=None):
        """
        Recursively crawl a website up to max_depth
        
        Args:
            url (str): URL to start crawling from
            depth (int): Current depth level
            visited (set): Set of already visited URLs
            
        Returns:
            list: List of page data dictionaries for all crawled pages
        """
        if visited is None:
            visited = set()
            
        if depth > self.max_depth or url in visited:
            return []
            
        # Add random delay to be respectful
        time.sleep(self.delay + random.uniform(0, 1))
        
        # Mark as visited before making request
        visited.add(url)
        
        page_result = self.scrape_page(url)
        if not page_result:
            return []
            
        page_data, links = page_result
        results = [page_data]
        
        # Follow links if not at max depth
        if depth < self.max_depth:
            for link in links[:3]:  # Limit to 3 links per page
                if link not in visited:
                    child_results = self.crawl(link, depth + 1, visited)
                    results.extend(child_results)
                    
        return results
        
    def search_and_crawl(self, query, num_results=3, categories=None, ollama_client=None, 
                         page_summary_prompt=None, briefing_prompt=None):
        """
        Search SearXNG and only process the direct links returned (no crawling)
        
        Args:
            query (str): Search query
            num_results (int): Number of search results to process
            categories (list): List of SearXNG categories to search in
            ollama_client (OllamaClient): Optional Ollama client for generating responses
            page_summary_prompt (str): Optional system prompt for individual page summaries
            briefing_prompt (str): Optional system prompt for final briefing
            
        Returns:
            dict: Search results with page data, summaries, and final briefing
        """
        urls = self.search(query, num_results, categories)
        
        if not urls:
            return {"query": query, "results": []}
            
        logger.info(f"Found {len(urls)} results for query: '{query}'")
        
        all_results = []
        all_summaries = []
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing result {i}/{len(urls)}: {url}")
            
            # Just scrape the direct page without crawling further
            page_result = self.scrape_page(url)
            
            if not page_result:
                logger.info(f"Could not scrape {url}")
                continue
                
            page_data, _ = page_result  # Ignore the links
            
            if ollama_client:
                # Prepare content for Ollama
                content = f"Title: {page_data['title']}\nURL: {page_data['url']}\n\nContent:\n{page_data['text']}"
                
                logger.info(f"Generating summary for: {page_data['url']}")
                # Use summary model
                raw_response = ollama_client.generate_response(content, page_summary_prompt, use_briefing_model=False)
                
                # Clean the response
                cleaned_response = ollama_client.clean_response(raw_response)
                
                # Add both raw and cleaned responses to the page data
                page_data['raw_summary'] = raw_response
                page_data['cleaned_summary'] = cleaned_response
                
                # Add to all summaries for briefing
                summary_info = {
                    'title': page_data['title'],
                    'url': page_data['url'],
                    'summary': cleaned_response
                }
                all_summaries.append(summary_info)
            
            all_results.append({
                "seed_url": url,
                "pages": [page_data]  # Just include the single page
            })
        
        result = {
            "query": query,
            "results": all_results
        }
        
        # Generate final briefing if we have summaries
        if ollama_client and all_summaries:
            logger.info("Generating final briefing document...")
            
            # Prepare briefing content
            briefing_content = f"Search Query: {query}\n\n"
            briefing_content += "Summaries from search results:\n\n"
            
            for i, summary in enumerate(all_summaries, 1):
                briefing_content += f"Source {i}: {summary['title']} ({summary['url']})\n"
                briefing_content += f"{summary['summary']}\n\n"
            
            # Generate briefing using briefing model
            raw_briefing = ollama_client.generate_response(briefing_content, briefing_prompt, use_briefing_model=True)
            
            # Clean the briefing
            cleaned_briefing = ollama_client.clean_response(raw_briefing)
            
            # Add briefings to the result
            result["raw_briefing"] = raw_briefing
            result["briefing"] = cleaned_briefing
            
            # Save the briefing to a separate markdown file
            save_briefing(
                briefing=cleaned_briefing,
                query=query,
                briefing_model=ollama_client.briefing_model,
                output_dir="/home/david/Sara/briefings"
            )
            
        return result


def save_results(results, output_file="search_results.json"):
    """Save results to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_file}")



def save_briefing(briefing, query, briefing_model="model", output_dir="/home/david/Sara/briefings"):
    """
    Save briefing to a markdown file in the specified directory and send notifications
    
    Args:
        briefing (str): Briefing content
        query (str): Original search query
        briefing_model (str): Name of the model that generated the briefing
        output_dir (str): Directory to save the briefing file
    """
    # Create directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a safe filename from the query
    safe_query = re.sub(r'[^\w\s-]', '', query).strip().lower()
    safe_query = re.sub(r'[-\s]+', '-', safe_query)
    
    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d")
    filename = f"{safe_query}.md"
    file_path = os.path.join(output_dir, filename)
    
    # Add a title and metadata to the markdown
    md_content = f"# Briefing: {query}\n\n"
    md_content += f"*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    md_content += f"---\n\n"
    md_content += briefing
    
    # Save the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"Briefing saved to {file_path}")
    
    # Send notifications
    try:
        # Import notification service
        from notification_service import notification_service
        
        # Send ntfy notification
        notification_service.send_briefing_completion_notification(query, filename)
        logger.info(f"Sent notification for completed briefing: {query}")
    except Exception as e:
        logger.error(f"Error sending briefing notification: {str(e)}")
    
    return file_path

# Add to the _generate_briefing_background method in SearXNGBriefingHandler
# Right after saving the briefing (where you set self.active_searches[task_id]["briefing_path"])

# Add this code:
"""
# Send a WebSocket notification to the chat UI
try:
    from notification_service import notification_service
    notification_service.send_briefing_completion_notification(query, filename)
    logger.info(f"Sent notification for completed briefing: {query}")
except Exception as e:
    logger.error(f"Error sending briefing notification: {str(e)}")
"""


class SearXNGBriefingHandler:
    """Handler for processing /briefing commands and generating reports in background"""
    
    def __init__(self, briefings_dir="/home/david/Sara/briefings"):
        self.briefings_dir = briefings_dir
        self.active_searches = {}  # Dictionary to track active search tasks
        os.makedirs(briefings_dir, exist_ok=True)
        
        # Configure default settings
        self.searxng_instance = "http://10.185.1.8:4000"
        self.ollama_url = "http://100.104.68.115:11434/api/chat"
        self.summary_model = "qwen2.5:32b"
        self.briefing_model = "qwen2.5:32b"
        self.num_results = 5
        
        # Initialize prompt templates
        self.page_summary_prompt = "Summarize this content concisely, focusing on key information related to the title."
        
    async def process_briefing_command(self, query, conversation_id=None):
        """
        Process a /briefing command by starting a background task
        
        Args:
            query (str): The search query (without the /briefing prefix)
            conversation_id (str): Optional conversation ID to associate with this search
            
        Returns:
            dict: Information about the started background task
        """
        # Generate a task ID
        task_id = f"briefing-{int(time.time())}"
        
        # Start the background task
        task = asyncio.create_task(
            self._generate_briefing_background(task_id, query, conversation_id)
        )
        
        # Store the task information
        self.active_searches[task_id] = {
            "query": query,
            "conversation_id": conversation_id,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "task": task
        }
        
        # Return information about the started task
        return {
            "task_id": task_id,
            "query": query,
            "status": "started",
            "message": f"Started briefing search for '{query}'"
        }
    
    async def _generate_briefing_background(self, task_id, query, conversation_id):
        """
        Background task to perform the search and generate a briefing
        
        Args:
            task_id (str): Unique ID for this task
            query (str): The search query
            conversation_id (str): Optional conversation ID to associate with this search
        """
        try:
            logger.info(f"Starting background briefing task {task_id} for query: '{query}'")
            
            # Update task status
            self.active_searches[task_id]["status"] = "searching"
            
            # Initialize the Ollama client
            ollama_client = OllamaClient(
                api_url=self.ollama_url,
                summary_model=self.summary_model,
                briefing_model=self.briefing_model
            )
            
            # Initialize the SearXNG scraper
            scraper = SearXNGScraper(
                searxng_instance=self.searxng_instance,
                max_depth=0  # No crawling depth needed
            )
            
            # Use asyncio to run the CPU-bound task in a thread pool
            results = await asyncio.to_thread(
                scraper.search_and_crawl,
                query=query,
                num_results=self.num_results,
                ollama_client=ollama_client,
                page_summary_prompt=self.page_summary_prompt,
                briefing_prompt=SYSTEM_PROMPT
            )
            
            # Update task information
            self.active_searches[task_id]["status"] = "completed"
            self.active_searches[task_id]["end_time"] = datetime.now().isoformat()
            
            # Get the path to the saved briefing file
            briefing_path = None
            if "briefing" in results:
                # Get the safe filename used for saving
                safe_query = re.sub(r'[^\w\s-]', '', query).strip().lower()
                safe_query = re.sub(r'[-\s]+', '-', safe_query)
                filename = f"{safe_query}.md"
                briefing_path = os.path.join(self.briefings_dir, filename)
                
                self.active_searches[task_id]["briefing_path"] = briefing_path
                self.active_searches[task_id]["briefing"] = results["briefing"]
                try:
                    from notification_service import notification_service
                    notification_service.send_briefing_completion_notification(query, filename)
                    logger.info(f"Sent notification for completed briefing: {query}")
                except Exception as e:
                    logger.error(f"Error sending briefing notification: {str(e)}")
            
            logger.info(f"Completed briefing task {task_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error in background briefing task {task_id}: {e}")
            
            # Update task status to error
            self.active_searches[task_id]["status"] = "error"
            self.active_searches[task_id]["error"] = str(e)
            self.active_searches[task_id]["end_time"] = datetime.now().isoformat()
            
            return {"error": str(e)}
    
    def get_task_status(self, task_id):
        """
        Get the status of a briefing task
        
        Args:
            task_id (str): The ID of the task to check
            
        Returns:
            dict: Information about the task status
        """
        if task_id not in self.active_searches:
            return {
                "task_id": task_id,
                "status": "not_found",
                "message": f"No task found with ID {task_id}"
            }
        
        task_info = self.active_searches[task_id].copy()
        
        # Remove the task object from the response
        if "task" in task_info:
            del task_info["task"]
        
        return {
            "task_id": task_id,
            **task_info
        }
    
    def get_latest_briefing(self, query=None):
        """
        Get the latest completed briefing, optionally filtered by query
        
        Args:
            query (str): Optional query to filter by
            
        Returns:
            dict: Information about the latest briefing
        """
        # Filter to completed tasks first
        completed_tasks = {
            task_id: info for task_id, info in self.active_searches.items()
            if info.get("status") == "completed"
        }
        
        if not completed_tasks:
            return {
                "status": "not_found",
                "message": "No completed briefings found"
            }
        
        # If query is provided, filter to matching tasks
        if query:
            matching_tasks = {
                task_id: info for task_id, info in completed_tasks.items()
                if query.lower() in info.get("query", "").lower()
            }
            
            if not matching_tasks:
                return {
                    "status": "not_found",
                    "message": f"No completed briefings found for query '{query}'"
                }
            
            # Sort by end time (latest first)
            sorted_tasks = sorted(
                matching_tasks.items(),
                key=lambda x: x[1].get("end_time", ""),
                reverse=True
            )
        else:
            # Sort all completed tasks by end time (latest first)
            sorted_tasks = sorted(
                completed_tasks.items(),
                key=lambda x: x[1].get("end_time", ""),
                reverse=True
            )
        
        # Return the latest task
        latest_task_id, latest_task_info = sorted_tasks[0]
        
        # Format the response
        response = {
            "task_id": latest_task_id,
            **latest_task_info
        }
        
        # Remove the task object from the response
        if "task" in response:
            del response["task"]
        
        return response


# Create a singleton instance
briefing_handler = SearXNGBriefingHandler()


# Default system prompt for the briefing model
SYSTEM_PROMPT = """Summarize these summaries into a full page document. Use paragraphs only, no bullet points."""