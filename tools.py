import os
import requests
from bs4 import BeautifulSoup
import pandas as pd 
from langchain_openai import OpenAI
from langchain.tools import tool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
import asyncio
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from playwright.async_api import async_playwright
from typing import Type
import duckdb
import pandas as pd
import traceback
import warnings
import subprocess
import traceback
import tempfile
from PyPDF2 import PdfReader
import base64
import subprocess
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import numpy as np 
from typing import List

from dotenv import load_dotenv
load_dotenv()

key=os.getenv("OPENAI_API_KEY")




# Instantiate the tool for agent use





class OpenAI_Rag:
    """
    A Retrieval-based QA system using OpenAI embeddings without a generative chat model.
    """
    def __init__(self, api_key: str=key):
        """
        Initializes the OpenAI client.
        """
        self.client = OpenAI(api_key=api_key)
        self.documents = []
        self.embeddings = None
        print("OpenAI client initialized.")

    def build_index(self, documents: List[str], model: str = "text-embedding-3-small"):
        """
        Creates vector embeddings for a list of documents using OpenAI's API.
        
        Args:
            documents: A list of strings, where each string is a chunk of text.
            model: The embedding model to use.
        """
        self.documents = documents
        print(f"Building index for {len(documents)} documents using '{model}'...")
        
        # 1. Call the OpenAI Embeddings API
        response = self.client.embeddings.create(input=documents, model=model)
        
        # 2. Extract the embeddings and store them as a NumPy array
        self.embeddings = np.array([item.embedding for item in response.data])
        print("Index built successfully.")

    def get_answer(self, question: str, model: str = "text-embedding-3-small") -> str:
        """
        Finds and returns the most relevant document chunk for a given question.
        """
        if self.embeddings is None:
            return "Please build the index first by calling the .build_index() method."

        # 1. Embed the user's question
        response = self.client.embeddings.create(input=[question], model=model)
        question_embedding = np.array([response.data[0].embedding])

        # 2. Calculate cosine similarity between the question and all documents
        # The result is a 2D array, so we take the first (and only) row
        sims = cosine_similarity(question_embedding, self.embeddings)[0]

        # 3. Find the index of the most similar document
        most_similar_idx = np.argmax(sims)

        # 4. Return the original text of the most relevant document
        return self.documents[most_similar_idx]



class DuckDbQueryInput(BaseModel):
    query: str = Field(description="A complete and valid DuckDB SQL query to execute.")

class DuckDbQueryTool(BaseTool):
    name: str = "duckdb_query_tool"
    # --- THIS DESCRIPTION IS THE ONLY CHANGE ---
    description: str = (
        "Executes duckbs sql queries , Analyze: Study the user's request then again restudy the user's request  and the provided data context (schema, sample rows, examples , Format of Columns and closely examining date-types format) to understand the task then generate and  recheck for error in the query  "
        "On error, returns SQL_ERROR: <message> instead of throwing."
        "In case of an error change your query based upon the error"
    )
    args_schema: Type[BaseModel] = DuckDbQueryInput

    def _run_query(self, query: str) -> pd.DataFrame:
        """Helper function to connect to DuckDB and run a query."""
        try:
            with duckdb.connect(':memory:') as con:
                con.execute("INSTALL httpfs; LOAD httpfs;")
                con.execute("INSTALL parquet; LOAD parquet;")
                return con.execute(query).fetchdf()
        except Exception as e:
            return pd.DataFrame([{"error": str(e)}])

    def _run(self, query: str) -> str:
        """Executes the query and returns the result as a JSON string."""
        df_result = self._run_query(query)
        return df_result.to_json(orient='records', indent=2)

    async def _arun(self, query: str) -> str:
        """Asynchronously executes the query and returns the result as a JSON string."""
        return self._run(query)

# Create a single instance of your new generic tool
duckdb_query_tool = DuckDbQueryTool()

# Input schema remains the same
class PlaywrightToolInput(BaseModel):
    url: str = Field(description="The URL of the webpage to visit.")
    selector: str = Field(description="The CSS selector to locate the element on the page.")

class PlaywrightScrapingTool(BaseTool):
    # CHANGE 2: Add the ': str' type annotations here
    name: str = "playwright_scraper"
    description: str = (
        "Useful for scraping dynamic, JavaScript-heavy websites. "
        "Use this tool when you need to extract specific information from a modern website "
        "like an e-commerce site (e.g., Flipkart, Amazon). "
        "Provide a URL and a CSS selector."
    )
    args_schema: Type[BaseModel] = PlaywrightToolInput

    def _run(self, url: str, selector: str) -> str:
        """Use the tool synchronously."""
        # Create a new asyncio event loop to run the async method
        return asyncio.run(self._arun(url=url, selector=selector))


    async def _arun(self, url: str, selector: str) -> str:
        """Use the tool asynchronously."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                print(f"Navigating to {url}...")
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                
                print(f"Waiting for selector '{selector}'...")
                await page.wait_for_selector(selector, timeout=15000)
                
                element = await page.query_selector(selector)
                if not element:
                    return f"Error: Element with selector '{selector}' not found."
                
                content = await element.inner_text()
                await browser.close()
                
                return content.strip()

        except Exception as e:
            return f"An error occurred: {str(e)}"

# Create a single instance of your tool for easy import
playwright_tool = PlaywrightScrapingTool()



@tool
def scrape_web_page(url: str) -> str:
    """
    Scrapes the text content from a given web page URL.
    It returns the clean text content of the page, ready for analysis.
    For example, you can pass 'https://en.wikipedia.org/wiki/List_of_highest-grossing_films'.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Will raise an HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
            
        return soup.get_text(separator='\n', strip=True)
    except requests.RequestException as e:
        return f"Error scraping URL {url}: {e}"
    


def smart_data_loader(input_str: str):
    if input_str.startswith("http://") or input_str.startswith("https://"):
        response = requests.get(input_str)
        soup = BeautifulSoup(response.content, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return "No tables found on the page."
        dfs = pd.read_html(str(tables))  # Requires lxml
        return f"{len(dfs)} table(s) found. First 5 rows of first table:\n{dfs[0].head()}"
    
    # Handle CSV or Excel file content (base64 or raw bytes passed as string)
    try:
        if input_str.endswith(".csv"):
            with open(input_str, "r", encoding="utf-8") as f:
                df = pd.read_csv(f)
        elif input_str.endswith(".xlsx"):
            with open(input_str, "rb") as f:
                df = pd.read_excel(f)
        else:
            return "Unsupported file format. Provide a .csv or .xlsx file or a URL."
        return f"Data loaded. Preview:\n{df.head()}"
    except Exception as e:
        return f"Error reading file: {e}"




python_repl_tool = PythonREPLTool()





agent_tools = [scrape_web_page, python_repl_tool , smart_data_loader ]
