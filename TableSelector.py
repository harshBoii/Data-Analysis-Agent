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

def fetch_tables_with_metadata(url):
    """
    Returns two parallel lists:
      - tables: a list of pd.DataFrame
      - metas:  a list of strings describing caption/title+columns for each table
    """
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    tables = []
    metas = []
    for tbl in soup.find_all("table"):
        # 1) parse into DataFrame
        df = pd.read_html(str(tbl), flavor="bs4")[0]
        tables.append(df)

        # 2) extract <caption>, if any
        cap = tbl.caption.get_text(strip=True) if tbl.caption else ""
        # 3) extract the closest previous heading tag as “title”
        title = ""
        for prev in tbl.find_all_previous():
            if prev.name in ["h1","h2","h3","h4","h5","h6"]:
                title = prev.get_text(strip=True)
                break

        # 4) extract column headers
        cols = df.columns.tolist()
        

        # Build a single metadata string
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if cap:
            parts.append(f"Caption: {cap}")
        parts.append(f"Columns: {cols}")

        metas.append(" | ".join(parts))


    return tables, metas , cols

# def select_best_tables_by_metadata(question: str,
#                                   url: str,
#                                   openai_api_key: str=key,
#                                   embedding_model: str = "text-embedding-3-large"
#                                  ) -> tuple[int, float, pd.DataFrame]:
#     """
#     Scrapes `url`, builds metadata descriptions for each HTML table,
#     ranks by cosine similarity against `question`, and returns:
#       (best_table_index, score, best_dataframe)
#     """
#     client = OpenAI(api_key=openai_api_key)

#     # 1) fetch tables + metadata strings
#     tables, metas , cols = fetch_tables_with_metadata(url)

#     # 2) embed the question
#     q_resp = client.embeddings.create(model=embedding_model, input=[question])
#     q_emb = q_resp.data[0].embedding

#     # 3) embed each table’s metadata
#     tbl_resp = client.embeddings.create(model=embedding_model, input=metas)
#     tbl_embs = [item.embedding for item in tbl_resp.data]

#     # 4) compute cosine similarities
#     sims = cosine_similarity([q_emb], tbl_embs)[0]

#     num_results = min(4, len(tables))
#     top_indices = np.argsort(sims)[-num_results:][::-1]


#     # 5) pick best
#     best_idx = int(sims.argmax())
#     best_score = float(sims[best_idx])

#     return best_idx, best_score, top_indices

def select_best_tables_by_metadata(question: str,
                                  tables: list[pd.DataFrame],  # <<< CHANGE: Takes a list of tables
                                  openai_api_key: str = key,
                                  embedding_model: str = "text-embedding-3-small"
                                 ) -> tuple[int, float, list[int]]:
    """
    Ranks a list of DataFrames by cosine similarity against a question
    and returns the top results.
    """
    client = OpenAI(api_key=openai_api_key)
    
    # 1) Generate metadata for each provided table
    metas = []
    for df in tables:
        # We can't get title/caption anymore, so we use columns, which is most important
        cols = df.columns.tolist()
        metas.append(f"Columns: {cols}")

    # 2) embed the question
    q_resp = client.embeddings.create(model=embedding_model, input=[question])
    q_emb = q_resp.data[0].embedding

    # 3) embed each table’s metadata
    tbl_resp = client.embeddings.create(model=embedding_model, input=metas)
    tbl_embs = [item.embedding for item in tbl_resp.data]

    # 4) compute cosine similarities
    sims = cosine_similarity([q_emb], tbl_embs)[0]

    # 5) get top 4 indices
    num_results = min(4, len(tables))
    top_indices = np.argsort(sims)[-num_results:][::-1].tolist()

    # 6) get best overall index and score
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])

    return best_idx, best_score, top_indices



def rag_Company(question: str,
                tables: list[pd.DataFrame],  # <<< CHANGE: Takes a list of tables
                openai_api_key: str = key,
                embedding_model: str = "text-embedding-3-small"
                ) -> tuple[int, float, list[int]]:
    """
    Ranks a list of DataFrames by cosine similarity against a question
    and returns the top results.
    """
    client = OpenAI(api_key=openai_api_key)
    
    # 1) Generate metadata for each provided table
    metas = []
    for df in tables:
        # We can't get title/caption anymore, so we use columns, which is most important
        cols = df.columns.tolist()
        metas.append(f"Columns: {cols}")

    # 2) embed the question
    q_resp = client.embeddings.create(model=embedding_model, input=[question])
    q_emb = q_resp.data[0].embedding

    # 3) embed each table’s metadata
    tbl_resp = client.embeddings.create(model=embedding_model, input=metas)
    tbl_embs = [item.embedding for item in tbl_resp.data]

    # 4) compute cosine similarities
    sims = cosine_similarity([q_emb], tbl_embs)[0]

    # 5) get top 4 indices
    num_results = min(4, len(tables))
    top_indices = np.argsort(sims)[-num_results:][::-1].tolist()

    # 6) get best overall index and score
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])

    return best_idx, best_score, top_indices
