from openai import OpenAI
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

key=os.getenv("OPENAI_API_KEY")


def select_best_dataframe(dfs: list[pd.DataFrame], question: str, model="gpt-4o-mini") -> pd.DataFrame:
    """
    Given a list of pandas DataFrames and a question, this function uses an LLM
    to decide which DataFrame is best suited to answer the question.
    """
    client = OpenAI(api_key=key)

    df_descriptions = []
    for i, df in enumerate(dfs):
        sample = df.head(3).to_dict(orient='records')
        description = f"Table {i}:\nColumns: {list(df.columns)}\nSample Rows: {sample}\n"
        df_descriptions.append(description)

    prompt = f"""

A user asks: "{question}"



You are given multiple tables. Decide which table (number) is best suited to answer the question.

Only respond with the number of the best table (0-indexed). Do not provide explanations.

IMPORTANT- Only Respond with the index of the Best Table In One Word

Tables:
{chr(10).join(df_descriptions)}
    """

    # Call LLM
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    reply = response.choices[0].message.content

    try:
        table_index = int(reply)
        return table_index 
    except (ValueError, IndexError):
        raise ValueError(f"Invalid LLM response: '{reply}'")

