# agent.py

import os
import json
import re
from typing import TypedDict, Annotated, List ,Optional
import operator
import pandas as pd
from io import StringIO
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import io
from WebScrape import scrape_and_extract
from tools import smart_data_loader, python_repl_tool, duckdb_query_tool , scrape_web_page
from TableSelector import select_best_tables_by_metadata
from MostAppropriateTableidx import select_best_dataframe

from utils import (
    load_csv_or_excel,
    load_pdf_text,
    read_text_file,
    encode_image_to_base64,
    run_user_script_sandboxed
)

import os
from dotenv import load_dotenv
load_dotenv()

key=os.getenv("OPENAI_API_KEY")



# --- LLM and Tool Setup ---


llm = ChatOpenAI(
    model_name="gpt-5-nano",
    openai_api_key=key,
    openai_api_base="https://api.openai.com/v1",
    )

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0,
#     google_api_key="AIzaSyAro8d246YYA4IMfZOagZIuz8nzH9TFWcg"
# )


# --- Tool Definitions ---
tools = [
    Tool.from_function(
        func=smart_data_loader,
        name="smart_data_loader",
        description=(
            "Use this for a URL. It scrapes the page and returns the content."
        ),
    ),
    Tool.from_function(
        func=python_repl_tool.run,
        name="python_repl_tool",
        description=(
            "Executes Python code with pandas/matplotlib for analysis."
        ),
    ),
    Tool.from_function(
        func=duckdb_query_tool,
        name="duckdb_query_tool",
        description=(
           "Executes DuckDB SQL queries."
        ),
    ),
]
tool_map = {tool.name: tool for tool in tools}


# --- Graph State Definition ---
# class AgentState(TypedDict):
#     messages: Annotated[list, operator.add]
#     original_request: str
#     data_context: str
#     question: str
#     plan: str
#     critic_feedback: str
#     execution_result: str
#     execution_error: str
#     revision_number: int
#     run_script: str
#     file_paths: list[str]      # NEW: To hold paths to all attachments
#     data_context: str          # Will hold text extracted from all files
#     image_b64_data: list[str]  # NEW: To hold encoded image data
#     task_type: str             # NEW: The result from the router
#     user_script_to_run: str    # NEW: Path to the script to execute
#     dataframe_for_analysis: Optional[pd.DataFrame] = None

class AgentState(TypedDict):
    # Core fields for the agent's conversational state and planning
    messages: Annotated[list, operator.add]
    question: str
    plan: str
    critic_feedback: str
    execution_result: str
    execution_error: str
    revision_number: int

    # Fields for the new multi-modal, multi-task workflow
    file_paths: list[str]
    data_context: str          # Holds text from all files for context
    image_b64_data: list[str]
    task_type: str
    user_script_to_run: str
    
    # Dedicated field for the DataFrame to prevent errors
    dataframe_for_analysis: Optional[pd.DataFrame]



# --- Node Definitions ---

# def prepare_data_node(state: AgentState):
#     """
#     Entry Point: Prepares and CLEANS the data by either scraping a URL or extracting raw data.
#     """
#     request = state['original_request']
#     raw_data_context = ""
#     question = ""
    
#     # Step 1: Get the raw data
#     url_match = re.search(r'https?://\S+', request)
#     if url_match:
#         print("--- Found URL, scraping data... ---")
#         url = url_match.group(0)
#         question = request.replace(url, '').strip()
#         raw_data_context = scrape_web_page(url)
#         print("--- Scraping complete. ---")
#     else:
#         print("--- Found raw data, splitting... ---")
#         request_parts = request.split('question -')
#         if len(request_parts) < 2:
#             raw_data_context = request
#             question = "Summarize and describe this data."
#         else:
#             raw_data_context = request_parts[0].strip()
#             question = request_parts[1].strip()
#         print("--- Splitting complete. ---")
    
#     # Step 2: Clean the raw data using an LLM
#     print("--- Cleaning data context... ---")
#     cleaning_prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are an expert data cleaning utility. You will be given a block of text that may contain messy data (e.g., from a web scrape). Your task is to extract and return ONLY the clean, valid CSV data from this text. Remove any surrounding text, explanations, HTML tags, or other non-CSV artifacts. Your output should be ONLY the raw CSV data, starting with the header row."""),
#         ("user", "{messy_data}")
#     ])
#     cleaning_chain = cleaning_prompt | llm
#     cleaned_data_response = cleaning_chain.invoke({"messy_data": raw_data_context}) 
#     cleaned_data = cleaned_data_response.content.strip()

#     cleaned_data = re.sub(r"```[a-zA-Z]*\n", "", cleaned_data).replace("```", "").strip()
    
#     # Step 3: Update the state with the cleaned data and question
#     state['data_context'] = cleaned_data
#     state['question'] = question
#     print("--- Data cleaning complete. ---")
#     print(cleaned_data)

        
#     return state


def data_ingestion_node(state: AgentState):
    """
    Entry Point: Ingests data from file paths, separating text and images.
    """
    all_text_context = []
    image_data = []
    df_for_analysis = None 

    for file_path in state['file_paths']:
        print(f"--- Processing file: {file_path} ---")
        try:
            if file_path.endswith(('.csv', '.xls', '.xlsx')):
                df = load_csv_or_excel(file_path)
                df_for_analysis = df # Store the actual DataFrame
                all_text_context.append(f"--- Data from {os.path.basename(file_path)} ---\n{df.head().to_string()}") # Only add a preview to the text context
            elif file_path.endswith('.pdf'):
                text = load_pdf_text(file_path)
                all_text_context.append(f"--- Content from {file_path} ---\n{text}")
            elif file_path.endswith(('.py', '.sql', '.txt')):
                script_content = read_text_file(file_path)
                all_text_context.append(f"--- Script from {file_path} ---\n```\n{script_content}\n```")
            elif file_path.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                b64_image = encode_image_to_base64(file_path)
                image_data.append(b64_image)
            else:
                print(f"Warning: Unsupported file type {file_path}")

        except Exception as e:
            all_text_context.append(f"Error processing {file_path}: {e}")

    state['data_context'] = "\n\n".join(all_text_context)
    state['image_b64_data'] = image_data
    state['dataframe_for_analysis'] = df_for_analysis
    print("--- Data ingestion complete. ---")
    return state



def prepare_data_node(state: AgentState):
    """
    Entry Point: Prepares data and a question for the agent.
    - If a URL is in the request, it uses the scraping tool to find the most relevant table.
    - If no URL is present, it assumes the request is raw data and a question, separated by 'question -'.
    """
    request = state['original_request']
    data_context = ""
    question = ""
    
    # Step 1: Check for a URL and process accordingly
    url_match = re.search(r'https?://\S+', request)
    if url_match:
        print("--- URL detected. Using smart scraping tool... ---")
        url = url_match.group(0)
        question = request.replace(url, '').strip()
        if '?' in question:
            question=[p.strip() + '?' for p in question.split('?') if p.strip()]
            question=" ".join(question)

        data_text,data_context=scrape_and_extract(url)
        best_idx, best_score, top_indices = select_best_tables_by_metadata(question, url)
        
        preview_list_of_top_4_df=[]
        list_of_top_4_df=[]

        for i in top_indices:
            list_of_top_4_df.append(data_context[i])
            preview_list_of_top_4_df.append(data_context[i].head())



        table_index_to_Choose=select_best_dataframe(preview_list_of_top_4_df,question)

        data_context=list_of_top_4_df[table_index_to_Choose]

    else:
        print("--- Raw data detected. Parsing request... ---")
        # For raw data, we expect the format: <data> question - <question>
        pattern = r"(?i)(?:ques*|q\.)"
        parts = re.split(pattern, request)
        if len(parts) >= 2:
            data_context = parts[0].strip()
            # Re-join the rest in case 'question -' appears in the question itself
            question = " question - ".join(parts[1:]).strip()
        else:
            # If the format is wrong, we don't predefine a question.
            # We pass the raw data and let a later node handle the missing question.
            print("Warning: Could not find 'question -' separator. Treating entire input as data.")
            data_context = request
            question = "" 

    # Step 2: Update the state. No extra cleaning is needed.
    state['data_context'] = data_context
    print(data_context.head())
    state['question'] = question
    print("--- Data preparation complete. ---")
    print(f"question is {question}")
    
    return state

# In agent.py, ADD this new function somewhere before the graph assembly.

def task_router_node(state: AgentState):
    """
    Classifies the user's request into a specific task type.
    """
    question = state['question'].lower() # Use a lowercase version for checks
    
    # --- ADD THIS NEW LOGIC BLOCK ---
    sql_keywords = ['sql', 'query', 'group by', 'filter where','database']
    if any(keyword in question for keyword in sql_keywords):
        print("--- Router classified task as: sql_analysis (Keyword detected) ---")
        state['task_type'] = 'sql_analysis'
        return state

    url_match = re.search(r'https?://\S+', state['question'])
    if url_match:
        print("--- Router classified task as: web_scraping (URL detected) ---")
        state['task_type'] = 'web_scraping'
        return state

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a task routing expert. Based on the user's question and provided file types, classify the task. Your response must be one of the following exact strings:
        - 'python_analysis': If the user wants to analyze tabular data (CSV/Excel), needs a chart generated, or perform complex calculations.
        - 'summarization_qa': If the user wants a summary or asks a question about the content of text files (PDF/TXT) or images.
        - 'script_execution': If the user explicitly asks to run an attached Python or SQL script."""),
        ("user", "Question: {question}\n\nFile Paths: {file_paths}")
    ])
    
    router_chain = prompt | llm
    
    result = router_chain.invoke({
        "question": state['question'],
        "file_paths": [os.path.basename(p) for p in state['file_paths']] # Send just filenames
    })
    
    task_type = result.content.strip()
    print(f"--- Router classified task as: {task_type} ---")
    state['task_type'] = task_type
    
    if task_type == 'script_execution':
        # Find the first python script in the attachments to run
        script_path = next((p for p in state['file_paths'] if p.endswith('.py')), None)
        if script_path:
            state['user_script_to_run'] = script_path
        else:
            state['execution_result'] = "Error: Task is 'script_execution' but no Python script was provided."
            state['task_type'] = 'error' # Route to end
            
    return state

# In agent.py, use this as the definitive web_scraping_node

def web_scraping_node(state: AgentState):
    """
    Performs the streamlined "Scrape Once, Rank After" workflow.
    """
    print("--- Entering Web Scraping Node ---")
    question_text = state['question']
    
    url_match = re.search(r'https?://\S+', question_text)
    if not url_match:
        state['execution_error'] = "No URL found in the question for web scraping."
        state['task_type'] = 'error'
        return state
        
    url = url_match.group(0)
    question_for_analysis = question_text.replace(url, '').strip()

    try:
        # Step 1: Scrape ONCE to get a list of cleaned tables.
        _data_text, all_cleaned_tables = scrape_and_extract(url)

        if not all_cleaned_tables:
            raise ValueError("scrape_and_extract returned no tables.")

        # Step 2: Pass the CLEANED tables to the AI selector for ranking.
        best_idx, _best_score, _top_indices = select_best_tables_by_metadata(
            question=question_for_analysis,
            tables=all_cleaned_tables
        )

        preview_arr_of_top_4_df=[]
        arr_of_top_4_df=[]

        for i in _top_indices:
            arr_of_top_4_df.append(all_cleaned_tables[i])
            preview_arr_of_top_4_df.append(all_cleaned_tables[i].head())


        table_index_to_Choose=select_best_dataframe(preview_arr_of_top_4_df,question_for_analysis)


        
        # Step 3: Select the best DataFrame using the correct index.
        final_df = arr_of_top_4_df[table_index_to_Choose]

        # Step 4: Populate the state correctly for the next node.
        if final_df is not None and not final_df.empty:
            state['dataframe_for_analysis'] = final_df
            state['data_context'] = f"Data scraped from {url}:\n{final_df.head().to_string()}"
            state['question'] = question_for_analysis
        else:
            raise ValueError("No valid DataFrame was selected after ranking.")

    except Exception as e:
        # Step 5: Handle any errors gracefully.
        state['execution_error'] = f"Failed during web scraping. Error: {e}"
        state['task_type'] = 'error'

    return state

def summarization_qa_node(state: AgentState):
    """Handles summarization and general Q&A over text and images."""
    print("--- Entering Summarization/QA Node ---")
    
    user_prompt = state['question']
    
    message_content = [{"type": "text", "text": f"Context from files:\n{state['data_context']}\n\nQuestion:\n{user_prompt}"}]
    
    for b64_image in state['image_b64_data']:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
        })

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based on the provided text context and images."),
        ("user", message_content), 
    ])
    
    chain = prompt | llm
    result = chain.invoke({})
    
    state['execution_result'] = result.content
    return state

def script_executor_node(state: AgentState):
    """Executes the user-provided script."""
    print(f"--- Executing user script: {state['user_script_to_run']} ---")
    result = run_user_script_sandboxed(state['user_script_to_run'])
    state['execution_result'] = result
    return state

def worker_node(state: AgentState):
    """
    Worker Node: Generates analysis code based on the prepared data.
    """
    data_context_preview = state["dataframe_for_analysis"].head().to_string()

    full_data_context = state['data_context']

    system_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert data analyst Python programmer.
         A pandas DataFrame named `df` will be created with the user's data.
         Here is the FULL data to help you understand its content, columns, and types:
         --- FULL DATA ---
         {data_context}
         ---
         Your task is to write ONLY the Python code to perform the analysis on the `df` DataFrame based on the user's question.
         Your final output MUST be ONLY the raw Python code. Do NOT include markdown or explanations.
         The final answer of your script MUST be assigned to a variable named `result` .
         """
        ),
        ("user", "{user_prompt}"),
    ])
    
    chain = system_prompt | llm

    if state['revision_number'] == 0:
        user_prompt = state['question']
    else:
        user_prompt = (
            f"Your previous code was syntactically incorrect. Please write a new script to answer the original question based on the feedback.\n\n"
            f"--- ORIGINAL QUESTION ---\n{state['question']}\n\n"
            f"--- SYNTAX FEEDBACK ---\n{state['critic_feedback']}"
        )

    result = chain.invoke({
        "data_context": data_context_preview,
        "user_prompt": user_prompt
    })
    state["plan"] = result.content
    state["revision_number"] += 1
    print(result.content)
    print(user_prompt)
    return state

# def critic_node(state: AgentState):
#     # data_preview = pd.read_csv(StringIO(state['data_context'])).head().to_string()
#     full_data_context = state['data_context']
#     data_context_preview=state["data_context"].head()
#     """
#     Critic Node: Checks for basic Python syntax errors only.
#     """
#     system_prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          """You are an expert Python syntax checker. Your ONLY job is to check the provided Python code for syntax errors. Do not check for logical errors.
#          The plan is:
#          {plan}

#          Here is a preview of data to check the column datatypes and whether they are correctly used in code:
#          {data}

#          If the code is syntactically correct and will compile without a `SyntaxError`, respond with the single word 'approve'.
#          If the code has syntax errors, provide clear, concise feedback on how to fix ONLY the syntax."""
#         ),
#         ("user",    "the Failed Code is {plan}  and sample data is {data} make sure the'")

#     ])
    
#     chain = system_prompt | llm
#     result = chain.invoke({"plan": state["plan"],"data":data_context_preview})
#     state["critic_feedback"] = result.content
#     print(f"--- Critic (Syntax Check) Feedback: ---\n{result.content}\n--------------------")
#     return state

# def final_boss_node(state: AgentState):
#     """
#     Final Boss Node: Debugs logical and runtime errors from the execution node.
#     """
#     full_data_context = state['data_context']

#     data_context_preview=state["data_context"].head()

#     # data_preview = pd.read_csv(StringIO(state['data_context'])).head().to_string()
#     system_prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          """You are the ultimate Question solving expert, the 'Final Boss'.
#          A Python script has failed during execution with a logical or runtime error.
#          Your task is to analyze  the FAILED CODE, and the EXECUTION ERROR to provide a corrected version of ONLY THE ANALYSIS CODE.
#          Assume the data is already stored in a variavle named df as csv 

#          --- A Sample preview of fields in data -- 
#          {data}
#          --- FAILED CODE (ANALYSIS PART ONLY) ---
#          {plan}

#          --- EXECUTION ERROR ---
#          {error}

#          IMPORTANT:
#          - Your final output must be ONLY the corrected, raw Python code for the analysis part.
#          - Do NOT add any new data loading code (like `pd.read_csv` or `pd.read_html`).
#          - Do NOT include markdown or any explanations.
#          - MUST Fix the Error in the failed code Do not add anything else 
#          """
#         ),
        
#     ])

#     chain = system_prompt | llm
#     corrected_code_response = chain.invoke({
#         "question": state['question'],
#         "plan": state['plan'],
#         "error": state['execution_error'],
#         "data":data_context_preview,
#     })

#     # Update the plan with the corrected code and clear the error for the next attempt
#     state["plan"] = corrected_code_response.content
#     state["execution_error"] = "" 
#     print(f"--- Final Boss Corrected Code ---\n{state['plan']}\n--------------------")
#     return state

def critic_node(state: AgentState):
    """
    Critic Node: Checks for basic Python syntax errors only.
    """
    # Use the DataFrame for the data preview, not the text context
    data_context_preview = state["dataframe_for_analysis"].head().to_string()
    
    system_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert Python syntax checker. Your ONLY job is to check the provided Python code for syntax errors. Do not check for logical errors.
         The plan is:
         {plan}

         Here is a preview of data to check the column datatypes and whether they are correctly used in code:
         {data}

         If the code is syntactically correct and will compile without a `SyntaxError`, respond with the single word 'approve'.
         If the code has syntax errors, provide clear, concise feedback on how to fix ONLY the syntax."""
        ),
        ("user",    "the Failed Code is {plan}  and sample data is {data} make sure the'")

    ])
    
    chain = system_prompt | llm
    result = chain.invoke({"plan": state["plan"], "data": data_context_preview}) # Pass the correct preview
    state["critic_feedback"] = result.content
    print(f"--- Critic (Syntax Check) Feedback: ---\n{result.content}\n--------------------")
    return state

# FINAL CORRECTED final_boss_node

def final_boss_node(state: AgentState):
    """
    Final Boss Node: Debugs logical and runtime errors from the execution node.
    """
    # Use the DataFrame for the data preview, not the text context
    data_context_preview = state["dataframe_for_analysis"].head().to_string()

    system_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are the ultimate Question solving expert, the 'Final Boss'.
         A Python script has failed during execution with a logical or runtime error.
         Your task is to analyze  the FAILED CODE, and the EXECUTION ERROR to provide a corrected version of ONLY THE ANALYSIS CODE.
         Assume the data is already stored in a variavle named df as csv 

         --- A Sample preview of fields in data -- 
         {data}
         --- FAILED CODE (ANALYSIS PART ONLY) ---
         {plan}

         --- EXECUTION ERROR ---
         {error}

         IMPORTANT:
         - Your final output must be ONLY the corrected, raw Python code for the analysis part.
         - Do NOT add any new data loading code (like `pd.read_csv` or `pd.read_html`).
         - Do NOT include markdown or any explanations.
         - MUST Fix the Error in the failed code Do not add anything else 
         """
        ),
        
    ])

    chain = system_prompt | llm
    corrected_code_response = chain.invoke({
        "question": state['question'],
        "plan": state['plan'],
        "error": state['execution_error'],
        "data": data_context_preview, # Pass the correct preview
    })
    
    state["plan"] = corrected_code_response.content
    state["execution_error"] = "" 
    print(f"--- Final Boss Corrected Code ---\n{state['plan']}\n--------------------")
    return state


def execution_node(state: AgentState):
    """
    Execution Node: Assembles and runs the script, capturing errors for the Final Boss.
    """
    try:
        df = state['dataframe_for_analysis']
        if df is None:
            raise ValueError("No DataFrame found for analysis.")
        
        csv_data = df.to_csv(index=False)
        
        # Data Cleaning Step
        # cleaning_prompt = ChatPromptTemplate.from_messages([
        #     ("system", """You are an expert data cleaning utility. You will be given a block of text that may contain messy data (e.g., from a web scrape). Your task is to extract and return ONLY the clean, valid CSV data from this text. Remove any surrounding text, explanations, HTML tags, or other non-CSV artifacts. Your output should be ONLY the raw CSV data, starting with the header row."""),
        #     ("user", "{messy_data}")
        # ])
        # cleaning_chain = cleaning_prompt | llm
        # cleaned_data_response = cleaning_chain.invoke({"messy_data": data_context})
        # cleaned_data = cleaned_data_response.content.strip()
        # cleaned_data = re.sub(r"```[a-zA-Z]*\n", "", data_context).replace("```", "").strip()

        # Data Loading Script Creation
        data_loading_script = f"""
import pandas as pd
from io import StringIO

df = pd.read_csv(StringIO('''{csv_data}'''))

"""
        data_printing_scrupt= f"""
print(result)
        """
        # Combine and execute
        analysis_code = state["plan"].strip().replace("```python", "").replace("```", "").strip()
        full_script = data_loading_script + "\n" + analysis_code + "\n" + data_printing_scrupt
        print(f"--- EXECUTING SCRIPT ---\n{full_script}\n--------------------")
        print(f"Full script is === {full_script}")
        result = python_repl_tool.invoke(full_script)

        # --- FIX: Check the result string for error indicators ---
        result_str = str(result)
        error_indicators = ["Error", "Exception", "Traceback", "ModuleNotFoundError", "KeyError", "SyntaxError", "ParserError"]
        
        is_error = False
        if result is not None:
            for indicator in error_indicators:
                if indicator in result_str:
                    is_error = True
                    break
        
        if is_error:
            # On failure, record the error for the Final Boss
            print(f"--- Execution Failed! Error reported by tool: {result_str} ---")
            state["execution_error"] = result_str
            state["execution_result"] = "" # Clear result
        else:
            # On success
            state["execution_result"] = result_str
            state["run_script"]=full_script
            state["execution_error"] = "" # Clear error on success

    except Exception as e:
        # Catch other exceptions that might occur within this node itself
        print(f"--- Execution Node Crashed! Error: {e} ---")
        state["execution_error"] = str(e)

    return state

# --- Edge Logic ---
def should_continue(state: AgentState):
    """Checks critic feedback for syntax approval."""
    if state["revision_number"] > 3:
        return "end_failure"
    if state["critic_feedback"].strip().lower() == "approve":
        return "execute"
    else:
        return "revise"

def after_execution(state: AgentState):
    """Checks for execution errors to decide if debugging is needed."""
    if state["execution_error"]:
        print("--- Execution error detected, routing to Final Boss for debugging. ---")
        return "debug"
    else:
        print("--- Execution successful, ending workflow. ---")
        print(f"Output is : {state["execution_result"]}")
        return "end"

# --- Graph Assembly ---
workflow = StateGraph(AgentState)


workflow.add_node("data_ingestion", data_ingestion_node)
workflow.add_node("task_router", task_router_node)
workflow.add_node("prepare_data", prepare_data_node)
workflow.add_node("worker", worker_node)
workflow.add_node("critic", critic_node)
workflow.add_node("executor", execution_node)
workflow.add_node("final_boss", final_boss_node) # New node
workflow.add_node("summarizer_qa", summarization_qa_node)
workflow.add_node("script_executor", script_executor_node)
workflow.add_node("web_scraper", web_scraping_node)



# workflow.set_entry_point("prepare_data")

# workflow.add_edge("prepare_data", "worker")
# workflow.add_edge("worker", "critic")


workflow.set_entry_point("data_ingestion")

# Define the edges
workflow.add_edge("data_ingestion", "task_router")

def route_after_router(state: AgentState):
    if state.get("task_type") == 'error':
        return "end_error"
    return state["task_type"]

workflow.add_conditional_edges(
    "task_router",
    route_after_router,
    {
        "web_scraping": "web_scraper",
        "python_analysis": "worker",
        "summarization_qa": "summarizer_qa",
        "script_execution": "script_executor",
        "end_error": END
    }
)


workflow.add_edge("worker", "critic")
workflow.add_conditional_edges(
    "critic",
    should_continue, # Your original edge logic
    {"execute": "executor", "revise": "worker", "end_failure": END}
)
workflow.add_conditional_edges(
    "executor",
    after_execution, # Your original edge logic
    {"debug": "final_boss", "end": END}
)
workflow.add_edge("final_boss", "executor")
workflow.add_edge("web_scraper", "worker")




# # Critic loop (for syntax errors)
# workflow.add_conditional_edges(
#     "critic",
#     should_continue,
#     {"execute": "executor", "revise": "worker", "end_failure": END}
# )

# # New Executor loop (for logical/runtime errors)
# workflow.add_conditional_edges(
#     "executor",
#     after_execution,
#     {"debug": "final_boss", "end": END}
# )

# # Final Boss loops back to the executor
# workflow.add_edge("final_boss", "executor")


# --- The new branches go directly to the end ---
workflow.add_edge("summarizer_qa", END)
workflow.add_edge("script_executor", END)

# Compile the graph and export it
app = workflow.compile()
