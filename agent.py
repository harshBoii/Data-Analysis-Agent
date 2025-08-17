# agent.py
import pathlib
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
from tools import smart_data_loader, python_repl_tool, duckdb_query_tool , scrape_web_page, OpenAI_Rag
from TableSelector import select_best_tables_by_metadata
from MostAppropriateTableidx import select_best_dataframe
import duckdb # Make sure duckdb is imported
from langchain_text_splitters import RecursiveCharacterTextSplitter
import base64
from installer import install_dependencies
from generate_install import generate_install_script


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
    model_name="gpt-5-mini",
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

    #Classification

        # The user's original, complex question
    original_question: str

    # The list of deconstructed sub-tasks
    sub_tasks: List[dict]

    # The current task being worked on
    current_task: dict

    # A dictionary to store the answers as they are completed
    # Using Annotated for the add operator is good practice for accumulation
    completed_tasks: dict

    # The final, synthesized answer for the user
    final_answer: str




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

def is_base64_data_url(text):
    # Regex to detect base64 data URLs
    pattern = r"^data:([a-zA-Z0-9]+/[a-zA-Z0-9\-\+\.]+)?;base64,[A-Za-z0-9+/=]+$"
    if re.match(pattern, text.strip()):
        try:
            base64_part = text.split(",", 1)[1]
            base64.b64decode(base64_part, validate=True)
            return True
        except Exception:
            return False
    return False


def sql_worker_node(state: AgentState):
    """
    Generates a DuckDB SQL query to answer a single question based on the
    full context provided about the remote Parquet dataset.
    """
    print("--- Entering SQL Worker Node ---")
    
    # For this task, the 'question' field will contain the entire user prompt
    # with the schema, sample query, and all sub-questions.
    full_context_and_questions = state['question']

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert SQL data analyst. Your task is to write a single, executable SQL query to answer user's questions based on the extensive context they provide.

         The user's prompt will contain:
         1. A description of a dataset which might be even located in an S3 bucket or a Datframe Provided.
         2. The schema of the Parquet files.
         3. An example query showing how to access the data.
         4. A JSON object with one or more questions.

         Your instructions are:
         - Your output MUST be ONLY the raw SQL query. No explanations.
         - The query must be self-contained. It MUST include the necessary `INSTALL httpfs; LOAD httpfs;` and `INSTALL parquet; LOAD parquet;` commands at the beginning.
         - The `FROM` clause must correctly reference correct path provided in the user's context.
         """),
        ("user",
         """--- ORIGINAL FULL PROMPT ---
         {original_question}

         --- CURRENT SUB-QUESTION ---
         {current_sub_question}
         """)

    ])
    
    sql_chain = prompt | llm
    result = sql_chain.invoke({
        "original_question": state['original_question'],
        "current_sub_question": state['question'] # state['question'] was set by the preparer
    })
    
    print(f"--- Generated SQL Query ---\n{result.content}\n--------------------")
    state['plan'] = result.content
    return state

import duckdb

def sql_executor_node(state: AgentState):
    """
    Connects to an in-memory DuckDB instance and executes a raw SQL query.
    """
    print("--- Entering SQL Executor Node ---")
    sql_query = state.get('plan')
    
    if not sql_query:
        state['execution_error'] = "No SQL query was provided to the executor."
        state['task_type'] = 'error'
        return state

    try:
        # Connect to a fresh in-memory database
        con = duckdb.connect(database=':memory:', read_only=False)
        
        # Execute the query (which includes its own INSTALL/LOAD commands)
        # and return the result as a pandas DataFrame.

        result_df = con.execute(sql_query).df()
        print(f"Result_DF_Sql_is--------------------------------{result_df}")

        execution_result = result_df.to_string()
        
        # 2. Save the ACTUAL DataFrame object for the next node in the plan
        dataframe_for_analysis = result_df

        

        state['execution_result'] = result_df.to_string()

        return {
            "execution_result": execution_result,
            "dataframe_for_analysis": dataframe_for_analysis
        }

        
    except Exception as e:
        state['execution_error'] = f"An error occurred during SQL execution: {e}"

    print(f"Result_DF_Sql_is--------------------------------{state['execution_result']}")
    return state

def llm_synthesizer_node(state: AgentState):
    """
    Uses an LLM to take all the completed tasks and synthesize the final
    {q1: ans1, q2: ans2, ...} JSON object.
    """
    print("--- Entering LLM Synthesizer Node ---")



    
    # Prepare the context for the LLM
    completed_tasks = state.get('completed_tasks', {})

    # Create a simple, readable string of the results for the prompt
    results_summary = ""
    for question, answer in completed_tasks.items():
        results_summary += f"Question: {question}\nAnswer: {answer}\n---\n"
        awi=answer
    
    if len(results_summary)>6000:
        print("-----Aadhe Raaste--------------")
        return {"final_answer": awi}

    print("------Getting To The Prompt Part----------------")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert report synthesizer. Your job is to take a list of questions and their corresponding raw answers and format them into a clean, final JSON object.

         ---------------VERY IMPORTANT---------------------

         The final result should be a clean JSON object only with the required answer , There will be no code in the output
        

         - Clean up the answers if necessary (e.g., remove unnecessary whitespace or formatting).
         - Ensure your final output is ONLY the raw JSON object, with no surrounding text or markdown.

         --- VERY IMPORTANT --
            SUMMARISE THE ANSWER OF EACH QUESTION VERY CONCISELY , JUST ENOUGH TO ANSER THE QUESTION
         """),
        ("user",
         """Please synthesize the following results into the final JSON report:

         {results_summary}""")
    ])
    
    synthesis_chain = prompt | llm
    result = synthesis_chain.invoke({"results_summary": results_summary})
    
    # The result from the LLM is the final, clean JSON string
    final_answer_str = result.content.strip()
    
    print(f"--- Final Synthesized Report --- \n{final_answer_str}")
    
    # Return the final update to the state
    return {"final_answer": final_answer_str}



# def question_deconstructor_node(state: AgentState):
#     """
#     The Deconstructor: Analyzes the original question and breaks it down
#     into a structured list of sub-tasks.
#     """
#     print("--- Entering Question Deconstructor Node ---")
    
#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          """You are an expert project manager. Your job is to analyze a user's complex request and break it down into a list of individual, self-contained tasks.

#          You must classify each task into one of the following types: `sql_query`, `python_analysis`, `web_scraping`, or `summarization_qa`.

#          Your output MUST be a JSON list of dictionaries. Each dictionary must have two keys: "type" and "question".

#         --- HIGHEST PRIORITY RULE: URL DETECTION ---
#          If the user's request contains a URL for a website, you MUST classify that task as `web_scraping`. Do not choose any other type for tasks involving URLs.


#         --- IMPORTANT RULE #1 ---
#          Do not infer or add extra steps like summarization unless the user explicitly asks for it.

#          --- IMPORTANT RULE #2 ---
#          IF THERE ARE MULTIPLE QUESTIONS BELONGING TO SAME TYPE OF TASK THEN AGGREGATE THOSE QUESTIONS LIKE 
#             EXAMPLE:{{
#              "type": "python_analysis",
#              "question:"
#                     "q1. How many $2 bn movies were released before 2000?
#                     q2. Which is the earliest film that grossed over $1.5 bn?
#                     q3. What's the correlation between the Rank and Peak?"
#            }}

#          --- IMPORTANT RULE #4: SELF-CONTAINED PYTHON TASKS ---
#          If a task requires fetching data from a remote source (like S3) AND performing a Python-specific action (like plotting), classify it as `python_analysis`. The "question" for that step must be a complete, self-contained instruction to BOTH fetch the data AND perform the analysis.

#          Example of a single step:
#          User Request: "Process the attached CSV file and return a DataFrame with specific columns and data types."
#          Your Output:
#          [
#            {{
#              "type": "python_analysis",
#              "question": "Process the attached CSV file and return a DataFrame with specific columns and data types."
#            }}
#          ]

#          Example of multiple steps with a self-contained Python task:
#          User Request: "Which high court disposed the most cases from 2019-2022? Also, plot the delay by year for court=33_10 from the S3 dataset."
#          Your Output:
#          [
#            {{
#              "type": "sql_query",
#              "question": "Using the dataset at 's3://indian-high-court-judgments/...', which high court disposed the most cases from 2019 - 2022?"
#            }},
#            {{
#              "type": "python_analysis",
#              "question": "Using DuckDB, first fetch the 'year' and the delay in days (decision_date - date_of_registration) for court_code='33~10' from the S3 dataset at 's3://indian-high-court-judgments/...'. Then, plot the results as a scatterplot with a regression line and encode it as a Base64 data URI."
#            }}
#          ]

#          Now, deconstruct the user's request according to these strict rules."""),
#         ("user", "{original_question}")
#     ])
    
#     deconstructor_chain = prompt | llm
#     result = deconstructor_chain.invoke({"original_question": state['original_question']})

#     json_string = result.content.strip().replace("```json", "").replace("```", "")
#     sub_tasks = json.loads(json_string)

#     state['sub_tasks'] = sub_tasks
#     state['completed_tasks'] = {} # Initialize the answer dictionary
#     return state


def question_deconstructor_node(state: AgentState):
    """
    The Deconstructor: Analyzes the original question and breaks it down
    into a structured list of sub-tasks.
    """
    print("--- Entering Question Deconstructor Node ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert project manager. Your job is to analyze a user's complex request and break it down into a list of individual, self-contained tasks.

         You must classify each task into one of the following types: `python_analysis`, `web_scraping`(ONLY IF URL IS PRESENT), or `summarization_qa(IF THE QUESTION EXPLICITLY MENTIONS THE WORD LIKE SUMMARIZE)`.

         Your output MUST be a JSON list of dictionaries. Each dictionary must have two keys: "type" and "question".




        --- HIGHEST PRIORITY RULE: URL DETECTION ---
         1->If the user's request contains a URL for a website, you MUST classify that task as `web_scraping`. Do not choose any other type for tasks involving URLs.




        --- IMPORTANT RULE #1 ---
         Do not infer or add extra steps like summarization unless the user explicitly asks for it.

         --- IMPORTANT RULE #2 ---
         IF THERE ARE MULTIPLE QUESTIONS BELONGING TO SAME TYPE OF TASK THEN AGGREGATE THOSE QUESTIONS LIKE 
            EXAMPLE:{{
             "type": "python_analysis",
             "question:"
                    "q1. How many $2 bn movies were released before 2000?
                    q2. Which is the earliest film that grossed over $1.5 bn?
                    q3. What's the correlation between the Rank and Peak?"
           }}

         --- IMPORTANT RULE #4: SELF-CONTAINED PYTHON TASKS ---
         If a task requires fetching data from a remote source (like S3) AND performing a Python-specific action (like plotting), classify it as `python_analysis`. The "question" for that step must be a complete, self-contained instruction to BOTH fetch the data AND perform the analysis.

         Example of a single step:
         User Request: "Process the attached CSV file and return a DataFrame with specific columns and data types."
         Your Output:
         [
           {{
             "type": "python_analysis",
             "question": "Process the attached CSV file and return a DataFrame with specific columns and data types."
           }}
         ]

         Example of multiple steps with a self-contained Python task:
         User Request: "Which high court disposed the most cases from 2019-2022? Also, plot the delay by year for court=33_10 from the S3 dataset."
         Your Output:
         [
           {{
             "type": "python_analysis",
             "question": "Using the dataset at 's3://indian-high-court-judgments/...', which high court disposed the most cases from 2019 - 2022?"
           }},
           {{
             "type": "python_analysis",
             "question": "Using DuckDB, first fetch the 'year' and the delay in days (decision_date - date_of_registration) for court_code='33~10' from the S3 dataset at 's3://indian-high-court-judgments/...'. Then, plot the results as a scatterplot with a regression line and encode it as a Base64 data URI."
           }}
         ]

         Now, deconstruct the user's request according to these strict rules."""),
        ("user", "{original_question}")
    ])
    
    deconstructor_chain = prompt | llm
    result = deconstructor_chain.invoke({"original_question": state['original_question']})

    json_string = result.content.strip().replace("```json", "").replace("```", "")
    sub_tasks = json.loads(json_string)

    state['sub_tasks'] = sub_tasks
    state['completed_tasks'] = {} # Initialize the answer dictionary
    return state


def prepare_next_task_node(state: AgentState):
    """
    Peeks at the next task and populates the state.
    Crucially, it passes BOTH the sub-task question and the original full context.
    """
    print("--- Entering Task Preparer Node ---")
    
    if state['sub_tasks']:
        next_task = state['sub_tasks'][0]
        state['current_task'] = next_task
        # Set the 'question' for the specialist node to use
        state['question'] = next_task['question']
        # The 'original_question' containing full context remains untouched for use by workers
    
    return state

def route_tasks(state: AgentState):
    """
    Checks if there are pending tasks and returns the type of the next task
    for conditional routing.
    """
    if not state['sub_tasks']:
        return "end_loop"

    next_task_type = state['sub_tasks'][0]['type']
    print(f"--- Routing to: {next_task_type} ---")
    return next_task_type


def result_aggregator_node(state: AgentState):
    """
    Aggregates the result and correctly removes the completed task from the list.
    """
    print("--- Entering Result Aggregator Node ---")
    
    result = state.get('execution_result', state.get('execution_error', 'No result produced.'))
    
    # Update the dictionary of completed tasks
    if 'current_task' in state and state['current_task']:
        print("--- Aggregating a multi-step task result ---")
        current_question = state['current_task']['question']
        
        # Add the result to our dictionary of completed tasks
        state['completed_tasks'].update({current_question: result})
        
        # Remove the just-completed task from the sub_tasks list
        current_tasks = state.get('sub_tasks', [])
        if current_tasks:
            state['sub_tasks'] = current_tasks[1:]
    else:
        # This is a single-step task (like summarization), not part of a sub-task loop.
        print("--- Aggregating a single-step task result ---")
        original_question = state['original_question']
        
        # The final answer is the only result we have.
        state['completed_tasks'] = {original_question: result}
        
        # Since it's a single-step task, we signal that the loop is over.
        state['sub_tasks'] = []
    
    # Clear state for the next loop
    state['execution_result'] = ""
    state['execution_error'] = ""
    state['plan'] = ""
    state['revision_number'] = 0
    return state


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
    # sql_keywords = ['sql', 'query', 'group by', 'filter where','database']
    # if any(keyword in question for keyword in sql_keywords):
    #     print("--- Router classified task as: sql_analysis (Keyword detected) ---")
    #     state['task_type'] = 'sql_analysis'
    #     return state

    url_match = re.search(r'https?://\S+', state['question'])
    if url_match:
        print("--- Router classified task as: web_scraping (URL detected) ---")
        state['task_type'] = 'web_scraping'
        return state

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a task routing expert. Based on the user's question and provided file types, classify the task. Your response must be one of the following exact strings:
        - 'python_analysis': If the user wants to analyze tabular data (CSV/Excel), needs a chart generated, or perform complex calculations or HAS PYTHON CODE.
        - 'summarization_qa': If the user wants a summary or asks a question about the content of text files (PDF/TXT) or images.
        - 'script_execution': If the user explicitly asks to run an attached Python Code
        - 'presonalized_summary':If the user wants a summary or asks a question WHICH INCLUDES WORD `Smriti` '"""),
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

def script_executor_node(state: AgentState):
    """
    Executes a user-provided Python script in a sandboxed environment.
    """
    script_to_run = state.get('user_script_to_run')
    if not script_to_run:
        state['execution_error'] = "Router failed to identify a script to run."
        return state

    print(f"--- Entering Script Executor Node: Running {script_to_run} ---")
    
    # This calls the utility function you already have in your project
    result = run_user_script_sandboxed(script_to_run)
    
    state['execution_result'] = result
    return state


def web_scraping_node(state: AgentState):
    """
    Performs the streamlined "Scrape Once, Rank After" workflow.
    """
    print("--- Entering Web Scraping Node ---")
    question_text = state['question']
    
    url_match = re.search(r'https?://\S*[^\s.,\'")]', question_text)
    print
    if not url_match:
        print("------------------------No Url-----------------------------------")
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
# ------------------------------------------------------
# def summarization_qa_node(state: AgentState):
#     """Handles summarization and general Q&A over text and images."""
#     print("--- Entering Summarization/QA Node ---")
    
#     user_prompt = state['question']
    
#     message_content = [{"type": "text", "text": f"Context from files:\n{state['data_context']}\n\nQuestion:\n{user_prompt}"}]
    
#     for b64_image in state['image_b64_data']:
#         message_content.append({
#             "type": "image_url",
#             "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
#         })

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a helpful assistant. Answer the user's question based on the provided text context and images."),
#         ("user", message_content), 
#     ])
    
#     chain = prompt | llm
#     result = chain.invoke({})
    
#     state['execution_result'] = result.content
#     return state

# ---------------------------------------------------------------

def presonalized_summary_node(state:AgentState):
    print("--- Entering presonalized_summary Node ---")
    Raw_data=pathlib.Path('Smriti.txt').read_text()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # The size of each chunk in characters
        chunk_overlap=50   # Overlap helps keep context between chunks
    )

    # This creates smaller, more focused documents
    documents = text_splitter.split_text(Raw_data)
    print(f"Split context into {len(documents)} chunks.")


    Rag=OpenAI_Rag()
    Rag.build_index(documents)
    Context=Rag.get_answer(state['question'])

    print(f"Found context: {Context}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based on the provided text context and images(if provided or attatched)."),
        "user", "Question: {question}\n\nContext:\n{context}", 
    ])
    
    chain = prompt | llm
    result = chain.invoke({
    "question": state['question'],
    "context": Context
    })
    
    state['execution_result'] = result.content
    return state





def summarization_qa_node(state: AgentState):

    print("--- Entering Summarization/QA Node ---")
    if len(state["file_paths"])<1:
        Raw_data=None
        Context="Use Google"
    else:
        Raw_data=pathlib.Path(state["file_paths"][0]).read_text()
        documents = Raw_data.strip().split('\n\n')
        Rag=OpenAI_Rag()
        Rag.build_index(documents)
        Context=Rag.get_answer(state['question'])


    print(f"Found context: {Context}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based on the provided text context and images(if provided or attatched)."),
        "user", "Question: {question}\n\nContext:\n{context}", 
    ])
    
    chain = prompt | llm
    result = chain.invoke({
    "question": state['question'],
    "context": Context
    })
    
    state['execution_result'] = result.content
    return state




def worker_node(state: AgentState):
    """
    Worker Node: Generates analysis code based on the prepared data.
    """
    if state["dataframe_for_analysis"] is not None and not state["dataframe_for_analysis"].empty :
        data_context_preview = state["dataframe_for_analysis"].head().to_string()
    else:
        data_context_preview=[]

    full_data_context = state['data_context']


    system_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert data analyst Python programmer. Your task is to write a complete, self-contained Python script to answer the user's question.
            
            IMPORT ALL USED MODULES

         You have two ways to get data:
         1. If a DataFrame preview is provided, a pandas DataFrame named `df` is already loaded. You can use it directly.
         2. If no DataFrame preview is provided, you MUST write a script that loads its own data. for example like , For querying remote Parquet files on S3, ALWAYS use the `duckdb` library.

         --- EXAMPLE of a self-contained DuckDB script ---
         import duckdb
         import pandas as pd

         # Connect and load extensions
         con = duckdb.connect(database=':memory:')
         con.execute("INSTALL httpfs; LOAD httpfs;")
         con.execute("INSTALL parquet; LOAD parquet;")

         # Define the query
         sql_query = "SELECT * FROM read_parquet('s3://bucket/path/...')"
         
         # Execute and get result
         df_result = con.execute(sql_query).df()
         con.close()
         
         # Perform final calculations
         # The final answer must be assigned to a variable named 'result'
         result = df_result['column'].mean() 
         ---

        ---------VERY IMPORTANT----------
         Your final output MUST be ONLY the Python code. 
         DO NOT include markdown or EXPLANATIONS.
         The final answer of your script MUST be assigned to a class DICT variable named `result `.
         
         --- DataFrame Preview (if available) ---
         {data_context}
         --- Question is -> {user_prompt}---
         """),
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
    if state["dataframe_for_analysis"] is not None and not state["dataframe_for_analysis"].empty :
        data_context_preview = state["dataframe_for_analysis"].head().to_string()
    else:
        data_context_preview=[]
     
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
    if state["dataframe_for_analysis"] is not None and not state["dataframe_for_analysis"].empty :
        data_context_preview = state["dataframe_for_analysis"].head().to_string()
    else:
        data_context_preview=[]


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

         --- VERY EXTRA IMPORTANT ---
         RETURN ONLY THE CORRECT CODE WITH NO MARKUPS NO COMMENTS 

         IMPORTANT:
         - Your final output must be ONLY the corrected, raw Python code for the analysis part.
         - Do NOT add any new data loading code (like `pd.read_csv` or `pd.read_html`).
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
            # raise ValueError("No DataFrame found for analysis.")
            csv_data = """Dummy CSV"""
        else:
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
        data_printing_scrupt= f"""
print(result)
        """
        # Combine and execute

        analysis_code = state["plan"].strip().replace("```python", "").replace("```", "").strip()
        install_script = generate_install_script(analysis_code)
        data_loading_script = f"""
import pandas as pd
from io import StringIO
import matplotlib
matplotlib.use('Agg')


df = pd.read_csv(StringIO('''{csv_data}'''))

"""

        full_script = install_script + "\n" + data_loading_script + "\n" + analysis_code + "\n" + data_printing_scrupt
        print(f"--- EXECUTING SCRIPT ---\n{full_script}\n--------------------")
        print(f"Full script is === {full_script}")
        result = python_repl_tool.invoke(full_script)


        # --- FIX: Check the result string for error indicators ---
        result_str = result
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

# 1. Add all nodes for the Hybrid Agent
# Management Nodes
workflow.add_node("data_loader", data_ingestion_node)
workflow.add_node("task_router", task_router_node) # Your smart dispatcher
workflow.add_node("deconstructor", question_deconstructor_node) # Now a specialist
workflow.add_node("task_preparer", prepare_next_task_node)
workflow.add_node("aggregator", result_aggregator_node)
workflow.add_node("synthesizer", llm_synthesizer_node)

# Specialist Nodes
workflow.add_node("summarizer_qa", summarization_qa_node)
workflow.add_node("web_scraper", web_scraping_node)
workflow.add_node("sql_worker", sql_worker_node)
workflow.add_node("sql_executor", sql_executor_node)
workflow.add_node("worker", worker_node)
workflow.add_node("critic", critic_node)
workflow.add_node("python_executor", execution_node)
workflow.add_node("final_boss", final_boss_node)
workflow.add_node("script_executor", script_executor_node) # ADDED: New node
workflow.add_node("presonalized_summary",presonalized_summary_node)



workflow.set_entry_point("data_loader")

workflow.add_edge("data_loader", "task_router")


workflow.add_conditional_edges(
    "task_router",
    lambda state: state.get("task_type", "error"),
    {
        # Simple, one-step tasks go directly to their specialist node
        "summarization_qa": "summarizer_qa",
        "web_scraping": "web_scraper",
        "presonalized_summary":"presonalized_summary",

        # Complex, multi-step tasks are sent to the deconstructor first
        "python_analysis": "deconstructor",
        "sql_query": "deconstructor",

        "script_execution": "script_executor",
        # If something goes wrong with routing
        "error": END
    }
)

workflow.add_edge("script_executor", "aggregator") 

workflow.add_edge("deconstructor", "task_preparer")


# 3. The conditional edge uses the 'route_tasks' function to decide where to go
workflow.add_conditional_edges(
    "task_preparer",
    route_tasks,
    {
        "sql_query": "sql_worker",
        "python_analysis": "worker",
        "web_scraping": "web_scraper",
        "end_loop": "synthesizer" ,
        "summarization_qa": "summarizer_qa",
        "presonalized_summary":"presonalized_summary"
    }
)

# Simple QA and Web Scraping branches
workflow.add_edge("summarizer_qa", "aggregator")
workflow.add_edge("presonalized_summary", "aggregator")
workflow.add_edge("web_scraper", "worker") # Web scraping is followed by Python analysis


# 4. Define the paths for each specialist branch
# SQL Branch
workflow.add_edge("sql_worker", "sql_executor")
workflow.add_edge("sql_executor", "aggregator")

# Web Scraping Branch (which then becomes a Python task)
workflow.add_edge("web_scraper", "worker")

# Python Analysis Branch (with its own internal review loop)
workflow.add_edge("worker", "critic")
workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "execute": "python_executor",
        "revise": "worker",
        "end_failure": "aggregator" # If it fails, aggregate the error message
    }
)
workflow.add_conditional_edges(
    "python_executor",
    after_execution,
    {
        "debug": "final_boss",
        "end": "aggregator" # On success, aggregate the result
    }
)
workflow.add_edge("final_boss", "worker") # The boss sends corrected code back for another try

# 5. After any task is completed, the aggregator loops back to the preparer
workflow.add_edge("aggregator", "task_preparer")

# 6. The synthesizer is the final step before ending
workflow.add_edge("synthesizer", END)

# Compile the final, correct graph
app = workflow.compile()
