# import logging

# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("langchain").setLevel(logging.DEBUG)
# logging.getLogger("langgraph").setLevel(logging.DEBUG)
from typing import List
from fastapi import Body ,Request
import json
import re
from fastapi import FastAPI, Form, UploadFile, File
from typing import Optional
from agent import app as analysis_workflow
from langchain_core.messages import HumanMessage
import tempfile
import shutil
from agent import app as agent_app
import ast
import os
from fastapi.middleware.cors import CORSMiddleware
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Optional
import tempfile
import os
import shutil
import re





app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that can analyze data from a URL or an uploaded file.",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# This class is no longer used by the endpoint
# class InputData(BaseModel):
#     question: str


# @app.post("/api")
# # CHANGE IS HERE: Use Form(...) for the question instead of Body(...)
# async def analyze_data(question: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
#     """
#     This endpoint performs data analysis. It can:
#     1. Scrape a URL mentioned in the 'question' text.
#     2. Analyze data from a file uploaded via the 'file' field.

#     The agent will first try to use the file data if provided.
#     """
#     final_input = question

#     # If a file is uploaded, read its content and prepend it to the question.
#     if file:
#         try:
#             file_contents = await file.read()
            
#             # CHANGE IS HERE: Check if the file is a .txt file
#             if file.filename and file.filename.lower().endswith('.txt'):
#                 # If it's a text file, its content becomes the question
#                 final_input = file_contents.decode('utf-8')
#                 print(f"Using content of '{file.filename}' as the question.")
#             else:
#                 # For other files (like CSV), prepend content to the form question
#                 file_text = file_contents.decode('utf-8')
#                 final_input = (
#                     f"Here is the content of the file named '{file.filename}':\n"
#                     f"---START OF FILE---\n"
#                     f"{file_text}\n"
#                     f"---END OF FILE---\n\n"
#                     f"Based on the file content above, please answer the following question: {question}"
#                 )
#                 print(f"Content from '{file.filename}' has been added to the prompt.")

#         except Exception as e:
#             return {"error": f"Failed to read or decode the file: {e}"}

#     # Invoke the agent with the combined input
#     try:
#         # The new workflow expects a state dictionary as input. We create it here.
#         initial_state = {
#             "messages": [HumanMessage(content=final_input)] ,
#             "plan": "",
#             "critic_feedback": "",
#             "execution_result": "",
#             "revision_number": 0,
#             "original_request":final_input
#         }
        
#         # We call our imported workflow. It will run the Worker/Critic loop.
#         final_state = await analysis_workflow.ainvoke(initial_state)
        
#         # The final result is now in the 'execution_result' key of the final state.
#         output_str = final_state.get("execution_result", "")

#         # --- NO CHANGES HERE ---
#         # The logic for parsing the final JSON output remains the same.
#         json_match = re.search(r'\[.*\]', output_str, re.DOTALL)
#         if json_match:
#             json_str = json_match.group(0)
#             try:
#                 response_json = json.loads(json_str)
#                 return response_json
#             except json.JSONDecodeError:
#                 return {"error": "Agent returned a string that looks like JSON, but failed to parse.", "raw_output": output_str}
#         else:
#             return {"output": output_str}

#     except Exception as e:
#         # Adding more detailed error logging for debugging
#         import traceback
#         print(traceback.format_exc())
#         return {"error": f"An error occurred during agent execution: {str(e)}"}


# # ============================================================================================

# @app.post("/api")
# async def analyze_route(
#     question_file: UploadFile = File(...),
#     attachment: Optional[UploadFile] = File(None)
# ):
    
    
#     """
#     This endpoint handles file uploads for the agent.
#     It saves the files, gets their paths, and invokes the agent workflow.
#     """
#     # Create a secure, temporary directory that will be automatically cleaned up
#     with tempfile.TemporaryDirectory() as temp_dir:
        
#         # --- 1. Save all uploaded files and collect their paths ---
#         file_paths = []
        
#         # A list of all files uploaded in the request
#         all_files = [question_file]
#         if attachment:
#             all_files.append(attachment)

#         for up_file in all_files:
#             # Create the full path for the file inside the temporary directory
#             temp_file_path = os.path.join(temp_dir, up_file.filename)
            
#             # Save the file to that path
#             with open(temp_file_path, "wb") as buffer:
#                 shutil.copyfileobj(up_file.file, buffer)
            
#             # Add the server-side path to our list
#             file_paths.append(temp_file_path)
#             print(f"Saved file to temporary path: {temp_file_path}")

#         # --- 2. Read the question from the saved question.txt file ---
#         question_path = next(
#             (p for p in file_paths if re.search(r'question', p, re.IGNORECASE)),
#             None
#         )
#         if not question_path:
#             return {"error": "A file containing 'question' in its name is required."}
        
#         with open(question_path, "r") as f:
#             question_text = f.read()

#         # --- 3. Invoke the agent with the correct initial state ---
#         print(f"Invoking agent with files: {file_paths}")
        
#         initial_state = {
#              "original_question": question_text,
#             "question": question_text,
#             "file_paths": file_paths,
#             "messages": [],             # Start with an empty list of messages
#             "revision_number": 0,       # Initialize the revision number to 0
#             "plan": "",
#             "critic_feedback": "",
#             "execution_result": "",
#             "execution_error": "",
#             "user_script_to_run": "",
#             "image_b64_data": [],
#             "dataframe_for_analysis": None,
#         }
        
#         try:
#             # We call our imported workflow with the correct state.
#             final_state = await agent_app.ainvoke(initial_state)
            
#             # The final result is now in the 'execution_result' key.
#             output_str = final_state.get("final_answer", "No result found.")
            
#             return {"output": output_str}

#         except Exception as e:
#             import traceback
#             print(traceback.format_exc())
#             return {"error": f"An error occurred during agent execution: {str(e)}"}

# # ============================================================================================



@app.post("/api")
async def analyze_route(request: Request):
    print("New Version")
    """
    This endpoint is designed to handle the specific multipart request format
    sent by the unchangeable run.py script.
    """

    # 1. Parse the form data
    form_data = await request.form()

    question_file = form_data.get("questions.txt")
    attachment_file = next((v for k, v in form_data.items() if k != "questions.txt"), None)

    # 2. Detect files flexibly
    for field_name, uploaded_item in form_data.items():
        if isinstance(uploaded_item, UploadFile):
            fname = uploaded_item.filename.lower()
            field = field_name.lower()

            # Pick question file if "question" appears in filename or field name
            if "question" in fname or "question" in field:
                question_file = uploaded_item
            else:
                # Everything else is treated as attachment (first one wins)
                if not attachment_file:
                    attachment_file = uploaded_item

    # 3. Validate
    if not question_file:
        raise HTTPException(
            status_code=422,
            detail="No file with 'question' in its name/field was found."
        )
    if not attachment_file:
        raise HTTPException(
            status_code=422,
            detail="No data attachment file was found."
        )

    # 4. Save files to a secure temp dir
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        for up_file in [question_file, attachment_file]:
            temp_file_path = (up_file.filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(up_file.file, buffer)
            file_paths.append(temp_file_path)
            print(f"Saved file to temporary path: {temp_file_path}")

        # 5. Read question text
        question_path = next(
            (p for p in file_paths if "question" in os.path.basename(p).lower()),
            None
        )
        with open(question_path, "r") as f:
            question_text = f.read()

        # 6. Invoke your agent
        initial_state = {
            "original_question": question_text,
            "question": question_text,
            "file_paths": file_paths,
            "messages": [],
            "revision_number": 0,
            "plan": "",
            "critic_feedback": "",
            "execution_result": "",
            "execution_error": "",
            "user_script_to_run": "",
            "image_b64_data": [],
            "dataframe_for_analysis": None,
        }

        try:
            final_state = await agent_app.ainvoke(initial_state)
            
            # The final result is now in the 'execution_result' key.
            output_str = final_state.get("final_answer", "No result found.")


            # match = re.search(r"\{.*\}", output_str, re.DOTALL)
            # if match:
            #     dict_str = match.group(0)

            #     # 2. Convert string dict -> real Python dict
            #     data = ast.literal_eval(dict_str)

            #     # 3. Now you can use it like a dictionary
            #     print(type(data))       # <class 'dict'>
            #     print(data.keys())      # dict_keys([...])
            if len(output_str)>6000:
                output_str=output_str.replace("'",'"')
                res=json.loads(output_str)
            else:
                res=json.loads(output_str)


            print (f"---------------------------------------------------\n{type(res)}\n---------------------------------------------------")
            print (f"---------------------------------------------------\n{res}\n---------------------------------------------------")

            return res



        
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {"error": f"An error occurred during agent execution: {str(e)}"}

@app.api_route("/debug", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def debug_request_inspector(request: Request):
    """
    This endpoint captures and prints the full details of any incoming HTTP request.
    It's a powerful tool for debugging client-side issues.
    """
    print("=========================================================")
    print("=                INCOMING REQUEST DETAILS               =")
    print("=========================================================")

    # 1. Basic Request Info
    print(f"\n[INFO] Method: {request.method}")
    print(f"[INFO] URL: {request.url}")
    print(f"[INFO] Client IP: {request.client.host}")

    # 2. Headers
    # Headers are a dictionary-like object. We can iterate through them.
    print("\n[HEADERS]")
    headers_dict = dict(request.headers)
    if headers_dict:
        for key, value in headers_dict.items():
            print(f"  {key}: {value}")
    else:
        print("  (No headers received)")

    # 3. Query Parameters
    # These are the parameters in the URL after the '?' (e.g., /path?id=123)
    print("\n[QUERY PARAMETERS]")
    query_params_dict = dict(request.query_params)
    if query_params_dict:
        for key, value in query_params_dict.items():
            print(f"  {key}: {value}")
    else:
        print("  (No query parameters)")

    # 4. Request Body
    # This is the most critical part for debugging POST requests.
    print("\n[BODY CONTENT]")
    try:
        # We first read the raw bytes of the body.
        body_bytes = await request.body()
        
        if not body_bytes:
            print("  (Body is empty)")
        else:
            # Try to interpret the body in different formats
            
            # Attempt 1: As JSON
            try:
                json_body = await request.json()
                print("  [Body interpreted as JSON]:")
                import json
                print(json.dumps(json_body, indent=2))
            
            # Attempt 2: As Form Data (for file uploads)
            except Exception:
                try:
                    form_data = await request.form()
                    print("  [Body interpreted as Form Data]:")
                    if form_data:
                        for key, value in form_data.items():
                            if hasattr(value, 'filename'): # Check if it's an UploadFile
                                print(f"  - Field '{key}': File '{value.filename}' ({value.content_type})")
                            else:
                                print(f"  - Field '{key}': '{value}'")
                    else:
                        print("    (Form data is empty)")

                # Attempt 3: As Raw Text
                except Exception:
                    print("  [Body interpreted as Raw Text]:")
                    print(f"    {body_bytes.decode('utf-8', errors='ignore')}")

    except Exception as e:
        print(f"  [ERROR] Could not read or parse request body: {e}")

    print("\n=========================================================")
    
    # Return a simple confirmation response
    return {"message": "Request received and details have been logged to the server console."}

