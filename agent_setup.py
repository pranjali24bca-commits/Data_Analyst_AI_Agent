import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# -------------------------------
# 📂 LOAD ALL CSVs INTO A LIST
# -------------------------------
def load_csv_folder(folder_path: str):
    dfs = []
    filenames = []

    if not os.path.exists(folder_path):
        print(f"⚠️ Folder {folder_path} not found.")
        return []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
            filenames.append(file)

    # We print the mapping so you know which df index corresponds to which file
    print("✅ Loaded tables:")
    for i, name in enumerate(filenames):
        print(f"  df{i+1}: {name}")
        
    return dfs

# -------------------------------
# 🧠 LOAD PROMPT
# -------------------------------
def load_prompt(prompt_path="prompt.txt"):
    if not os.path.exists(prompt_path):
        return "You are a helpful assistant that analyzes data using pandas."
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------
# 🤖 CREATE PANDAS AGENT
# -------------------------------
def get_agent(csv_folder="data", prompt_path="prompt.txt"):
    # 1. Load DataFrames
    dfs = load_csv_folder(csv_folder)
    system_prompt = load_prompt(prompt_path)

    # 2. Initialize Model 
    # Note: Using 'gemini-2.0-flash' as it is the current high-performance stable model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0, # Lower temperature is better for data tasks
    )

    agent = create_pandas_dataframe_agent(
        model,
        dfs,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors="The previous output could not be parsed. You MUST respond using the exact Action/Action Input/Observation/Final Answer format. Do NOT output free-form text outside of Final Answer.",
        max_iterations=8,
    )

    return agent, system_prompt