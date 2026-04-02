# Data Agent – Natural Language Analytics on CSVs

This project is a simple but powerful data assistant that lets you ask questions in plain English and get answers directly from your CSV files.

It uses Google Gemini with LangChain and a FAISS-based retrieval layer so the model actually understands your data structure before answering.

---

## What this project does

- Ask questions like: "Which artist generated the most revenue?"
- Works across multiple related CSV files (albums, artists, invoices, etc.)
- Uses a RAG layer to fetch schema + important values before answering
- Shows intermediate reasoning steps in the UI
- Returns both explanation + data (not just raw tables)

---

## Tech used

- Streamlit (UI)
- Google Gemini
- LangChain (Pandas Agent)
- FAISS (vector search)
- Pandas

---

## Project structure

app.py – Streamlit app  
agent_setup.py – LLM + agent config  
use_rag.py – FAISS retrieval logic  
prompt.txt – rules + schema context  
summarize_data.py – generates schema summary  
run_agent.py – CLI testing  
Generate_Vectorestore.ipynb – builds vector DB  

---

## Setup

### 1. Install dependencies

pip install streamlit pandas numpy faiss-cpu langchain langchain-google-genai langchain-experimental google-genai python-dotenv

---

### 2. Add API key

Create a .env file:

GOOGLE_API_KEY=your_key_here

---

## How to run

### Step 1 – Prepare data

python summarize_data.py

This creates a schema summary file.

---

### Step 2 – Build vector store

Run the notebook:

Generate_Vectorestore.ipynb

This creates the FAISS index used for retrieval.

---

### Step 3 – Start the app

streamlit run app.py

Open: http://localhost:8501

---

## Notes

- The agent can execute pandas code → only use trusted data
- Output is limited to 50 rows to keep things readable
- Always returns explanation along with results

---

## Example questions

- Top 5 customers by revenue
- Which genre has most tracks?
- Show invoices from USA
- Revenue by artist

---

## Author

Pranjali Waradkar
