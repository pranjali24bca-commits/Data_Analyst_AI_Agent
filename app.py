import io
import re
import os
import pandas as pd
import streamlit as st
from agent_setup import get_agent
from use_rag import get_similar_texts

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Data Agent",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Dark industrial theme */
.stApp {
    background: #0d0d0f;
    color: #e8e6e1;
}

/* Header */
.main-header {
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #2a2a2e;
    margin-bottom: 1.5rem;
}
.main-header h1 {
    font-size: 1.4rem;
    font-weight: 600;
    color: #f0ede8;
    letter-spacing: 0.02em;
    margin: 0;
}
.main-header p {
    font-size: 0.78rem;
    color: #555;
    margin: 0.3rem 0 0;
    font-family: 'JetBrains Mono', monospace;
}

/* Chat messages */
.chat-user {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 12px 12px 2px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.92rem;
    color: #e8e6e1;
    max-width: 75%;
    margin-left: auto;
    text-align: right;
}

.chat-assistant {
    background: #111113;
    border: 1px solid #222226;
    border-radius: 2px 12px 12px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.92rem;
    color: #e8e6e1;
    max-width: 85%;
}

/* Steps container */
.steps-box {
    background: #0a0a0c;
    border: 1px solid #1e3a2f;
    border-left: 3px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #4ade80;
}
.steps-box .step-label {
    color: #16a34a;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.steps-box pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    color: #86efac;
}

/* Final answer box */
.answer-box {
    background: #0f1a0f;
    border: 1px solid #1a3a1a;
    border-left: 3px solid #a3e635;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0 0.2rem;
    font-size: 0.93rem;
    color: #d9f99d;
    line-height: 1.6;
}
.answer-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #65a30d;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}

/* Input area */
.stChatInput > div {
    background: #18181b !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 10px !important;
}
.stChatInput textarea {
    color: #e8e6e1 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
}

/* Spinner */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #1a2a1a;
    border: 1px solid #2d4a2d;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #4ade80;
    margin: 0.5rem 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0f; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

/* Loaded tables expander */
.stExpander {
    background: #111113 !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
}

/* Plain text answer */
.answer-text {
    font-size: 0.93rem;
    color: #d9f99d;
    line-height: 1.6;
    margin-top: 0.3rem;
}

/* Dataframe styling override */
[data-testid="stDataFrame"] {
    margin-top: 0.5rem;
    border-radius: 8px;
    overflow: hidden;
}
iframe[title="st_dataframe"] {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load agent ONCE using st.cache_resource
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading agent & data...")
def load_agent():
    agent, system_prompt = get_agent()
    return agent, system_prompt


# ─────────────────────────────────────────────
# Keyword extractor for RAG
# ─────────────────────────────────────────────
def extract_keywords(text):
    stopwords = {
        "what", "is", "the", "of", "a", "an", "in", "on", "for", "to",
        "and", "with", "who", "which", "give", "me", "info", "about",
        "tell", "show", "how", "many", "much", "did", "does", "do",
        "are", "was", "were", "has", "have", "had", "can", "could",
        "would", "should", "will", "get", "find", "list"
    }
    return [w for w in text.lower().split() if w not in stopwords]


# ─────────────────────────────────────────────
# Run agent and capture intermediate steps
# ─────────────────────────────────────────────
def run_agent_with_steps(agent, system_prompt, question):
    keywords = extract_keywords(question)

    # RAG context retrieval
    context_hits = []
    for keyword in keywords:
        try:
            results = get_similar_texts(keyword, 1)
            if results:
                context_hits.append(results)
        except Exception:
            pass

    flat_context = "\n".join([str(c) for c in context_hits])

    agent_input = {
        "input": (
            f"REMEMBER THIS : {system_prompt}\n\n"
            f"### RAG CONTEXT:\n{flat_context}\n\n"
            f"### USER QUESTION:\n{question}"
        )
    }

    try:
        result = agent.invoke({**agent_input, "return_intermediate_steps": True})
        steps = result.get("intermediate_steps", [])
        final_answer = result.get("output", "")
        return steps, final_answer, keywords, flat_context

    except Exception as e:
        err_str = str(e)

        # ── Recover from LLM output parsing errors ──────────────────────
        # LangChain raises: "Could not parse LLM output: `<actual answer>`"
        # Extract the real answer from inside the backticks.
        parse_match = re.search(
            r"Could not parse LLM output:\s*`(.*)`",
            err_str,
            re.DOTALL,
        )
        if parse_match:
            recovered = parse_match.group(1).strip()
            # Try to get intermediate steps from the exception if available
            steps = getattr(e, "intermediate_steps", [])
            return steps, recovered, keywords, flat_context

        # Unknown error — surface it cleanly
        return [], f"❌ Error: {err_str}", keywords, flat_context


# ─────────────────────────────────────────────
# Parse tabular text → DataFrame
# ─────────────────────────────────────────────
def try_parse_dataframe(text: str):
    """
    Try to parse a table from the final answer text into a real DataFrame.
    Handles two formats:
      1. Markdown pipe tables  (| col | col | ...)
      2. Whitespace-aligned pandas string tables
    Returns a DataFrame if successful, else None.
    """
    text = text.strip()
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return None

    # ── Format 1: Markdown pipe table ──────────────────────────────────
    pipe_lines = [l for l in lines if l.strip().startswith("|")]
    if len(pipe_lines) >= 2:
        try:
            # Filter out separator rows like |---|---|
            data_lines = [l for l in pipe_lines if not re.match(r"^\s*\|[\s\-|:]+\|\s*$", l)]
            if len(data_lines) < 2:
                return None

            def parse_row(line):
                return [c.strip() for c in line.strip().strip("|").split("|")]

            headers = parse_row(data_lines[0])
            rows = [parse_row(l) for l in data_lines[1:]]
            rows = [r for r in rows if len(r) == len(headers)]
            if not rows:
                return None

            df = pd.DataFrame(rows, columns=headers)

            # Try to cast numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

            return df
        except Exception:
            pass

    # ── Format 2: Whitespace-aligned pandas string table ───────────────
    spaced = sum(1 for l in lines if re.search(r'\s{2,}', l))
    if spaced / len(lines) >= 0.5:
        try:
            df = pd.read_csv(io.StringIO(text), sep=r'\s{2,}', engine='python', index_col=0)
            if df.shape[1] >= 1 and df.shape[0] >= 1:
                df.index.name = None
                return df
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────
# Render final answer: table or text
# ─────────────────────────────────────────────
def render_answer(text: str, df_override=None):
    """
    Render the final answer.
    - If a real DataFrame was passed (df_override), show any prose above it, then the table.
    - If no df_override, try to parse a table from the text; show prose before the table.
    - If no table at all, show plain styled text.
    """
    st.markdown('<div class="answer-label">✦ Final Answer</div>', unsafe_allow_html=True)

    df = df_override if df_override is not None else None

    # Split the text into prose and table parts
    prose = text
    table_text = None

    if df is None:
        # Try to find and extract a table block from the text
        lines = text.splitlines()
        pipe_start = next((i for i, l in enumerate(lines) if l.strip().startswith("|")), None)
        ws_start = None
        if pipe_start is None:
            # Check for whitespace table
            for i, l in enumerate(lines):
                if re.search(r'\s{2,}', l) and not l.strip().startswith("#"):
                    ws_start = i
                    break

        table_start = pipe_start if pipe_start is not None else ws_start
        if table_start is not None:
            prose = "\n".join(lines[:table_start]).strip()
            table_text = "\n".join(lines[table_start:]).strip()
            df = try_parse_dataframe(table_text)

    # Render prose summary (if any)
    if prose:
        st.markdown(f'<div class="answer-text">{prose}</div>', unsafe_allow_html=True)

    # Render table (if any)
    if df is not None:
        st.dataframe(df, use_container_width=True, hide_index=False)
    elif not prose:
        # Fallback: just show the raw text
        st.markdown(f'<div class="answer-text">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 Data Agent</h1>
    <p>pandas · gemini · rag · langchain</p>
</div>
""", unsafe_allow_html=True)

# Load agent (cached)
agent, system_prompt = load_agent()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">💬 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        # History: final answer only (steps were temporary)
        with st.container():
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            render_answer(msg["content"], df_override=msg.get("last_df"))
            st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask anything about your data..."):
    st.markdown(f'<div class="chat-user">💬 {prompt}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Placeholder for live steps — erased after answer is ready
    live_steps_placeholder = st.empty()
    steps, final_answer, last_df = [], "", None

    with st.spinner("🧠 Thinking..."):
        steps, final_answer, keywords, context = run_agent_with_steps(
            agent, system_prompt, prompt
        )

        # Show steps live while spinner is active
        with live_steps_placeholder.container():
            for i, (action, observation) in enumerate(steps, 1):
                tool = getattr(action, "tool", "python_repl")
                code = getattr(action, "tool_input", str(action))
                st.markdown(f"""
                <div class="steps-box">
                    <div class="step-label">Step {i} · {tool}</div>
                    <pre>{code}</pre>
                </div>
                """, unsafe_allow_html=True)
                if observation:
                    st.markdown(f"""
                    <div class="steps-box" style="border-left-color:#3b82f6; color:#93c5fd;">
                        <div class="step-label" style="color:#1d4ed8;">↳ Output</div>
                        <pre>{str(observation)[:800]}</pre>
                    </div>
                    """, unsafe_allow_html=True)

    # Erase steps — show only final answer
    live_steps_placeholder.empty()

    # Extract real DataFrame from last observation if available
    if steps:
        for _, obs in reversed(steps):
            if isinstance(obs, pd.DataFrame):
                last_df = obs
                break

    with st.container():
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        render_answer(final_answer, df_override=last_df)
        st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "steps": steps,
        "last_df": last_df,
    })