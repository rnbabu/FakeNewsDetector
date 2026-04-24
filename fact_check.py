import streamlit as st
import time
import os
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

# ===================== LOAD ENV =====================
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    st.error("Please set TAVILY_API_KEY in your .env file")
    st.stop()

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Model",
        ["llama3.2:3b", "gemma2:2b", "phi3:3.8b", "llama3.1:8b"],
        index=0
    )

    max_results = st.slider("Search Results", 3, 8, 4)

# ===================== INIT TOOLS =====================
llm = ChatOllama(
    model=model_name,
    temperature=0.2,
)

search_tool = TavilySearch(
    max_results=max_results,
    topic="general"
)

# ===================== CHAT HISTORY =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ===================== FORMAT RESPONSE =====================
def format_response(response_text):
    verdict_icons = {
        "REAL": "✅",
        "FAKE": "❌",
        "MISLEADING": "⚠️",
        "UNVERIFIED": "❓"
    }

    formatted = response_text

    for verdict, icon in verdict_icons.items():
        formatted = formatted.replace(
            f"**Verdict:** {verdict}",
            f"**Verdict:** {icon} {verdict}"
        )

    return formatted

# ===================== CHAT INPUT =====================
if user_input := st.chat_input("Paste news claim here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        start_time = time.time()

        with st.spinner(f"Analyzing with {model_name}..."):
            try:
                # Step 1: Search evidence
                search_results = search_tool.invoke(user_input)

                evidence_text = ""
                for item in search_results.get("results", []):
                    evidence_text += f"- {item.get('content', '')}\nSource: {item.get('url', '')}\n\n"

                # Step 2: Build prompt
                prompt = f"""
You are an impartial fake news detector.

Analyze the following claim using the evidence provided.

Claim:
{user_input}

Evidence:
{evidence_text}

Return your answer in this exact format:

**Verdict:** REAL / FAKE / MISLEADING / UNVERIFIED
**Explanation:** Brief reasoning with key facts
**Sources:** List important sources
"""

                # Step 3: LLM analysis
                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)

                # Step 4: Format output
                formatted_answer = format_response(answer)

                st.markdown(formatted_answer, unsafe_allow_html=True)
                st.caption(f"⏱️ Took {time.time() - start_time:.1f} seconds")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted_answer
                })

            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
