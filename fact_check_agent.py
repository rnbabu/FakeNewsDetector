import streamlit as st
import time
import os
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("Please set TAVILY_API_KEY in .env file")
 

st.markdown("""
    <style>
        button[title="Stop"] {
            display: none;
        }
        button[title="Deploy"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)
st.set_page_config(page_title="🕵️ Fake News Detector", layout="centered", page_icon="🔍")
 
# ===================== SIDEBAR =====================
with st.sidebar:
    st.title("⚡ Settings")
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama3.2:3b"
    
    model_name = st.selectbox(
        "Model (smaller = faster)",
        ["llama3.2:3b", "gemma2:2b", "phi3:3.8b", "llama3.1:8b"],
        index=0,
        key="model_selector",
        help="Choose the model for analysis. Smaller models are faster but may be less accurate."
    )
    
    if model_name != st.session_state.selected_model:
        st.session_state.selected_model = model_name
        if "agent" in st.session_state:
            del st.session_state.agent
        st.rerun()
    
    max_results = st.slider("Search Results", 3, 8, 4, help="Number of search results returned from Tavily for fact checking. Lesser the number faster the response")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
 
# ===================== MAIN UI =====================
st.set_page_config(page_title="Fake News Detector", page_icon="")
st.title("🕵️ Fake News Detector")
st.caption(f"Current Model: **{st.session_state.selected_model}**")
 
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Paste a news claim below."}]
 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
# ===================== LANGGRAPH AGENT =====================
@st.cache_resource(show_spinner=False)
def get_agent(model_name: str):
    llm = ChatOllama(
        model=model_name,
        temperature=0.2,
        max_tokens=800,
        base_url="http://localhost:11434"
    )
    
    search_tool = TavilySearch(
        max_results=max_results,
        search_depth="basic",
        include_answer=True
    )
    
    system_prompt = """
You are a quick, impartial fake news detector.
Use the Tavily search tool to gather evidence.
Be objective and evidence-based.
Always format your final response EXACTLY like this:
Verdict: REAL / FAKE / MISLEADING / UNVERIFIED
Explanation:
- concise reasoning
Sources:
- source links
"""
 
    # Modern LangGraph create_react_agent (no state_modifier)
    agent = create_react_agent(
        model=llm,
        tools=[search_tool],
        prompt=system_prompt   # Use 'prompt' instead of state_modifier
    )
    
    return agent
 
# Recreate agent when model changes
if "agent" not in st.session_state or st.session_state.get("last_model") != st.session_state.selected_model:
    st.session_state.agent = get_agent(st.session_state.selected_model)
    st.session_state.last_model = st.session_state.selected_model
 
agent = st.session_state.agent
 
# ===================== CHAT INPUT =====================
if user_input := st.chat_input("Paste news claim here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        start_time = time.time()

        with st.spinner(f"Analyzing with {st.session_state.selected_model}..."):
            try:
                response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

                # Safe extraction from LangGraph response
                messages = response.get("messages", [])
                if messages:
                    final_msg = messages[-1]
                    answer = getattr(final_msg, "content", str(final_msg))
                else:
                    answer = str(response)

                # ===================== FORMAT VERDICT =====================
                verdict_styles = {
                    "REAL": ("✅", "green"),
                    "FAKE": ("❌", "red"),
                    "MISLEADING": ("⚠️", "orange"),
                    "UNVERIFIED": ("❓", "gray")
                }

                formatted_answer = answer

                if "Verdict:" in answer:
                    verdict = answer.split("Verdict:")[1].split("\n")[0].strip().upper()
                    icon, color = verdict_styles.get(verdict, ("ℹ️", "blue"))

                    formatted_answer = formatted_answer.replace(
                        f"Verdict: {verdict}",
                        f"<b>Verdict:</b> <b style='color:{color}; font-size:20px'>{icon} {verdict}</b>"
                    )

                formatted_answer = formatted_answer.replace(
                    "Explanation:",
                    "<br><br><b>Explanation:</b>"
                )

                formatted_answer = formatted_answer.replace(
                    "Sources:",
                    "<br><br><b>Sources:</b>"
                )

                st.markdown(formatted_answer, unsafe_allow_html=True)

                st.caption(f"⏱️ Took {time.time() - start_time:.1f} seconds")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted_answer
                })

            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
