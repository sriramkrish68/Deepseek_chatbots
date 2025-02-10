import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
import networkx as nx
import matplotlib.pyplot as plt

# Custom CSS styling
st.markdown("""
<style>
    .main { background-color: #1a1a1a; color: #ffffff; }
    .sidebar .sidebar-content { background-color: #2d2d2d; }
    .stTextInput textarea { color: #ffffff !important; }
    .stSelectbox div[data-baseweb="select"] { color: white !important; background-color: #3d3d3d !important; }
    .stSelectbox svg { fill: white !important; }
    .stSelectbox option { background-color: #2d2d2d !important; color: white !important; }
    div[role="listbox"] div { background-color: #2d2d2d !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Thought Process Visualizer")
st.caption("🚀 Real-Time Reasoning Breakdown with DeepSeek R1")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:8b"], index=0)
    compare_gpt4 = st.checkbox("Compare with GPT-4")
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initialize the chat engine
llm_engine = ChatOllama(model=selected_model, base_url="http://localhost:11434", temperature=0.3)

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI reasoning assistant. Break down your logical steps clearly before arriving at an answer."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! Ask me a reasoning-based question! 🤖"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your question here...")

def parse_thought_process(response_text):
    steps = response_text.split("\n")
    reasoning_steps = [step for step in steps if step.strip()]
    return reasoning_steps

def visualize_thought_process(steps):
    G = nx.DiGraph()
    for i, step in enumerate(steps):
        G.add_node(i, label=step)
        if i > 0:
            G.add_edge(i - 1, i)
    
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color="lightblue", edge_color="gray", font_size=10)
    plt.show()
    st.pyplot(plt)

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    with st.spinner("🧠 Analyzing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    reasoning_steps = parse_thought_process(ai_response)
    
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    st.subheader("🧐 Thought Process Breakdown")
    visualize_thought_process(reasoning_steps)
    
    st.rerun()
