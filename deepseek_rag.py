
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import requests
from bs4 import BeautifulSoup
import time
import tempfile

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #1a1a1a; color: #ffffff; }
    .sidebar .sidebar-content { background-color: #2d2d2d; }
    .stTextInput textarea, .stChatInput input { color: #ffffff !important; }
    .stSelectbox div[data-baseweb="select"], div[role="listbox"] div {
        color: white !important; background-color: #3d3d3d !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📘 QueryGenius")
st.caption("🚀 Your Intelligent Document & Web Assistant")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:8b"], index=1)
    data_source = st.radio("Select Data Source", ["Upload PDF", "Scrape Website"])
    uploaded_pdf = None
    website_url = ""
    if data_source == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload a PDF Document", type=["pdf"])
    else:
        website_url = st.text_input("Enter Website URL")
    
    # Sidebar additional info
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("- 📄 Document Analysis\n- 🌐 Web Scraping\n- 🤖 AI-Powered Assistance")
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initialize models dynamically
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

def get_language_model():
    return OllamaLLM(model=selected_model)

PROMPT_TEMPLATE = """
You are an expert assistant. Use the provided context to answer the query.
If unsure, say you don't know. Be concise and factual.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        return "\n".join([p.get_text() for p in soup.find_all("p")])
    return "Failed to retrieve content."

def chunk_documents(text):
    text_processor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_processor.create_documents([text])

def index_documents(chunks):
    DOCUMENT_VECTOR_DB.add_documents(chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | get_language_model()
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# Process document or website data
if uploaded_pdf:
    file_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(file_path)
    processed_text = "\n\n".join(doc.page_content for doc in raw_docs)
    processed_chunks = chunk_documents(processed_text)
    index_documents(processed_chunks)
    st.success("✅ PDF Processed Successfully!")
elif website_url:
    scraped_text = scrape_website(website_url)
    if scraped_text != "Failed to retrieve content.":
        processed_chunks = chunk_documents(scraped_text)
        index_documents(processed_chunks)
        st.success("✅ Website Scraped Successfully!")

# Chat Interface
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! How can I assist you today?"}]

chat_container = st.container()

with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_input = st.chat_input("Type your question here...")

def animated_typing_effect(response):
    display_text = ""
    placeholder = st.empty()
    for char in response:
        display_text += char
        time.sleep(0.02)
        placeholder.markdown(display_text)

def generate_ai_response(user_query):
    relevant_docs = find_related_documents(user_query)
    return generate_answer(user_query, relevant_docs)

if user_input:
    st.session_state.message_log.append({"role": "user", "content": user_input})
    with st.spinner("Analyzing..."):
        ai_response = generate_ai_response(user_input)
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    animated_typing_effect(ai_response)
    st.rerun()
