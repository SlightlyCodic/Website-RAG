from dotenv import load_dotenv
import streamlit as st
import os
import requests
from bs4 import BeautifulSoup

os.environ["OPENAI_API_KEY"] = st.secrets["API_KEY"]

# Delete the file at app start if it exists
if os.path.exists("website_text.txt"):
    try:
        os.remove("website_text.txt")
    except Exception as e:
        st.warning(f"Could not remove old website_text.txt: {e}")

# --- Parsing function ---
def parse_website(url, output_file="website_text.txt"):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.MissingSchema:
        return "Invalid URL. Please include http:// or https://."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again or check your connection."
    except requests.exceptions.RequestException as e:
        return f"Error fetching the website: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        elements = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
            if tag.find_parent(['footer', 'nav']):
                continue
            text = tag.get_text(strip=True)
            if text:
                elements.append(text)
        text = '\n'.join(elements)
        if not text.strip():
            return "No usable text found on the website."
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(text)
    except Exception as e:
        return f"Error parsing or saving website data: {e}"
    return None

# --- RAG chatbot setup ---
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain

def get_qa_chain():
    try:
        loader = TextLoader("website_text.txt", encoding="utf-8")
        docs = loader.load()
        if not docs:
            st.error("No data loaded from website_text.txt.")
            return None
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        if not splits:
            st.error("No text chunks created from website data.")
            return None
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(
            model="meta-llama/llama-3.3-8b-instruct:free",
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
        system_prompt = (
            "You are a helpful assistant. You will be provided with website parsed data. "
            "Answer user questions using only the information from the website data. "
            "If the answer is not present in the data, say you don't know."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            verbose=False,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error setting up the chatbot: {e}")
        return None

# --- Streamlit App ---
st.set_page_config(page_title="Website RAG Chatbot", page_icon="💬")
st.title("💬 Website RAG Chatbot")

# Session state for chat and website
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "website_url" not in st.session_state:
    st.session_state.website_url = ""
if "parsed" not in st.session_state:
    st.session_state.parsed = False

def clear_all():
    st.session_state.chat_history = []
    st.session_state.qa_chain = None
    st.session_state.website_url = ""
    st.session_state.parsed = False
    if os.path.exists("website_text.txt"):
        try:
            os.remove("website_text.txt")
        except Exception as e:
            st.warning(f"Could not remove website_text.txt: {e}")

# Sidebar for website input and controls
with st.sidebar:
    st.header("Setup")
    website_url = st.text_input("Enter website URL", value=st.session_state.website_url)
    if st.button("Parse Website"):
        if website_url:
            error = parse_website(website_url)
            if error:
                st.error(error)
                st.session_state.parsed = False
            else:
                qa_chain = get_qa_chain()
                if qa_chain is not None:
                    st.session_state.website_url = website_url
                    st.session_state.qa_chain = qa_chain
                    st.session_state.parsed = True
                    st.session_state.chat_history = []
                    st.success("Website parsed and chatbot ready!")
                else:
                    st.session_state.parsed = False
        else:
            st.warning("Please enter a website URL.")
    if st.button("Clear & Restart"):
        clear_all()
        st.rerun()

# Add custom CSS for chat message boxes and scrollable chat window
st.markdown(
    """
    <style>
    .chat-window {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 8px;
        margin-bottom: 1em;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        background: #f7f7f7;
    }
    .chat-message {
        border: 1.5px solid #e0e0e0;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        padding: 0.7em 1em;
        margin-bottom: 0.7em;
        background: #fafbfc;
    }
    .chat-message.user {
        background: #e3f2fd;
        border-color: #90caf9;
    }
    .chat-message.bot {
        background: #f1f8e9;
        border-color: #aed581;
    }
    .stTextInput>div>div>input {
        font-size: 1.1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main chat interface (render only once)
if st.session_state.parsed and st.session_state.qa_chain:
    st.subheader(f"Chatting with: {st.session_state.website_url}")
    # Scrollable chat window
    chat_html = '<div class="chat-window">'
    for user, bot in st.session_state.chat_history:
        chat_html += f'<div class="chat-message user"><b>You:</b> {user}</div>'
        chat_html += f'<div class="chat-message bot"><b>Bot:</b> {bot}</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input box stays at the bottom
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question:", key="user_input")
        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            try:
                result = st.session_state.qa_chain.invoke({"question": user_input, "chat_history": []})
                st.session_state.chat_history.append((user_input, result["answer"]))
                st.rerun()
            except Exception as e:
                st.error(f"Error getting response from chatbot: {e}")
else:
    st.info("Enter a website URL in the sidebar and click 'Parse Website' to start chatting.")
