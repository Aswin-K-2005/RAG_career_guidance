import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama


LLM_MODEL = "gemma3:latest" 
EMBEDDING_MODEL = "nomic-embed-text"
PDF_FILE_NAME = "skillsense - pdf.pdf"
CHROMA_PATH = "./chroma_db"

# --- RAG Core Functions ---

@st.cache_resource(show_spinner=f"Connecting to Ollama and checking {LLM_MODEL}...")
def check_ollama_status():
    """Checks if the Ollama server and the specified LLM are available."""
    try:
        # Test connectivity by initializing the model
        llm = Ollama(model=LLM_MODEL, base_url="http://localhost:11434")
        # Use a short timeout to quickly check connectivity
        llm.invoke("Hi", config={'timeout': 5}) 
        return True
    except Exception:
        return False

def format_docs(docs):
    """Formats the retrieved documents into a single string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource(show_spinner="Loading, chunking, and embedding PDF...")
def initialize_rag_components(pdf_file_name):
    """Loads document, splits it, creates embeddings, and returns the retriever."""
    try:
        # 1. Load Documents using PyPDFLoader
        loader = PyPDFLoader(pdf_file_name)
        docs = loader.load()
        if not docs:
            st.error(f"Failed to load content from {pdf_file_name}.")
            return None

        # 2. Split into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # 3. Create Embeddings and Vector Store (ChromaDB)
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")
        
        # Create a persistent vector store from the document splits
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embedding_model, 
            persist_directory=CHROMA_PATH
        )
        
        # 4. Create Retriever to fetch top 3 relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return retriever

    except Exception as e:
        st.error(f"Error during RAG initialization: {e}")
        st.info("Check console for details. Ensure 'pypdf' is installed and models are pulled via Ollama.")
        return None

def setup_rag_chain(retriever):
    """Sets up and returns the Gemma RAG chain using LangChain Expression Language (LCEL)."""
    
    # 1. Initialize Gemma LLM
    llm = ChatOllama(model=LLM_MODEL, base_url="http://localhost:11434", temperature=0)

    # 2. Define the RAG Prompt Template
    template = """You are an expert Q&A assistant. Use the following retrieved context to answer the user's question.
If the context does not contain the answer, you can respond politely using general knowledge.
Keep the answer concise and helpful.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate.from_template(template)

    # 3. Define the RAG chain
    rag_chain = (
        # Retrieve context first
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        # Pass context and question to the prompt template
        | prompt
        # Send augmented prompt to Gemma LLM
        | llm
        # Parse the output to a string
        | StrOutputParser()
    )
    return rag_chain

# --- Streamlit UI ---

def main():
    # Page configuration must be the very first Streamlit call
    st.set_page_config(
        page_title="Career Navigator Bot",
        page_icon="🧭", # Compass/Guide icon
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧭 Career Navigator Pro")
    st.caption("Your AI-Powered Career Guide")
    
    # --- Custom CSS for a professional Dark Theme chat look ---
    def load_custom_css():
        st.markdown("""
            <style>
            /* General Streamlit app adjustments for a modern feel */
            section.main {
                padding-top: 2rem; 
            }
            
            /* --- Dark Theme specific overrides --- */
            
            /* Bot (Assistant) Message Bubble: Use secondary background color for bot bubble (left side) */
            .stChatMessage:not(:has(> [data-testid="chat-message-container"] > [data-testid="stMarkdownContainer"] > p:first-child:empty)) {
                background-color: #161b22; /* A shade lighter than the main background for contrast */
                border-radius: 12px 12px 12px 0; 
                margin-right: 20%;
                padding: 10px 15px;
                border: 1px solid #30363d; /* Subtle dark border */
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
            }

            /* User Message Bubble: Use primary color for user bubble (right side) */
            .stChatMessage:has(> [data-testid="chat-message-container"] > [data-testid="stMarkdownContainer"] > p:first-child:empty) {
                background-color: #0077b6; /* Primary blue accent color */
                color: white; /* Ensure text is white on colored background */
                border-radius: 12px 12px 0 12px; 
                margin-left: 20%;
                padding: 10px 15px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            }
            
            /* Fix markdown color inside user bubble (if needed, to ensure white text) */
            .stChatMessage:has(> [data-testid="chat-message-container"] > [data-testid="stMarkdownContainer"] > p:first-child:empty) * {
                color: white !important;
            }

            /* Custom Avatars for the Career Guide Bot (with blue border) */
            .stChatMessage div[data-testid="stVerticalBlock"] > div:first-child img {
                border-radius: 50%;
                width: 30px; 
                height: 30px;
                object-fit: cover;
                border: 2px solid #00b4d8; /* Brighter accent blue border */
            }

            /* Style the chat input box */
            .stChatInput > div > div > input {
                border-radius: 20px;
                padding: 10px 15px;
            }

            /* Hide the default Streamlit footer */
            footer {visibility: hidden;}
            
            </style>
            """, unsafe_allow_html=True)

    load_custom_css() 
    
    # --- Step 1: Pre-run checks ---
    if not os.path.exists(PDF_FILE_NAME):
        st.error(f"File not found: {PDF_FILE_NAME}. Please place it in the same directory as this script.")
        return
        
    if not check_ollama_status():
        st.error(f"Ollama Server not running or model '{LLM_MODEL}' not found.")
        st.markdown("Please verify `ollama serve` is running and you have run `ollama pull gemma:7b`.")
        return

    # --- Step 2: Initialize RAG ---
    # The empty containers you saw may disappear once this step completes successfully.
    retriever = initialize_rag_components(PDF_FILE_NAME)
    if retriever is None:
        return 

    # --- Step 3: Setup Chain ---
    rag_chain = setup_rag_chain(retriever)

    # --- Step 4: Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add an initial welcome message
        st.session_state.messages.append({"role": "assistant", "content": f"Hello! I am ready to answer questions based on the content of **{PDF_FILE_NAME}**."})

    # Display existing messages
    for message in st.session_state.messages:
        avatar = "🧭" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your PDF..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Get and display streaming response
        with st.chat_message("assistant", avatar="🧭"):
            with st.spinner(f"Generating response with {LLM_MODEL}..."):
                try:
                    # Use invoke for non-streaming response
                    full_response = rag_chain.invoke(prompt)
                    st.markdown(full_response)
                except Exception as e:
                    full_response = f"Error generating response: {e}"
                    st.error(full_response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
