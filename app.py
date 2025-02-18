import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import bs4

# Load environment variables

OPENAI_API_KEY = "<YOUR_OPENAI_API_KEY_HERE>"

def initialize_session_state():
    """Initialize session state variables"""
    session_state_vars = {
        "messages": [],
        "db": None,
        "chain": None
    }
    
    for var, value in session_state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = value

def load_document(uploaded_file):
    """Load document based on file type"""
    # Get the file extension
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    # Create a temporary file to store the uploaded content
    with open("temp_file" + file_extension, "wb") as f:
        f.write(uploaded_file.getvalue())
        file_path = f.name
        
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
        
    documents = loader.load()
    
    # Clean up the temporary file
    os.remove(file_path)
    
    return documents

def process_documents(documents):
    """Process documents through text splitter and create vector store"""
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma.from_documents(splits, embeddings)
    
    return db

def create_conversation_chain(db):
    """Create a conversation chain"""
    llm = ChatOpenAI( temperature=0, openai_api_key=OPENAI_API_KEY, model_name = "gpt-4o-mini")
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return chain

def main():
    # Initialize session state
    initialize_session_state()
    
    # Set up the Streamlit page
    st.set_page_config(page_title="Document Chat Assistant", page_icon="📚")
    st.title("Document Chat Assistant 📚")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
    
    if uploaded_file and not st.session_state.db:
        with st.spinner("Processing document..."):
            # Load and process the document
            documents = load_document(uploaded_file)
            st.session_state.db = process_documents(documents)
            st.session_state.chain = create_conversation_chain(st.session_state.db)
            st.success("Document processed successfully!")
    
    # Chat interface
    if st.session_state.db:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document:"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get response from chain
                    result = st.session_state.chain(
                        {"question": prompt, 
                         "chat_history": [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]}
                    )
                    response = result["answer"]
                    
                    st.markdown(response)
                    
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Instructions if no document is uploaded
    else:
        st.info("Please upload a document to start chatting!")

if __name__ == "__main__":
    main() 