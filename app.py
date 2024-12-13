import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def setup_page_config():
    st.set_page_config(
        page_title="PDF RAG Chatbot",
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .reportview-container {
        background-color: #F0F2F6;
    }
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #4CAF50;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def process_pdf(uploaded_file) -> Optional[FAISS]:

    try:
        with st.spinner("Processing PDF..."):
            with open("temp_uploaded.pdf", "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader("temp_uploaded.pdf")
            text_doc = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_overlap=200, 
                chunk_size=1000
            )
            documents = text_splitter.split_documents(text_doc)

            db = FAISS.from_documents(documents, OpenAIEmbeddings())
            
            st.success("PDF processed successfully! Text converted to embeddings.")
            return db
    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def retrieve_context(retriever, query: str, top_k: int = 3) -> str:
    try:
        retriever_result = retriever.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in retriever_result[:top_k]])
        return context
    except Exception as e:
        st.error(f"Context retrieval error: {e}")
        return ""

def create_rag_chain(llm, prompt_template):
    return (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

def main():
    setup_page_config()
    
    st.title("🤖 PDF RAG Chatbot")
    st.markdown("Upload a PDF and ask questions about its content.")
    

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox(
            "Choose AI Model", 
            ["gpt-4o-mini", "gpt-3.5-turbo"]
        )
        temperature = st.slider(
            "Model Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            step=0.1
        )
    uploaded_file = st.file_uploader(
        "Upload PDF", 
        type="pdf", 
        help="Upload a PDF file to start chatting"
    )

    if uploaded_file:
        vector_db = process_pdf(uploaded_file)
        
        if vector_db:
            user_query = st.text_input(
                "Ask a question about the document", 
                placeholder="What insights can you provide?"
            )
            
            if user_query:
                llm = ChatOpenAI(
                    model=model_choice, 
                    temperature=temperature, 
                    max_retries=2
                )
                
                prompt_template = ChatPromptTemplate.from_template("""
                Based strictly on the provided context, answer the question.
                If no relevant information exists, respond: "I cannot find the answer in the document."

                Context: {context}
                Question: {question}
                """)

                retriever = vector_db.as_retriever(search_kwargs={"k": 3})
                context = retrieve_context(retriever, user_query)
                rag_chain = create_rag_chain(llm, prompt_template)

                with st.spinner("Generating response..."):
                    try:
                        result = rag_chain.invoke({
                            "context": context, 
                            "question": user_query
                        })
                        st.markdown("### 🧠 AI Response:")
                        st.markdown(f"> {result}", unsafe_allow_html=True)
                        with st.expander("Retrieved Context"):
                            st.write(context)
                    
                    except Exception as e:
                        st.error(f"Response generation failed: {e}")

if __name__ == "__main__":
    main()