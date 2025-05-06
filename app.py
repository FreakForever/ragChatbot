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
    .reportview-container { background-color: #F0F2F6; }
    .sidebar .sidebar-content { background-color: #FFFFFF; color: #333333; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; }
    .stTextInput>div>div>input { border-radius: 10px; border: 1px solid #4CAF50; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

def process_pdfs(uploaded_files: List) -> Optional[FAISS]:
    all_documents = []
    try:
        with st.spinner("Processing PDFs..."):
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                loader = PyPDFLoader(temp_path)
                text_doc = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_overlap=200,
                    chunk_size=1000
                )
                documents = text_splitter.split_documents(text_doc)
                all_documents.extend(documents)

            db = FAISS.from_documents(all_documents, OpenAIEmbeddings())
            st.success("All PDFs processed and converted to embeddings.")
            return db
    except Exception as e:
        st.error(f"Error processing PDFs: {e}")
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
    st.markdown("Upload **one or more** PDFs and ask questions about their content.")

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Choose AI Model", ["gpt-4o-mini", "gpt-3.5-turbo"])
        temperature = st.slider("Model Temperature", 0.0, 1.0, 0.2, 0.1)

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files"
    )

    if uploaded_files:
        vector_db = process_pdfs(uploaded_files)

        if vector_db:
            user_query = st.text_input(
                "Ask a question about the documents",
                placeholder="e.g. Summarize all reports"
            )

            if user_query:
                llm = ChatOpenAI(
                    model=model_choice,
                    temperature=temperature,
                    max_retries=2
                )

                prompt_template = ChatPromptTemplate.from_template("""
                Based strictly on the provided context, answer the question.
                If no relevant information exists, respond: "I cannot find the answer in the documents."

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
                        with st.expander("📄 Retrieved Context"):
                            st.write(context)
                    except Exception as e:
                        st.error(f"Response generation failed: {e}")

if __name__ == "__main__":
    main()
