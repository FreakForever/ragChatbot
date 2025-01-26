# PDF RAG Chatbot 🤖📄

## Overview
PDF RAG (Retrieval-Augmented Generation) Chatbot is an interactive Streamlit application that allows users to upload a PDF document and ask questions about its content. The chatbot uses advanced AI language models and embedding techniques to provide context-aware responses.

## Features
- Upload and process PDF documents
- Ask questions about the uploaded document
- Choose between different AI models (GPT-4o-mini, GPT-3.5-turbo)
- Adjust model temperature for creative/precise responses
- View retrieved context for transparency

## Prerequisites
- Python 3.8+
- OpenAI API Key

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Requirements.txt
Here's a sample `requirements.txt` for your project:
```
streamlit
python-dotenv
langchain
langchain-community
langchain-openai
pypdf
faiss-cpu
tiktoken
```

## Running the Application
```bash
streamlit run app.py
```

## How to Use
1. Open the application in your browser
2. Upload a PDF document
3. Select an AI model and adjust temperature (optional)
4. Ask questions about the document

## Configuration Options
- **AI Model**: Choose between GPT-4o-mini and GPT-3.5-turbo
- **Temperature**: Controls the randomness of the model's responses
  - Lower values (0.0-0.2): More focused and deterministic
  - Higher values (0.7-1.0): More creative and varied responses

## Troubleshooting
- Ensure you have a valid OpenAI API key
- Check internet connectivity
- Verify PDF file is not corrupted

## Limitations
- Works best with text-based PDFs
- Maximum context retrieval is limited to 3 most relevant chunks
- Relies on OpenAI's embedding and language models

## Contributing
Contributions are welcome! Please submit pull requests or open issues on the GitHub repository
```

