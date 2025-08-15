# PDF Chatbot with OCR

This project is a Streamlit-based chatbot that allows users to upload PDF documents and interact with their content through natural language queries.  
It uses LangChain, Mistral AI, and vector search embeddings to retrieve relevant document excerpts, applying OCR when necessary for scanned PDFs.

## Features

- PDF Upload – Upload any PDF file via the web interface.
- Intelligent Document Indexing – Automatically splits and indexes document content for efficient semantic search.
- Context-Aware Q&A – Answers are generated based on the document content using Mistral AI.
- Vector Store Search – Uses embeddings for semantic similarity search across document chunks.
- Chat History – Maintains a conversation context while interacting with the document.
- OCR Support – Handles scanned PDFs through PyPDFLoader with OCR capabilities (if required).

## Tech Stack

- Frontend: [Streamlit](https://streamlit.io/)  
- Backend: [LangChain](https://www.langchain.com/)  
- LLM Provider: [Mistral AI](https://mistral.ai/)  
- Embeddings: `mistral-embed` model  
- Vector Store: `InMemoryVectorStore`  
- PDF Processing: `PyPDFLoader` with chunking via `RecursiveCharacterTextSplitter`  
- Environment Management: `python-dotenv`  

## Project Structure

```
.
├── app.py                  # Streamlit UI for chat interaction
├── chatbot_documentos.py   # Core logic for loading, indexing, and answering questions
├── .env                    # API keys and credentials
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables  
Create a `.env` file in the project root:
```env
LANGSMITH_API_KEY=your_langsmith_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

1. Upload a PDF file.  
2. Wait for the document to be indexed.  
3. Type a question about the document content.  
4. Receive AI-generated answers based on semantic search and context reasoning.

## How It Works

1. PDF Loading – Uses `PyPDFLoader` to extract text from PDFs.  
2. Chunking – Splits text into overlapping chunks (`chunk_size=2500`, `chunk_overlap=500`) to preserve context.  
3. Embedding & Indexing – Generates embeddings using `MistralAIEmbeddings` and stores them in an in-memory vector database.  
4. Query Handling – On user query, retrieves the most relevant chunks (`k=12`) and sends them to the LLM.  
5. Response Generation – The LLM (`mistral-large-latest`) analyzes the context and produces a concise answer.  

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
