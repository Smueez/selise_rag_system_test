# RAG Agent System with Self-Reflection

An intelligent Retrieval-Augmented Generation (RAG) system built with FastAPI, Azure OpenAI, and Qdrant vector database. Features agentic behavior with tool calling and self-reflection for accurate, grounded responses.

## 🎯 Overview

This system allows you to:
- Upload PDF documents and automatically process them
- Query the documents using natural language
- Get accurate, context-grounded answers with citations
- Stream responses in real-time
- Self-reflection mechanism to minimize hallucinations

## 🏗️ Architecture
```
User Query → FastAPI API → Reflective Agent
                              ↓
                         Tool Router
                         ├── Semantic Search (Qdrant)
                         ├── Multi-Query Search
                         ├── Exact Match Search
                         └── Answer Validator
                              ↓
                         Self-Reflection
                              ↓
                         Streaming Response
```

## 📋 Prerequisites

- Python 3.12+
- Docker & Docker Compose (for Qdrant)
- Azure OpenAI API access with:
  - GPT-4o-mini deployment (for chat)
  - text-embedding-ada-002 deployment (for embeddings)

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Create project directory
mkdir rag-agent-project
cd rag-agent-project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file:
```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure Embedding Configuration
AZURE_EMBEDDING_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_EMBEDDING_API_KEY=your_api_key_here
AZURE_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
AZURE_EMBEDDING_API_VERSION=2024-02-15-preview

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=document_embeddings

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# Agent Configuration
MAX_ITERATIONS=3
ENABLE_SELF_REFLECTION=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### 4. Start Qdrant Vector Database
```bash
docker-compose up -d
```

Verify Qdrant is running:
```bash
curl http://localhost:6333/
```

### 5. Start the API Server
```bash
uvicorn app.main:app --reload
```

The API will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

## 📖 Usage

### 1. Upload a PDF Document

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

**Response:**
```json
{
  "message": "File uploaded successfully. Processing started in background.",
  "filename": "document.pdf",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

### 2. Check Processing Status
```bash
curl http://localhost:8000/api/v1/job/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "filename": "document.pdf",
  "progress": "Processing complete!",
  "chunks_processed": 150,
  "total_chunks": 150
}
```

### 3. Query the Document (Non-Streaming)
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main topics covered in this document?",
    "conversation_history": []
  }'
```

**Response:**
```json
{
  "answer": "Based on the document, the main topics covered are...",
  "context": "Retrieved context from the document...",
  "iterations": 2,
  "tool_calls": 1,
  "success": true
}
```

### 4. Query with Streaming (Real-time Response)
```bash
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the key concepts discussed",
    "conversation_history": []
  }'
```

**Streaming Events:**
```
data: {"type": "tool_execution", "tool": "semantic_search", "status": "completed"}
data: {"type": "reflection", "status": "started"}
data: {"type": "reflection", "result": {...}}
data: {"type": "answer_start"}
data: {"type": "answer_chunk", "content": "The key concepts "}
data: {"type": "answer_chunk", "content": "discussed in the "}
data: {"type": "answer_chunk", "content": "document are..."}
data: {"type": "answer_end"}
data: {"type": "metadata", "iterations": 2, "tool_calls": 1}
data: {"type": "done"}
```

### 5. Using Python
```python
import requests

# Upload document
files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/api/v1/upload', files=files)
job_id = response.json()['job_id']
print(f"Job ID: {job_id}")

# Query
query_response = requests.post(
    'http://localhost:8000/api/v1/query',
    json={
        "query": "What is this document about?",
        "conversation_history": []
    }
)
print(query_response.json()['answer'])

# Streaming query
stream_response = requests.post(
    'http://localhost:8000/api/v1/query/stream',
    json={"query": "Summarize the key points"},
    stream=True
)

for line in stream_response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/job/{job_id}` | Check processing status |
| `POST` | `/api/v1/query` | Query without streaming |
| `POST` | `/api/v1/query/stream` | Query with SSE streaming |
| `GET` | `/api/v1/health` | Health check |
## 🧪 Testing

Run the test script:
```bash
python test_agent.py
```

Or use the interactive API documentation:
```
http://localhost:8000/docs
```

## 🔍 How It Works

### Document Processing Pipeline

1. **Upload**: User uploads a PDF from command line [Usage: python process_n_store.py <path_to_pdf>]
2. **Extraction**: Text is extracted from PDF pages
3. **Chunking**: Document is split into overlapping chunks (token-aware)
4. **Embedding**: Each chunk is embedded using Azure OpenAI
5. **Storage**: Embeddings are stored in Qdrant vector database

### Query Processing with Agent

1. **Query Reception**: User submits a question
2. **Tool Selection**: Agent decides which tools to use
3. **Retrieval**: Semantic search finds relevant chunks
4. **Answer Generation**: LLM generates answer from context
5. **Self-Reflection**: Agent validates answer quality
6. **Refinement**: If needed, agent refines the answer
7. **Streaming**: Response is streamed to user in real-time

### Self-Reflection Mechanism

The agent evaluates its own answers:
- ✅ **Grounding Check**: Is the answer based on retrieved context?
- ✅ **Completeness Check**: Does it fully answer the question?
- ✅ **Accuracy Check**: Are there any contradictions?

If issues are found, the agent:
- Retrieves more information
- Reformulates the answer
- Validates again (up to MAX_ITERATIONS)

## ⚙️ Configuration

### Chunking Settings
```env
CHUNK_SIZE=1000           # Target chunk size in characters
CHUNK_OVERLAP=200         # Overlap between chunks
```

**Why overlap?** Ensures context isn't lost at chunk boundaries.

### Retrieval Settings
```env
TOP_K_RESULTS=5           # Number of chunks to retrieve
SIMILARITY_THRESHOLD=0.7  # Minimum similarity score (0-1)
```

**Higher threshold** = More precise but might miss relevant info
**Lower threshold** = More comprehensive but might include noise

### Agent Settings
```env
MAX_ITERATIONS=3              # Max refinement loops
ENABLE_SELF_REFLECTION=true   # Enable/disable reflection
```

## 🛠️ Troubleshooting

### Qdrant not starting
```bash
# Check if port 6333 is in use
docker compose -f docker-compose-infra.yml
```

### Token limit errors
- Reduce `CHUNK_SIZE` in `.env`
- Current max: ~6000 tokens per chunk

### Slow embeddings
- Reduce batch size in `embedding_service.py`
- Check Azure OpenAI quota limits

### Agent not answering
- Check logs: `tail -f logs/app.log`
- Verify documents are uploaded: `GET /api/v1/health`
- Try disabling reflection: `ENABLE_SELF_REFLECTION=false`

Check system health:
```bash
curl http://localhost:8000/api/v1/health
```

View logs:
```bash
tail -f logs/app.log
```

Check Qdrant dashboard:
```
http://localhost:6333/dashboard
```
