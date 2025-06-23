# Hate Speech Detection System

A comprehensive GenAI-powered hate speech detection system built with FastAPI, Streamlit, and advanced ML techniques. The system classifies text content, retrieves relevant policies using hybrid RAG, provides detailed reasoning, and recommends appropriate moderation actions.

## 🚀 Features

### Core Functionality
- **Multi-Class Text Classification**: Classifies content as Hate, Toxic, Offensive, Neutral, or Ambiguous using DIAL API integration
- **Hybrid RAG Policy Retrieval**: Combines semantic similarity and keyword matching to fetch relevant policy documents from Qdrant vector database
- **Intelligent Reasoning**: Provides detailed explanations for classification decisions using retrieved policy context
- **Action Recommendation**: Suggests appropriate moderation actions (Escalate, Warn, Review, Allow) based on classification and confidence scores
- **Audio Input Support**: Speech-to-text functionality using Whisper for voice input processing

### Technical Architecture
- **Microservices Design**: Separate FastAPI backend and Streamlit frontend
- **Vector Database**: Qdrant for semantic policy document storage and retrieval
- **Agent-Based System**: Modular agents for classification, retrieval, reasoning, and recommendations
- **Error Handling**: Comprehensive error management with graceful fallbacks
- **Docker Support**: Full containerization with Docker Compose orchestration

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   FastAPI API   │────│   Qdrant Vector│
│   (Port 8501)   │    │   (Port 8000)   │    │   DB (6333)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   Orchestrator  │              │
         │              └─────────────────┘              │
         │                       │                       │
    ┌─────────┐        ┌─────────────────┐    ┌─────────────────┐
    │ Audio   │        │     Agents      │    │   Policy Docs   │
    │ Input   │        │  - Classifier   │    │   (.txt files)  │
    │(Whisper)│        │  - Retriever    │    │                 │
    └─────────┘        │  - Reasoner     │    └─────────────────┘
                       │  - Recommender  │
                       └─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.11+
- Docker and Docker Compose (optional)
- DIAL API key for LLM services

### Quick Start

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd hate-speech-detection
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your DIAL_API_KEY
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize policy database**
```bash
python -m app.services.policy_loader
```

5. **Run with Docker Compose (Recommended)**
```bash
docker-compose up --build
```

**Services will be available at:**
- Streamlit UI: http://localhost:8501
- FastAPI API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard

### Manual Setup

**Start Qdrant (if not using Docker)**
```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

**Run FastAPI backend**
```bash
uvicorn api.main:app --reload --port 8000
```

**Run Streamlit frontend**
```bash
cd ui
streamlit run app.py --server.port 8501
```

## 📁 Project Structure

```
.
├── api/                          # FastAPI backend
│   ├── main.py                   # API endpoints and middleware
│   └── routes/                   # API route handlers
├── ui/                           # Streamlit frontend
│   ├── app.py                    # Main Streamlit application
│   └── components/               # UI components
├── app/                          # Core application logic
│   ├── agents/                   # Agent implementations
│   │   ├── base.py               # Base agent class
│   │   ├── classification_agent.py  # Text classification
│   │   ├── retriever.py          # Hybrid RAG retrieval
│   │   ├── reasoner.py           # Policy reasoning
│   │   ├── recommender.py        # Action recommendations
│   │   └── error_handler.py      # Error management
│   ├── models/                   # Pydantic schemas
│   │   └── schemas.py            # Data models
│   ├── services/                 # External services
│   │   ├── llm_services.py       # DIAL API integration
│   │   ├── embed_service.py      # Embedding generation
│   │   ├── qdrant_client.py      # Vector database client
│   │   └── policy_loader.py      # Policy document processing
│   └── utils/                    # Utilities
│       └── exceptions.py         # Custom exceptions
├── data/
│   └── policy_docs/              # Policy documents (.txt)
│       ├── reddit_policy.txt
│       ├── meta_community_standards.txt
│       ├── indian_legal_framework.txt
│       ├── youtube_community_guidelines.txt
│       └── google_prohibited_content.txt
├── docker/                       # Docker configuration
│   ├── Dockerfile                # Application container
│   └── entrypoint.sh             # Container startup script
├── tests/                        # Test suite
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Python dependencies
└── README.md
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Required
DIAL_API_KEY=your_dial_api_key_here

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=hate_speech_policies

# Application Settings
LOG_LEVEL=INFO
MAX_INPUT_LENGTH=5000
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.6
```

### Policy Documents

Place your policy documents as `.txt` files in `data/policy_docs/`. The system includes support for:

- **Reddit Community Policy** - Community guidelines and hate speech rules
- **Meta Community Standards** - Facebook/Instagram content policies  
- **Indian Legal Framework** - IPC sections and IT Act provisions
- **YouTube Community Guidelines** - Video platform content policies
- **Google Prohibited Content** - Search and ads content restrictions

Each policy document should contain clear guidelines about hate speech, harassment, and content moderation rules.

## 🔌 API Usage

### Classification Endpoint

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text content here",
    "include_audio": false
  }'
```

### Response Format

```json
{
  "hate_speech": {
    "classification": "Toxic",
    "confidence": "HIGH",
    "reason": "Contains aggressive language targeting individuals"
  },
  "policies": [
    {
      "source": "Reddit",
      "summary": "Reddit Content Policy: Prohibits harassment and hate speech...",
      "relevance_score": 89.5
    }
  ],
  "reasoning": "The content violates multiple platform policies due to targeted harassment language...",
  "action": {
    "action": "WARN",
    "severity": "MEDIUM",
    "reasoning": "Toxic content warrants user warning and behavior monitoring."
  }
}
```

## 🧪 Testing

Run the complete test suite:

```bash
# Run all tests with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/api/          # API endpoint tests
```

**Coverage Requirements**: >80% test coverage maintained

## 🏃‍♂️ Development

### Code Quality Standards

```bash
# Format code
black .

# Check linting
flake8

# Type checking
mypy app/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Agent System

The system uses a modular agent architecture:

1. **ClassificationAgent**: Handles text classification using DIAL LLM services
2. **HybridRetriever**: Combines semantic search with keyword matching for policy retrieval
3. **PolicyReasoner**: Generates explanations using LLM and retrieved policy context
4. **ActionRecommender**: Maps classifications to moderation actions with confidence-based logic
5. **ErrorHandlerAgent**: Provides graceful error handling and user feedback

### Adding New Policy Documents

1. Add your `.txt` file to `data/policy_docs/`
2. Update the provider mapping in `policy_loader.py`
3. Reinitialize the vector database:
   ```bash
   python -m app.services.policy_loader
   ```

## 🚀 Deployment

### Docker Production Deployment

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up --scale hate_speech_app=3
```

### Key Dependencies

- **FastAPI 0.115.12** - Modern web framework for APIs
- **Streamlit 1.45.1** - Interactive web applications
- **Qdrant Client 1.14.2** - Vector database operations
- **Sentence Transformers 4.1.0** - Text embeddings
- **OpenAI Whisper 20240930** - Speech recognition
- **LangChain 0.3.25** - LLM application framework
- **DIAL Integration** - Custom LLM service integration

## 🔍 Monitoring & Observability

- **Health Checks**: Built-in health endpoints for all services
- **Logging**: Structured logging with configurable levels
- **Metrics**: Performance metrics and classification statistics
- **Error Tracking**: Comprehensive error reporting and alerting



