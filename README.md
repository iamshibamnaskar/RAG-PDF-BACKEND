# RAG Backend

## 🚀 Try it out

[https://rag.shibamnaskar.in/](https://rag.shibamnaskar.in/)

## 🛠️ Installation

### Install system dependencies (Linux)
```bash
sudo apt-get install poppler-utils tesseract-ocr libmagic-dev
```

### Setup Python environment
```bash
# Clone the repository
git clone <your-repository-url>
cd <repository-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Create .env file
Create a `.env` file in the root directory with:
```
GOOGLE_API_KEY=<Google ai studios api key>
DB_DIR=db/chroma_db
MONGO_URI=<mongodb-url>
MONGO_DB=pdf_uploader
MONGO_COLLECTION=files
```

### Install Redis
Redis is required for this application.

## 🚀 Running the application

### Start Celery worker
```bash
celery -A celery_app worker --loglevel=info
```

### Start the server
```bash
python server.py
```
