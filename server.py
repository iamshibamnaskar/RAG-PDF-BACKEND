# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Path
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path as PlPath
import shutil
import os
import uuid
from datetime import datetime

from tasks import process_pdf, generate_final_answer
from celery.result import AsyncResult
from celery_app import celery_app

from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from celery.utils.log import get_task_logger

# Async Mongo driver
import motor.motor_asyncio

app = FastAPI(title="PDF Upload API", version="1.0.0")
logger = get_task_logger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # list of origins (no "*")
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

UPLOAD_DIR = PlPath("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mongo setup - read URI and DB name from env (with sensible defaults)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "pdf_uploader")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "files")

mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
files_collection = mongo_db[MONGO_COLLECTION]


def _serialize_doc(doc: dict) -> dict:
    """
    Convert Mongo document to JSON-serializable dict (convert ObjectId and datetimes).
    """
    if not doc:
        return {}
    doc = dict(doc)  # make a shallow copy
    _id = doc.get("_id")
    if _id is not None:
        doc["_id"] = str(_id)
    for k, v in doc.items():
        if isinstance(v, datetime):
            doc[k] = v.isoformat()
    return doc


@app.get("/")
async def root():
    return {"message": "PDF Upload API is running", "status": "healthy"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must have .pdf extension")

    try:
        file_uuid = str(uuid.uuid4())
        uuid_filename = f"{file_uuid}.pdf"
        file_location = UPLOAD_DIR / uuid_filename

        # Save uploaded file to disk
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file.file.close()

        # Schedule the Celery task to run after 6 seconds
        async_result = process_pdf.apply_async(args=[file_uuid, str(file_location)], countdown=6)

        # Prepare metadata doc to insert into MongoDB
        doc = {
            "filerealname": file.filename,
            "uuid_filename": uuid_filename,
            "file_uuid": file_uuid,
            "task_id": async_result.id,
            "file_size": os.path.getsize(file_location),
            "file_path": str(file_location),
            "status": "scheduled",  # you can update this later from your tasks
            "created_at": datetime.utcnow(),
        }

        insert_result = await files_collection.insert_one(doc)
        doc["_id"] = insert_result.inserted_id

        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF uploaded and processing scheduled (after 6s)",
                "original_filename": file.filename,
                "uuid_filename": uuid_filename,
                "file_id": file_uuid,
                "file_size": os.path.getsize(file_location),
                "file_path": str(file_location),
                "task_id": async_result.id,
                "db_id": str(insert_result.inserted_id),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Check task status and result by Celery task_id
    """
    async_result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": async_result.status}

    if async_result.status == "PENDING":
        response["info"] = "Task is pending (not yet started or unknown to the broker)."
    elif async_result.status == "STARTED":
        response["info"] = "Task started."
    elif async_result.status == "PROCESSING":
        # custom state from task.update_state
        response["info"] = async_result.info
    elif async_result.status == "SUCCESS":
        response["result"] = async_result.result
    elif async_result.status in ("FAILURE", "REVOKED"):
        try:
            response["error"] = str(async_result.result)
        except Exception:
            response["error"] = "Unknown error"

    return JSONResponse(status_code=200, content=response)


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@app.post("/search/{collection_id}")
async def search_collection_post(
    collection_id: str = Path(..., description="Chroma collection id"),
    payload: SearchRequest = Body(...),
):
    """
    POST /search/{collection_id}
    Body: { "query": "your query", "k": 5 }
    """
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large:335m")
    persist_directory = os.getenv("DB_DIR", "db/chroma_db")

    try:
        root = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        available = [c.name for c in root._client.list_collections()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")

    if collection_id not in available:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_id}' not found. Available: {available}")

    try:
        db = Chroma(
            collection_name=collection_id,
            persist_directory=persist_directory,
            embedding_function=embedding_model,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load collection '{collection_id}': {e}")

    try:
        retriver = db.as_retriever(search_kwargs={"k": payload.k})
        chunks = retriver.invoke(payload.query)

        answer = generate_final_answer(chunks, payload.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {e}")

    return JSONResponse(
        status_code=200,
        content={
            "collection_id": collection_id,
            "query": payload.query,
            "k": payload.k,
            "result": answer,
        },
    )


@app.get("/files")
async def list_files(limit: int = 100):
    """
    Return metadata for uploaded files.
    Optional query param: ?limit=50
    """
    try:
        cursor = files_collection.find().sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        serialized = [_serialize_doc(d) for d in docs]
        return JSONResponse(status_code=200, content={"count": len(serialized), "files": serialized})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch files from DB: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
