# tasks.py
import time
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from celery_app import celery_app
from celery.utils.log import get_task_logger
from dotenv import load_dotenv

# unstructured
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# langchain / embeddings (optional - only if you will persist)
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
logger = get_task_logger(__name__)

def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """
        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        # Instruction part
        prompt_text += """
        YOUR TASK:
        Generate a comprehensive, searchable description that covers:

        1. Key facts, numbers, and data points from text and tables
        2. Main topics and concepts discussed  
        3. Questions this content could answer
        4. Visual content analysis (charts, diagrams, patterns in images)
        5. Alternative search terms users might use

        Make it detailed and searchable - prioritize findability over brevity.

        SEARCHABLE DESCRIPTION:
        """

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        logger.exception("AI summary failed: %s", e)
        # Fallback to simple summary
        summary = f"{(text or '')[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary

def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    logger.info("🧠 Processing %d chunks with AI Summaries...", len(chunks))
    
    langchain_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        logger.info("Processing chunk %d/%d", current_chunk, total_chunks)
        
        # Analyze chunk content
        content_data = separate_content_types(chunk)
        
        # Debug info
        logger.debug("Types found: %s", content_data['types'])
        logger.debug("Tables: %d, Images: %d", len(content_data['tables']), len(content_data['images']))
        
        # Create AI-enhanced summary if chunk has tables/images
        if content_data['tables'] or content_data['images']:
            logger.info("Creating AI summary for mixed content (chunk %d/%d)...", current_chunk, total_chunks)
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
                logger.info("AI summary created for chunk %d", current_chunk)
                logger.debug("Enhanced content preview: %s", str(enhanced_content)[:200])
            except Exception as e:
                logger.exception("AI summary failed for chunk %d: %s", current_chunk, e)
                enhanced_content = content_data['text']
        else:
            logger.info("Using raw text for chunk %d (no tables/images)", current_chunk)
            enhanced_content = content_data['text']
        
        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        
        langchain_documents.append(doc)
    
    logger.info("✅ Processed %d chunks", len(langchain_documents))
    return langchain_documents

def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."




@celery_app.task(bind=True)
def process_pdf(self, file_id: str, file_path: str) -> Dict[str, Any]:
    try:
        p = Path(file_path).resolve()
        logger.info("process_pdf called for file_id=%s path=%s", file_id, p)

        if not p.exists():
            msg = f"File not found at path: {p}"
            logger.error(msg)
            self.update_state(state="FAILURE", meta={"error": msg})
            return {"status": "error", "message": msg}

        # Update state
        self.update_state(state="PROCESSING", meta={"file_id": file_id, "stage": "partitioning"})
        logger.info("Partitioning document: %s", p)

        elements = partition_pdf(
            filename=str(p),
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True
        )

        logger.info("Extracted %d elements from %s", len(elements), p)
        self.update_state(state="PROCESSING", meta={"file_id": file_id, "stage": "chunking"})

        chunks = chunk_by_title(elements, max_characters=3000, new_after_n_chars=2400, combine_text_under_n_chars=500)
        logger.info("Created %d chunks", len(chunks))

        self.update_state(state="PROCESSING", meta={"file_id": file_id, "stage": "summarising"})
        processed_docs = summarise_chunks(chunks)


        ## Vectorize documents
        self.update_state(state="PROCESSING", meta={"file_id": file_id, "stage": "Vectorizing"})
        logger.info("------------------------Processing Vector store--------------------------")

        embedding_model = OllamaEmbeddings(model="mxbai-embed-large:335m")
        persist_directory= os.getenv("DB_DIR")

        vectorstore = Chroma.from_documents(
            documents=processed_docs,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space":"cosine"},
            collection_name=file_id
        )

        
        result = {
            "status": "processed",
            "file_id": file_id,
            "file_path": str(p),
            "num_elements": len(elements),
            "num_chunks": len(chunks),
        }
        logger.info("process_pdf finished for file_id=%s", file_id)
        return result

    except Exception as exc:
        logger.exception("Error processing PDF for file_id=%s: %s", file_id, exc)
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise
