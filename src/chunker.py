import os 
import sys 

from dotenv import load_dotenv


from src.exception import CustomException
from src.logger import logging

logging.info("chunker.py module loaded successfully")

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class ThresholdSemanticChunker:
    """Chunks text based on cosine similarity between consecutive sentences."""

    def __init__(self, model_name = "all-MiniLM-L6-v2", threshold= 0.6):
        try:
            logging.info(f"Initializing ThresholdSemanticChunker with model: '{model_name}' and threshold: {threshold}")
            self.model = SentenceTransformer(model_name)
            self.threshold = threshold
            logging.info("ThresholdSemanticChunker embedding model loaded successfully.")
            
        except Exception as e:
            logging.error("Failed to initialize ThresholdSemanticChunker model.")
            raise CustomException(e, sys)

    def split(self, text: str) -> list[str]:
        try:
            logging.info("Executing split method in ThresholdSemanticChunker.")
            
            # Splitting text by newline (\n) as optimized for paragraph structure
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
            if not sentences:
                logging.info("No text content found to split. Returning empty chunk list.")
                return []

            logging.info(f"Generating semantic embeddings for {len(sentences)} lines/sentences.")
            embeddings = self.model.encode(sentences)
            chunks, current_chunk = [], [sentences[0]]

            for i in range(1, len(sentences)):
                sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
                if sim >= self.threshold:
                    current_chunk.append(sentences[i])
                else:
                    chunks.append(". ".join(current_chunk) + ".")
                    current_chunk = [sentences[i]]

            if current_chunk:
                chunks.append(". ".join(current_chunk) + ".")

            logging.info(f"Successfully divided text into {len(chunks)} structural chunks.")
            return chunks

        except Exception as e:
            logging.error("Error occurred while calculating semantic similarity or chunking text.")
            raise CustomException(e, sys)


class SmartPDFProcessor:
    """Loads a PDF and produces semantically chunked LangChain Documents."""

    def __init__(self, semantic_chunker: ThresholdSemanticChunker):
        self.chunker = semantic_chunker

    def _clean_text(self, text: str) -> str:
        text = " ".join(text.split())
        text = text.replace("f!", "fi")
        return text

    def process_pdf(self, pdf_path: str) -> list[Document]:
        logging.info("Entered process_pdf method inside SmartPDFProcessor class")
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Target PDF file does not exist at path: {pdf_path}")

            logging.info(f"Invoking PyPDFLoader for path target: '{pdf_path}'")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            processed_chunks = []
            logging.info(f"Loaded {len(pages)} raw pages from PDF. Beginning clean-and-chunk cycle.")

            for page_num, page in enumerate(pages):
                cleaned = self._clean_text(page.page_content)
                if len(cleaned.strip()) < 50:
                    logging.info(f"Skipping page {page_num + 1}: Character count below noise threshold (<50 chars).")
                    continue

                # Pass clean layout text to our semantic similarity splitter
                page_chunks = self.chunker.split(cleaned)

                for chunk_text in page_chunks:
                    doc = Document(
                        page_content=chunk_text,
                        metadata={
                            **page.metadata,
                            "page": page_num + 1,
                            "total_pages": len(pages),
                            "chunk_method": "semantic",
                            "char_count": len(chunk_text),
                        },
                    )
                    processed_chunks.append(doc)

            # Assign globally scaled collection indexing numbers to metadata tracking
            for i, chunk in enumerate(processed_chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(processed_chunks)

            logging.info(f"Smart PDF Processing run finished successfully. Yielded {len(processed_chunks)} total documents.")
            return processed_chunks

        except Exception as e:
            logging.error(f"Critical error isolated during processing stages of: {pdf_path}")
            raise CustomException(e, sys)
        


   