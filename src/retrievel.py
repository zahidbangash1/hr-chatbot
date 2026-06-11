import os
import sys
import logging

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder 


from src.chunker import ThresholdSemanticChunker, SmartPDFProcessor
from src.logger import logging
from src.exception import CustomException

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)

logging.info("retriever.py module loaded successfully")

def build_advanced_retriever(documents, persist_directory="./chroma_db", collection_name="hr_semantic_collection"):
    """Builds a hybrid retriever with cross-encoder reranking."""
    try:
        logging.info(f"Starting advanced retriever build process with {len(documents)} documents...")

        if not documents:
            raise ValueError("The documents list is empty. Cannot build a retriever without processed documents.")

        # 1. Embeddings & Vectorstore
        logging.info("Initializing HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        logging.info(f"Creating Chroma vectorstore at '{persist_directory}' with collection '{collection_name}'...")
        vectorstore = Chroma.from_documents(
            documents=documents,       
            embedding=embeddings,        
            persist_directory=persist_directory,
            collection_name=collection_name
        )

        # 2. Hybrid Retrieval Setup
        logging.info("Setting up BM25 Sparse Retriever...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 3  

        logging.info("Setting up Dense MMR Retriever...")
        dense_retriever_mmr = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
        )  

        logging.info("Combining into Ensemble Retriever (Weights: BM25=0.3, Dense=0.7)...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever_mmr],
            weights=[0.3, 0.7] 
        )  

        # 3. Cross-Encoder Reranking
        logging.info("Initializing Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)...")
        cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
            
        logging.info("Wrapping with ContextualCompressionRetriever...")
        advanced_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        
        logging.info("Advanced hybrid retriever successfully built.")
        return advanced_retriever

    except Exception as e:
        logging.error("Failed to build the advanced retriever.")
        raise CustomException(e, sys)
    



