import os
import sys

from dotenv import load_dotenv
from chunker import ThresholdSemanticChunker, SmartPDFProcessor
from retrievel import build_advanced_retriever
from chain import create_hr_rag_chain
from logger import logging
from exception import CustomException

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)

logging.info("pipeline.py module loaded successfully")


class HRPipelineOrchestrator:

    def __init__(self, pdf_path="pdf/hr_policy.pdf", llm_model="qwen/qwen3-32b"):
        self.pdf_path = pdf_path
        self.llm_model = llm_model
        self.retriever = None
        self.rag_chain = None
        logging.info(f"HRPipelineOrchestrator initialized | PDF: '{pdf_path}' | Model: '{llm_model}'")

    def initialize_pipeline(self):
        try:
            logging.info("Pipeline initialization started...")

            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"Document not found at: {self.pdf_path}")

            # Step 1: Chunking
            logging.info("Creating ThresholdSemanticChunker...")
            chunker = ThresholdSemanticChunker(threshold=0.6)
            logging.info("Creating SmartPDFProcessor...")
            processor = SmartPDFProcessor(semantic_chunker=chunker)
            logging.info(f"Processing PDF: '{self.pdf_path}'...")
            smart_chunks = processor.process_pdf(self.pdf_path)
            logging.info(f"PDF processed successfully. Total chunks: {len(smart_chunks)}")

            # Step 2: Retriever
            logging.info("Building advanced retriever...")
            self.retriever = build_advanced_retriever(smart_chunks)
            logging.info("Advanced retriever built successfully.")

            # Step 3: RAG Chain
            logging.info(f"Creating HR RAG chain with model: '{self.llm_model}'...")
            self.rag_chain = create_hr_rag_chain(self.retriever, self.llm_model)
            logging.info("HR RAG chain created successfully.")

            logging.info("Pipeline initialization completed successfully.")
            return True

        except Exception as e:
            logging.error("Pipeline initialization failed.")
            raise CustomException(e, sys)

    def ask_question(self, question: str):
        try:
            logging.info(f"Question received: '{question}'")

            if not self.rag_chain:
                raise ValueError("Pipeline not initialized. Call initialize_pipeline() first.")

            logging.info("Invoking RAG chain...")
            response = self.rag_chain.invoke({
                "question": question,
                "chat_history": []  # skipped as requested
            })

            logging.info("Response received from RAG chain successfully.")
            return response

        except Exception as e:
            logging.error(f"Failed to process question: '{question}'")
            raise CustomException(e, sys)


# ─── Testing ─────────────────────────────────────────────────────────────────
#if __name__ == "__main__":
#    try:
#        logging.info("Starting pipeline test...")

        # Step 1: Orchestrator banao
#        pipeline = HRPipelineOrchestrator(
#            pdf_path="E:/HR_chatbot/src/hr_policy.pdf",  # ← apna path daalo
#           llm_model="qwen/qwen3-32b"
#        )

        # Step 2: Pipeline initialize karo
#        pipeline.initialize_pipeline()

        # Step 3: Test query
#        test_question = "What is the leave policy?"
#       logging.info(f"Running test query: '{test_question}'")
#        response = pipeline.ask_question(test_question)

#        logging.info("Pipeline test completed successfully.")
#        print("\n--- HR Assistant Response ---")
#        print(response)

#    except Exception as e:
#        logging.error(f"Pipeline test failed: {e}")
#        raise CustomException(e, sys)