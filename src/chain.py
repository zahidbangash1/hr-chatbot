import sys
import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from chunker import ThresholdSemanticChunker, SmartPDFProcessor
from retrievel import build_advanced_retriever

from logger import logging
from exception import CustomException

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)

logging.info("chain.py module loaded successfully")

def format_docs(docs):
    try:
        if not docs:
            logging.warning("No documents provided to format_docs. Returning empty string.")
            return ""
        logging.info(f"Formatting {len(docs)} documents for context injection.")
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logging.error("Failed to format documents into string context.")
        raise CustomException(e, sys)

#def format_chat_history(messages):
#   try:
#       logging.info(f"Formatting chat history containing {len(messages)} messages.")
#      history = ""
#     for msg in messages:
#        role = "User" if msg.get("role") == "user" else "HR Assistant"
#       history += f"{role}: {msg.get('content', '')}\n"
#  return history
# except Exception as e:
#    logging.error("Failed to format chat history.")
#   raise CustomException(e, sys)

def create_hr_rag_chain(retriever, model_name="qwen/qwen3-32b"):
    try:
        logging.info(f"Starting initialization of HR RAG Chain using model: '{model_name}'...")

        logging.info("Initializing Groq LLM...")
        llm = ChatGroq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))

        prompt_template = """
       You are a professional HR Assistant.

    IMPORTANT RULES:
- Give short, clear, and direct answers
- Do NOT show thinking process or reasoning
- Do NOT write "think>" or internal steps
- Do NOT repeat context
- Use bullet points if needed
- Keep answer under 6-8 lines



        Context: 
        {context}
        
        Employee Question: {question}
        HR Assistant Answer:
        """
        logging.info("Configuring PromptTemplate...")
        prompt = PromptTemplate.from_template(prompt_template)

        logging.info("Constructing LCEL chain mapping...")
        rag_chain = (
            {
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                ##"chat_history": lambda x: format_chat_history(x["chat_history"]),
                "question": lambda x: x["question"]
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        logging.info("HR RAG Chain successfully built and ready for invocation.")
        return rag_chain

    except Exception as e:
        logging.error("Critical error encountered while building the RAG chain.")
        raise CustomException(e, sys)


# ─── Testing ────────────────────────────────────────────────────────────────
#if __name__ == "__main__":
#    try:
        

#        # Step 1: Chunks banao
#        logging.info("Creating chunker and processor...")
#        chunker = ThresholdSemanticChunker()
#        processor = SmartPDFProcessor(semantic_chunker=chunker)
#        docs = processor.process_pdf(r"E:\HR_chatbot\src\hr_policy.pdf")  # ← apna PDF path daalo
#        logging.info(f"Total chunks: {len(docs)}")

        # Step 2: Retriever banao
#        logging.info("Building retriever...")
#        retriever = build_advanced_retriever(docs)
#        logging.info("Retriever built!")
#
#        # Step 3: Chain banao
#        logging.info("Building HR RAG chain...")
#        chain = create_hr_rag_chain(retriever)
#        logging.info("Chain built!")

        # Step 4: Test query
#        logging.info("Running test query...")
#        response = chain.invoke({
#            "question": "What is the attendance Policy?",
           # "chat_history": []
#        })
#        logging.info(f"Chain response received.")
#        print("\n--- HR Assistant Response ---")
#        print(response)

#    except Exception as e:
#        logging.error(f"Chain test failed: {e}")
#        raise CustomException(e, sys)