import os
from dotenv import load_dotenv
import textwrap
import numpy as np
import pandas as pd
import google.generativeai as genai
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

# Constants
DEFAULT_MODEL = "models/embedding-001"
TASK_TYPE = "retrieval_document"


@dataclass
class Document:
    """Represents a document with its title and content."""

    title: str
    content: str


# Load environment variables
load_dotenv()

# Get API key with error handling
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample documents
DOCUMENTS = [
    {
        "title": "Operating the Climate Control System",
        "content": "Your Googlecar has a climate control system...",
    },
    {
        "title": "Touchscreen",
        "content": "Your Googlecar has a large touchscreen display...",
    },
    {
        "title": "Shifting Gears",
        "content": "Your Googlecar has an automatic transmission...",
    },
]


class EmbeddingService:
    """Service class to handle Google GenerativeAI embeddings."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        """Initialize the embedding service with API key."""
        self.api_key = api_key
        self.model = model
        genai.configure(api_key=self.api_key)

    def list_embedding_models(self) -> List[str]:
        """
        List all available models that support embeddings.

        Returns:
            List[str]: Names of models supporting embedContent
        """
        try:
            embedding_models = [
                m.name
                for m in genai.list_models()
                if "embedContent" in m.supported_generation_methods
            ]
            logger.info(f"Found {len(embedding_models)} embedding models")
            return embedding_models
        except Exception as e:
            logger.error(f"Error listing embedding models: {str(e)}")
            raise

    def create_embedding(self, document: Document) -> np.ndarray:
        """Create embedding for a document."""
        try:
            embedding = genai.embed_content(
                model=self.model,
                content=document.content,
                task_type=TASK_TYPE,
                title=document.title,
            )
            logger.info(f"Created embedding for document: {document.title}")
            return np.array(embedding["embedding"])
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    def search_documents(
        self, query: str, df: pd.DataFrame, top_k: int = 3
    ) -> pd.DataFrame:
        """Search documents using dot product similarity."""
        query_embedding = genai.embed_content(
            model=self.model, content=query, task_type="retrieval_query"
        )

        # Calculate dot products
        dot_products = np.dot(
            np.stack(df["embedding"].to_numpy()), query_embedding["embedding"]
        )

        # Add scores and sort
        df["score"] = dot_products
        results = df.sort_values("score", ascending=False).head(top_k)
        return results[["title", "content", "score"]]


class QAService:
    """Service for question answering using document search and Gemini."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.gen_model = genai.GenerativeModel("gemini-1.5-pro-latest")

    def make_prompt(self, query: str, relevant_passage: str) -> str:
        """Create prompt for Gemini model."""
        escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
        return textwrap.dedent("""
            You are a helpful and informative bot that answers questions using text from the reference passage included below. \
            Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
            However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
            strike a friendly and conversational tone. \
            If the passage is irrelevant to the answer, you may ignore it.
            
            QUESTION: '{query}'
            PASSAGE: '{relevant_passage}'
            
            ANSWER:
        """).format(query=query, relevant_passage=escaped)

    def answer_question(self, query: str, documents_df: pd.DataFrame) -> str:
        """Search documents and generate answer."""
        # Find most relevant document
        results = self.embedding_service.search_documents(query, documents_df, top_k=1)
        if results.empty:
            return "No relevant information found."

        # Generate answer
        relevant_text = results.iloc[0]["content"]
        prompt = self.make_prompt(query, relevant_text)
        response = self.gen_model.generate_content(prompt)
        return response.text


def create_embeddings_df(
    documents: List[Dict], service: EmbeddingService
) -> pd.DataFrame:
    """Create DataFrame with document metadata and embeddings."""
    rows = []
    for doc_dict in documents:
        doc = Document(title=doc_dict["title"], content=doc_dict["content"])
        embedding = service.create_embedding(doc)
        rows.append(
            {
                "title": doc.title,
                "content": doc.content,
                "embedding": embedding,
                "embedding_size": len(embedding),
            }
        )
    return pd.DataFrame(rows)


def format_search_results(results: pd.DataFrame) -> str:
    """Format search results for display."""
    output = []
    for _, row in results.iterrows():
        output.append(f"Title: {row['title']}")
        output.append(f"Score: {row['score']:.4f}")
        output.append(f"Content: {row['content'][:200]}...\n")
    return "\n".join(output)


def main():
    """Main demo function."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    # Initialize services
    embedding_service = EmbeddingService(api_key)
    qa_service = QAService(embedding_service)

    # Create document database
    df = create_embeddings_df(DOCUMENTS, embedding_service)

    # Demo Q&A
    questions = [
        "How do I control the temperature in the car?",
        "What are the different gear positions?",
        "How do I use the touchscreen?",
    ]

    print("\nQ&A Demo:")
    for question in questions:
        print(f"\nQ: {question}")
        answer = qa_service.answer_question(question, df)
        print(f"A: {answer}\n")
        print("-" * 80)


if __name__ == "__main__":
    main()
