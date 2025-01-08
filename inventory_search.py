import os
from dotenv import load_dotenv
import textwrap
import numpy as np
import pandas as pd
import google.generativeai as genai
import logging
from typing import List, Dict
from dataclasses import dataclass

# Constants
DEFAULT_MODEL = "models/embedding-001"
TASK_TYPE = "retrieval_document"


@dataclass
class PharmacyItem:
    """Represents a pharmacy inventory item with its details."""

    name: str
    description: str
    category: str
    stock_info: str


# Load environment variables
load_dotenv()

# Get API key with error handling
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample pharmacy inventory
INVENTORY_ITEMS = [
    {
        "name": "Acetaminophen 500mg",
        "description": "Over-the-counter pain reliever and fever reducer. Available in tablet form. "
        "Common uses include headache, muscle aches, arthritis, backache, toothaches, "
        "colds, and fevers.",
        "category": "Pain Relief",
        "stock_info": "Available in 50, 100, and 500 count bottles. Current stock: 324 bottles. "
        "Located in Aisle 3, Shelf B. Requires temperature control below 25°C.",
    },
    {
        "name": "Blood Glucose Monitor Kit",
        "description": "Digital blood glucose monitoring system with test strips and lancets. "
        "Includes carrying case and detailed instructions for home testing.",
        "category": "Medical Devices",
        "stock_info": "Available in standard kit format. Current stock: 45 units. "
        "Located in Aisle 1, Shelf A. Recommended storage at room temperature.",
    },
    {
        "name": "Amoxicillin 500mg",
        "description": "Prescription antibiotic used to treat various bacterial infections. "
        "Available in capsule form. Must be dispensed by licensed pharmacist.",
        "category": "Prescription Antibiotics",
        "stock_info": "Prescription only. Current stock: 200 bottles. "
        "Located in Secure Storage Area B. Temperature controlled storage required.",
    },
]


class PharmacyEmbeddingService:
    """Service class to handle pharmacy inventory embeddings."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        """Initialize the embedding service with API key."""
        self.api_key = api_key
        self.model = model
        genai.configure(api_key=self.api_key)

    def create_embedding(self, item: PharmacyItem) -> np.ndarray:
        """Create embedding for a pharmacy item."""
        try:
            # Combine item information for embedding
            content = f"{item.description} {item.stock_info}"
            embedding = genai.embed_content(
                model=self.model,
                content=content,
                task_type=TASK_TYPE,
                title=item.name,
            )
            logger.info(f"Created embedding for item: {item.name}")
            return np.array(embedding["embedding"])
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    def search_inventory(
        self, query: str, df: pd.DataFrame, top_k: int = 3
    ) -> pd.DataFrame:
        """Search inventory using semantic similarity."""
        query_embedding = genai.embed_content(
            model=self.model, content=query, task_type="retrieval_query"
        )

        # Calculate similarities
        dot_products = np.dot(
            np.stack(df["embedding"].to_numpy()), query_embedding["embedding"]
        )

        # Add scores and sort
        df["relevance_score"] = dot_products
        results = df.sort_values("relevance_score", ascending=False).head(top_k)
        return results[
            ["name", "category", "description", "stock_info", "relevance_score"]
        ]


class PharmacyQAService:
    """Service for answering questions about pharmacy inventory."""

    def __init__(self, embedding_service: PharmacyEmbeddingService):
        self.embedding_service = embedding_service
        self.gen_model = genai.GenerativeModel("gemini-1.5-pro-latest")

    def make_prompt(self, query: str, relevant_item: str) -> str:
        """Create prompt for Gemini model."""
        escaped = relevant_item.replace("'", "").replace('"', "").replace("\n", " ")
        return textwrap.dedent("""
            You are a knowledgeable pharmacy assistant helping with inventory queries. \
            Provide accurate, clear information based on the reference information below. \
            For prescription medications, always include a reminder that a prescription is required. \
            Do not provide medical advice or dosage recommendations. \
            If the information is irrelevant to the query, acknowledge that and suggest speaking with a pharmacist.

            QUERY: '{query}'
            INVENTORY INFORMATION: '{relevant_item}'

            RESPONSE:
        """).format(query=query, relevant_item=escaped)

    def answer_query(self, query: str, inventory_df: pd.DataFrame) -> str:
        """Search inventory and generate response."""
        results = self.embedding_service.search_inventory(query, inventory_df, top_k=1)
        if results.empty:
            return "I couldn't find that item in our inventory. Please speak with a pharmacist for assistance."

        item = results.iloc[0]
        relevant_info = f"{item['name']} - {item['description']} {item['stock_info']}"
        prompt = self.make_prompt(query, relevant_info)
        response = self.gen_model.generate_content(prompt)
        return response.text


def create_inventory_df(
    items: List[Dict], service: PharmacyEmbeddingService
) -> pd.DataFrame:
    """Create DataFrame with inventory items and embeddings."""
    rows = []
    for item_dict in items:
        item = PharmacyItem(**item_dict)
        embedding = service.create_embedding(item)
        rows.append(
            {
                **item_dict,
                "embedding": embedding,
                "embedding_size": len(embedding),
            }
        )
    return pd.DataFrame(rows)


def format_inventory_results(results: pd.DataFrame) -> str:
    """Format inventory search results for display."""
    output = []
    for _, row in results.iterrows():
        output.append(f"Product: {row['name']}")
        output.append(f"Category: {row['category']}")
        output.append(f"Relevance Score: {row['relevance_score']:.4f}")
        output.append(f"Description: {row['description'][:200]}...")
        output.append(f"Stock Information: {row['stock_info']}\n")
    return "\n".join(output)


def main():
    """Main demo function."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    # Initialize services
    embedding_service = PharmacyEmbeddingService(api_key)
    qa_service = PharmacyQAService(embedding_service)

    # Create inventory database
    df = create_inventory_df(INVENTORY_ITEMS, embedding_service)

    # Demo queries
    queries = [
        "Do you have any pain relievers in stock?",
        "Where can I find blood glucose testing supplies?",
        "What antibiotics do you carry?",
    ]

    print("\nPharmacy Inventory Query Demo:")
    for query in queries:
        print(f"\nQ: {query}")
        answer = qa_service.answer_query(query, df)
        print(f"A: {answer}\n")
        print("-" * 80)


if __name__ == "__main__":
    main()
