from decimal import Decimal
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import google.generativeai as genai
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class InventoryItem:
    name: str
    description: str
    category: str
    stock_info: str
    quantity: int
    price: float
    embedding: Optional[np.ndarray] = None


class InventorySearch:
    def __init__(self, api_key: str):
        self.model = "models/embedding-001"
        genai.configure(api_key=api_key)
        self.items_df = None

    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text using Gemini API."""
        result = genai.embed_content(
            model=self.model, content=text, task_type="retrieval_document"
        )
        return np.array(result["embedding"])

    def index_items(self, items: List[Dict]) -> None:
        """Create search index from inventory items."""
        rows = []
        for item in items:
            # Combine item text for embedding
            text = f"{item['name']} {item['description']} {item['category']}"
            embedding = self.create_embedding(text)

            rows.append({**item, "embedding": embedding})

        self.items_df = pd.DataFrame(rows)

    def search(self, query: str, top_k: int = 3) -> pd.DataFrame:
        """Search inventory using query embedding."""
        query_embedding = self.create_embedding(query)

        # Calculate similarity scores
        scores = np.dot(
            np.stack(self.items_df["embedding"].to_numpy()), query_embedding
        )

        # Return top matches
        self.items_df["score"] = scores
        results = self.items_df.sort_values("score", ascending=False).head(top_k)
        return results[["name", "description", "category", "stock_info", "score"]]


def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    INVENTORY_ITEMS = [
        {
            "name": "Acetaminophen 500mg",
            "description": "Over-the-counter pain reliever and fever reducer. Safe for most adults and children.",
            "category": "Pain Relief",
            "stock_info": "Available in 50, 100, and 500 count bottles",
            "price": Decimal("9.99"),
            "quantity_available": 324,
            "reorder_threshold": 50,
        },
        {
            "name": "Ibuprofen 200mg",
            "description": "Anti-inflammatory pain reliever for headaches, muscle aches, and fever reduction.",
            "category": "Pain Relief",
            "stock_info": "Available in 30, 90 count bottles",
            "price": Decimal("8.49"),
            "quantity_available": 256,
            "reorder_threshold": 40,
        },
        {
            "name": "Digital Thermometer",
            "description": "Fast-reading digital thermometer with LCD display and fever alert.",
            "category": "Medical Devices",
            "stock_info": "Individual units with protective case",
            "price": Decimal("12.99"),
            "quantity_available": 89,
            "reorder_threshold": 20,
        },
        {
            "name": "Vitamin D3 1000IU",
            "description": "Daily supplement for bone health and immune system support.",
            "category": "Supplements",
            "stock_info": "90 count bottles",
            "price": Decimal("15.99"),
            "quantity_available": 178,
            "reorder_threshold": 30,
        },
        {
            "name": "First Aid Kit",
            "description": "Complete emergency kit with bandages, antiseptic wipes, and basic medical supplies.",
            "category": "First Aid",
            "stock_info": "Compact travel size",
            "price": Decimal("24.99"),
            "quantity_available": 45,
            "reorder_threshold": 15,
        },
        {
            "name": "Blood Pressure Monitor",
            "description": "Automatic digital blood pressure monitor with memory function.",
            "category": "Medical Devices",
            "stock_info": "Includes carry case and batteries",
            "price": Decimal("49.99"),
            "quantity_available": 32,
            "reorder_threshold": 10,
        },
    ]
    # Initialize search
    search = InventorySearch(api_key)
    search.index_items(INVENTORY_ITEMS)

    # Example searches
    queries = ["pain medicine", "fever reducer", "headache treatment"]

    for query in queries:
        print(f"\nSearch: {query}")
        results = search.search(query)
        print(results[["name", "score"]].to_string())


if __name__ == "__main__":
    main()
