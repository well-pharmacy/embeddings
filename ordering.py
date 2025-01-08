import os
from dotenv import load_dotenv
import textwrap
import numpy as np
import pandas as pd
import google.generativeai as genai
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from langdetect import detect
from googletrans import Translator
import arabic_reshaper
from bidi.algorithm import get_display
import uuid
from decimal import Decimal


# Constants
DEFAULT_MODEL = "models/embedding-001"
TASK_TYPE = "retrieval_document"


class PharmacyException(Exception):
    """Base exception class for pharmacy-related errors."""

    pass


class InventoryError(PharmacyException):
    """Raised when inventory operations fail."""

    pass


class PrescriptionError(PharmacyException):
    """Raised when prescription validation fails."""

    pass


@dataclass(frozen=True)
class PharmacyConfig:
    """Configuration settings for pharmacy system."""

    api_key: str
    model_name: str = DEFAULT_MODEL
    min_stock_threshold: int = 10
    max_order_quantity: int = 100
    require_prescription_verification: bool = True
    log_level: int = logging.INFO

    def __post_init__(self):
        """Validate configuration settings."""
        if not self.api_key:
            raise PharmacyException("API key is required")
        if self.min_stock_threshold < 0:
            raise ValueError("Stock threshold cannot be negative")
        if self.max_order_quantity <= 0:
            raise ValueError("Maximum order quantity must be positive")


class OrderLimitError(PharmacyException):
    """Raised when order exceeds allowed limits."""

    pass


class PrescriptionValidationError(PrescriptionError):
    """Raised when prescription validation fails."""

    pass


class StockThresholdError(InventoryError):
    """Raised when stock falls below minimum threshold."""

    pass


@dataclass
class PharmacyItem:
    """Represents a pharmacy inventory item with its details."""

    name: str
    description: str
    category: str
    stock_info: str
    price: Decimal
    quantity_available: int
    requires_prescription: bool = False
    reorder_threshold: int = 50


@dataclass
class Prescription:
    rx_number: str
    patient_id: str
    medication: str
    quantity: int
    expires: datetime
    refills: int
    is_valid: bool = True


# Load environment variables
load_dotenv()

# Get API key with error handling
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Represents a customer order."""

    order_id: str
    item_name: str
    quantity: int
    total_price: float
    prescription_id: Optional[str]
    customer_id: str
    status: str
    timestamp: datetime


@dataclass
class PharmacyMetrics:
    """Track important pharmacy metrics."""

    total_orders: int = 0
    failed_orders: int = 0
    prescription_validations: int = 0
    invalid_prescriptions: int = 0
    low_stock_alerts: int = 0
    api_calls: int = 0
    average_response_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "total_orders": self.total_orders,
            "failed_orders": self.failed_orders,
            "prescription_validations": self.prescription_validations,
            "invalid_prescriptions": self.invalid_prescriptions,
            "low_stock_alerts": self.low_stock_alerts,
            "api_calls": self.api_calls,
            "average_response_time": self.average_response_time,
        }


class InventoryManager:
    """Manages inventory operations and stock updates."""

    def __init__(self, initial_inventory: List[Dict]):
        self.inventory = pd.DataFrame(initial_inventory)

    def check_availability(self, item_name: str, quantity: int) -> bool:
        """Check if requested quantity is available."""
        item = self.inventory[self.inventory["name"] == item_name]
        if item.empty:
            return False
        return item.iloc[0]["quantity_available"] >= quantity

    def update_stock(self, item_name: str, quantity: int) -> bool:
        """Update stock after order placement."""
        if not self.check_availability(item_name, quantity):
            return False

        self.inventory.loc[
            self.inventory["name"] == item_name, "quantity_available"
        ] -= quantity
        return True

    def get_item(self, item_name: str) -> Optional[Dict]:
        """Get item details by name."""
        item = self.inventory[self.inventory["name"] == item_name]
        if item.empty:
            return None
        return item.iloc[0].to_dict()

    def get_quantity(self, item_name: str) -> int:
        """Get available quantity of an item."""
        item = self.inventory[self.inventory["name"] == item_name]
        if item.empty:
            return 0
        return item.iloc[0]["quantity_available"]

    def create_order(self, customer_id: str, items: List[tuple]) -> Optional[Order]:
        """Create an order and update inventory."""
        total_price = 0
        for item_name, quantity in items:
            item = self.get_item(item_name)
            if not item or not self.update_stock(item_name, quantity):
                return None
            total_price += item["unit_price"] * quantity

        order = Order(
            order_id=str(uuid.uuid4()),
            item_name=", ".join([item_name for item_name, _ in items]),
            quantity=sum([quantity for _, quantity in items]),
            total_price=total_price,
            prescription_id=None,
            customer_id=customer_id,
            status="pending",
            timestamp=datetime.now(),
        )
        return order


class OrderProcessor:
    """Handles order processing and validation."""

    def __init__(self, inventory_manager: InventoryManager, prescription_service):
        self.inventory_manager = inventory_manager
        self.prescription_service = prescription_service
        self.orders: List[Order] = []

    def create_order(
        self,
        item_name: str,
        quantity: int,
        customer_id: str,
        prescription_id: Optional[str] = None,
    ) -> Optional[Order]:
        """Create a new order with validation."""
        # Get item details
        item = self.inventory_manager.inventory[
            self.inventory_manager.inventory["name"] == item_name
        ].iloc[0]

        # Check prescription requirement
        if item["requires_prescription"] and not prescription_id:
            raise ValueError("This item requires a valid prescription.")

        # Validate availability
        if not self.inventory_manager.check_availability(item_name, quantity):
            raise ValueError("Requested quantity not available.")

        # Calculate total price
        total_price = item["unit_price"] * quantity

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            item_name=item_name,
            quantity=quantity,
            total_price=total_price,
            prescription_id=prescription_id,
            customer_id=customer_id,
            status="pending",
            timestamp=datetime.now(),
        )

        # Update inventory
        if self.inventory_manager.update_stock(item_name, quantity):
            self.orders.append(order)
            return order
        return None

    def process_order(
        self, item_name: str, quantity: int, rx_number: Optional[str] = None
    ) -> str:
        item = self.inventory_manager.get_item(item_name)

        if not item:
            return "Item not found in inventory"

        if item.requires_prescription:
            if not rx_number:
                return "This medication requires a valid prescription"

            rx = self.prescription_service.validate_prescription(rx_number)
            if not rx:
                return "Invalid or expired prescription"

            if rx.medication != item_name:
                return "Prescription does not match medication"

        if not self.inventory_manager.check_availability(item_name, quantity):
            return f"Insufficient stock. Available: {self.inventory_manager.get_quantity(item_name)}"

        order_id = str(uuid.uuid4())
        total_price = item["price"] * quantity

        self.orders.append(
            {
                "item": item_name,
                "quantity": quantity,
                "total": total_price,
                "status": "confirmed",
                "rx_number": rx_number,
            }
        )

        return f"Order confirmed. Order ID: {order_id}, Total: ${total_price}"


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


class MultilingualQAService(PharmacyQAService):
    """Multilingual question answering service."""

    def __init__(self, embedding_service: PharmacyEmbeddingService):
        super().__init__(embedding_service)
        self.translator = Translator()

    def format_arabic_text(self, text: str) -> str:
        """Format Arabic text for proper display."""
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)

    def answer_query(self, query: str, documents_df: pd.DataFrame) -> str:
        """Process multilingual queries and return answers."""
        try:
            # Detect language
            lang = detect(query)

            # Translate if not English
            if lang != "en":
                eng_query = self.translator.translate(query, dest="en").text
            else:
                eng_query = query

            # Get answer in English
            answer = super().answer_query(eng_query, documents_df)

            # Translate back if needed
            if lang != "en":
                answer = self.translator.translate(answer, dest=lang).text

                # Special handling for Arabic
                if lang == "ar":
                    answer = self.format_arabic_text(answer)

            return answer

        except Exception as e:
            logger.error(f"Error processing multilingual query: {str(e)}")
            return "Sorry, there was an error processing your question."


class EnhancedPharmacyQAService(MultilingualQAService):
    """Extended QA service with order processing capabilities."""

    def __init__(
        self,
        embedding_service: PharmacyEmbeddingService,
        order_processor: OrderProcessor,
    ):
        super().__init__(embedding_service)
        self.order_processor = order_processor

    def make_prompt(self, query: str, relevant_item: str) -> str:
        """Create prompt for Gemini model with ordering information."""
        escaped = relevant_item.replace("'", "").replace('"', "").replace("\n", " ")
        return textwrap.dedent("""
            You are a knowledgeable pharmacy assistant helping with inventory queries and orders. \
            Provide accurate, clear information based on the reference information below. \
            For prescription medications, always include a reminder that a prescription is required. \
            Do not provide medical advice or dosage recommendations. \
            If the item is available, mention that the customer can place an order and ask if they would like to do so. \
            For prescription items, remind them they'll need to provide a valid prescription ID. \
            If the information is irrelevant to the query, acknowledge that and suggest speaking with a pharmacist.

            QUERY: '{query}'
            INVENTORY INFORMATION: '{relevant_item}'

            RESPONSE:
        """).format(query=query, relevant_item=escaped)

    def process_order_request(
        self,
        item_name: str,
        quantity: int,
        customer_id: str,
        prescription_id: Optional[str] = None,
    ) -> str:
        """Process order request and return response."""
        try:
            order = self.order_processor.create_order(
                item_name=item_name,
                quantity=quantity,
                customer_id=customer_id,
                prescription_id=prescription_id,
            )

            if order:
                return f"""
                Order successfully placed!
                Order ID: {order.order_id}
                Item: {order.item_name}
                Quantity: {order.quantity}
                Total Price: ${order.total_price:.2f}
                Status: {order.status}
                """
            return (
                "Unable to process order. Please try again or speak with a pharmacist."
            )

        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.error(f"Error processing order: {str(e)}")
            return "An error occurred while processing your order. Please try again."


class PrescriptionOrderService:
    def __init__(self, qa_service, inventory_manager):
        self.qa_service = qa_service
        self.inventory = inventory_manager
        self.prescriptions: Dict[str, Prescription] = {}

    def verify_prescription(
        self, rx_number: str, patient_id: str
    ) -> Optional[Prescription]:
        rx = self.prescriptions.get(rx_number)
        if not rx or rx.patient_id != patient_id:
            return None
        if rx.expires < datetime.now() or rx.refills <= 0:
            return None
        return rx

    def process_order_request(
        self,
        item_name: str,
        quantity: int,
        customer_id: str,
        rx_number: Optional[str] = None,
    ) -> str:
        # Check if item requires prescription
        item = self.inventory.get_item(item_name)
        if not item:
            return "Sorry, that item is not available."

        if item["requires_prescription"]:
            if not rx_number:
                return "This medication requires a valid prescription. Please provide your prescription number."

            rx = self.verify_prescription(rx_number, customer_id)
            if not rx:
                return "Invalid or expired prescription. Please contact your healthcare provider."

            if rx.medication != item_name:
                return "Prescription does not match requested medication."

        # Check stock
        if not self.inventory.check_availability(item_name, quantity):
            return f"Sorry, we only have {self.inventory.get_quantity(item_name)} units in stock."

        # Create order
        order = self.inventory.create_order(customer_id, [(item_name, quantity)])
        if not order:
            return "Error processing order."

        # Update prescription if applicable
        if rx_number:
            rx = self.prescriptions[rx_number]
            rx.refills -= 1

        return f"Order placed successfully. Order ID: {order.order_id}"


class PrescriptionService:
    def __init__(self):
        self.prescriptions = {
            "RX123456": Prescription(
                rx_number="RX123456",
                patient_id="PATIENT001",
                medication="Amoxicillin 500mg",
                quantity=30,
                expires=datetime(2024, 12, 31),
                refills=3,
            )
        }

    def validate_prescription(self, rx_number: str) -> Optional[Prescription]:
        rx = self.prescriptions.get(rx_number)
        if not rx:
            return None
        if not rx.is_valid or rx.expires < datetime.now() or rx.refills <= 0:
            return None
        return rx


def create_inventory_df(
    items: List[Dict], service: PharmacyEmbeddingService
) -> pd.DataFrame:
    """Create DataFrame with embeddings from inventory items."""
    rows = []
    for item_dict in items:
        # Convert price to Decimal
        if "price" in item_dict:
            item_dict["price"] = Decimal(str(item_dict["price"]))

        # Create PharmacyItem instance
        try:
            item = PharmacyItem(
                name=item_dict["name"],
                description=item_dict["description"],
                category=item_dict["category"],
                stock_info=item_dict["stock_info"],
                price=item_dict["price"],
                quantity_available=item_dict["quantity_available"],
            )

            # Create embedding
            embedding = service.create_embedding(item)

            rows.append(
                {**item_dict, "embedding": embedding, "embedding_size": len(embedding)}
            )

        except KeyError as e:
            logger.error(f"Missing required field in item data: {e}")
            continue
        except Exception as e:
            logger.error(
                f"Error processing item {item_dict.get('name', 'unknown')}: {e}"
            )
            continue

    return pd.DataFrame(rows)


# Updated inventory items with proper price format
INVENTORY_ITEMS = [
    {
        "name": "Amoxicillin 500mg",
        "description": "Antibiotic medication for bacterial infections",
        "category": "Antibiotics",
        "stock_info": "Prescription only. Temperature controlled storage.",
        "price": "15.99",
        "quantity_available": 200,
        "requires_prescription": True,
        "reorder_threshold": 50,
    },
    {
        "name": "Acetaminophen 500mg",
        "description": "Over-the-counter pain reliever and fever reducer. Available in tablet form. "
        "Common uses include headache, muscle aches, arthritis, backache, toothaches, "
        "colds, and fevers.",
        "category": "Pain Relief",
        "stock_info": "Available in 50, 100, and 500 count bottles. Located in Aisle 3, Shelf B.",
        "requires_prescription": False,
        "unit_price": 9.99,
        "quantity_available": 324,
    },
    {
        "name": "Blood Glucose Monitor Kit",
        "description": "Digital blood glucose monitoring system with test strips and lancets. "
        "Includes carrying case and detailed instructions for home testing.",
        "category": "Medical Devices",
        "stock_info": "Available in standard kit format. Located in Aisle 1, Shelf A.",
        "requires_prescription": False,
        "unit_price": 49.99,
        "quantity_available": 45,
    },
    {
        "name": "Amoxicillin 500mg",
        "description": "Prescription antibiotic used to treat various bacterial infections. "
        "Available in capsule form. Must be dispensed by licensed pharmacist.",
        "category": "Prescription Antibiotics",
        "stock_info": "Prescription only. Located in Secure Storage Area B.",
        "requires_prescription": True,
        "unit_price": 24.99,
        "quantity_available": 200,
    },
]

INVENTORY = [
    {
        "name": "Amoxicillin 500mg",
        "requires_prescription": True,
        "unit_price": 24.99,
        "quantity_available": 200,
    },
    {
        "name": "Aspirin 81mg",
        "requires_prescription": False,
        "unit_price": 9.99,
        "quantity_available": 500,
    },
]


def create_order(
    item_name: str,
    quantity: int,
    customer_id: str,
    rx_number: Optional[str] = None,
    metrics: Optional[PharmacyMetrics] = None,
) -> Order:
    """Create a new order with validation."""

    if metrics:
        metrics.total_orders += 1

    # Generate order ID
    order_id = f"ORD-{uuid.uuid4().hex[:8]}"

    # Validate item exists in inventory
    item = next((i for i in INVENTORY_ITEMS if i["name"] == item_name), None)
    if not item:
        raise ValueError(f"Item not found: {item_name}")

    # Check prescription requirement
    if item["requires_prescription"]:
        if not rx_number:
            raise ValueError("Prescription required for this medication")
        if not validate_prescription(rx_number):
            if metrics:
                metrics.invalid_prescriptions += 1
            raise ValueError("Invalid prescription")

    # Verify stock
    if quantity > item["quantity_available"]:
        if metrics:
            metrics.low_stock_alerts += 1
        raise ValueError("Insufficient stock")

    # Calculate total price
    total_price = item["unit_price"] * quantity

    # Create order with current timestamp
    order = Order(
        order_id=order_id,
        item_name=item_name,
        quantity=quantity,
        total_price=total_price,
        prescription_id=rx_number,
        customer_id=customer_id,
        status="PENDING",
        timestamp=datetime.now(),
    )

    return order


def validate_prescription(rx_number: str) -> bool:
    """Mock prescription validation."""
    # In real implementation, would call external validation service
    return bool(rx_number and rx_number.startswith("RX"))


def check_inventory(item_name: str, quantity: int) -> bool:
    """Mock inventory check."""
    # In real implementation, would check actual inventory
    return True


def calculate_price(item_name: str, quantity: int) -> float:
    """Mock price calculation."""
    # In real implementation, would look up actual prices
    base_price = 10.0
    return base_price * quantity


def main():
    """Main demo function with ordering capability."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    # Initialize services
    embedding_service = PharmacyEmbeddingService(api_key)
    inventory_manager = InventoryManager(INVENTORY_ITEMS)
    prescription_service = PrescriptionService()
    order_processor = OrderProcessor(inventory_manager, prescription_service)
    qa_service = EnhancedPharmacyQAService(embedding_service, order_processor)
    prescription_order_service = PrescriptionOrderService(qa_service, inventory_manager)

    # Create inventory database
    df = create_inventory_df(INVENTORY_ITEMS, embedding_service)

    # Demo queries and orders
    print("\nInventory Query and Order Demo:")

    # Example 1: Query and order OTC medication
    query = "Do you have acetaminophen in stock?"
    print(f"\nCustomer: {query}")
    answer = qa_service.answer_query(query, df)
    print(f"Assistant: {answer}")

    # Place order
    print("\nCustomer: I'd like to order 2 bottles.")
    order_response = qa_service.process_order_request(
        item_name="Acetaminophen 500mg", quantity=2, customer_id="CUST123"
    )
    print(f"Assistant: {order_response}")

    # Example 2: Query and order prescription medication
    query = "I need amoxicillin."
    print(f"\nCustomer: {query}")
    answer = qa_service.answer_query(query, df)
    print(f"Assistant: {answer}")

    # Attempt to order without prescription
    print("\nCustomer: I'd like to order some.")
    order_response = qa_service.process_order_request(
        item_name="Amoxicillin 500mg", quantity=1, customer_id="CUST123"
    )
    print(f"Assistant: {order_response}")

    # Attempt to order with prescription
    print("\nCustomer: Here's my prescription: RX123456")
    order_with_rx = prescription_order_service.process_order_request(
        item_name="Amoxicillin 500mg",
        quantity=1,
        customer_id="CUST123",
        rx_number="RX123456",
    )
    print(f"Assistant: {order_with_rx}")

    # Initialize metrics
    metrics = PharmacyMetrics()

    try:
        # Basic order example
        order_no_rx = create_order(
            item_name="Aspirin 81mg", quantity=2, customer_id="CUST789"
        )
        metrics.total_orders += 1
        print(f"Basic order: {order_no_rx}")

        # Order with prescription
        order_with_rx = create_order(
            item_name="Amoxicillin 500mg",
            quantity=1,
            customer_id="CUST123",
            rx_number="RX123456",
        )
        metrics.total_orders += 1
        metrics.prescription_validations += 1
        print(f"Order with prescription: {order_with_rx}")

        # Print metrics summary
        print("\nPharmacy Metrics Summary:")
        for key, value in metrics.to_dict().items():
            print(f"{key}: {value}")

    except Exception as e:
        metrics.failed_orders += 1
        logger.error(f"Error processing orders: {str(e)}")
        raise

    metrics = PharmacyMetrics()

    try:
        # Basic order with non-prescription item
        order_no_rx = create_order(
            item_name="Aspirin 81mg",  # Now exists in inventory
            quantity=2,
            customer_id="CUST789",
        )
        metrics.total_orders += 1
        print(f"Basic order created: {order_no_rx}")

        # Order with prescription
        order_with_rx = create_order(
            item_name="Amoxicillin 500mg",
            quantity=1,
            customer_id="CUST123",
            rx_number="RX123456",
        )
        metrics.total_orders += 1
        print(f"Prescription order created: {order_with_rx}")

    except ValueError as e:
        metrics.failed_orders += 1
        logger.error(f"Order creation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
