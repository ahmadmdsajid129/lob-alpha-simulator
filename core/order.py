import time
from enum import Enum

class OrderType(Enum):
    LIMIT = "LIMIT"   # Wait in the queue for a specific price
    MARKET = "MARKET" # Execute immediately at the best available price

class Side(Enum):
    BUY = "BUY"       # Bid
    SELL = "SELL"     # Ask

class Order:
    def __init__(self, order_id: str, side: Side, order_type: OrderType, price: float, quantity: int):
        """
        The fundamental building block of the Limit Order Book.
        """
        self.order_id = order_id
        self.side = side
        self.type = order_type
        self.price = price
        self.quantity = quantity
        # Nanosecond precision timestamp - critical for High-Frequency Trading (HFT)
        self.timestamp = time.time_ns() 

    def __repr__(self):
        return f"Order({self.side.value}, Qty:{self.quantity} @ ₹{self.price:.2f})"