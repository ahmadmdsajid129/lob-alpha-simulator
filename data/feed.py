import random
from core.order import Order, Side, OrderType

class MarketDataFeed:
    def __init__(self, starting_price: float = 100.0, tick_size: float = 0.50):
        self.current_price = starting_price
        self.tick_size = tick_size
        self.order_counter = 0

    def _get_next_id(self) -> str:
        self.order_counter += 1
        return f"ORD-{self.order_counter}"

    def seed_market(self, depth: int = 15) -> list[Order]:
        """Pre-fills the order book with resting limit orders (Market Making)."""
        orders = []
        
        # Create Sellers (Asks) at prices ABOVE the current price
        for i in range(1, depth + 1):
            price = self.current_price + (i * self.tick_size)
            qty = random.randint(10, 100) * 10  # Random block of 100 to 1000 shares
            orders.append(Order(self._get_next_id(), Side.SELL, OrderType.LIMIT, round(price, 2), qty))
            
        # Create Buyers (Bids) at prices BELOW the current price
        for i in range(1, depth + 1):
            price = self.current_price - (i * self.tick_size)
            qty = random.randint(10, 100) * 10
            orders.append(Order(self._get_next_id(), Side.BUY, OrderType.LIMIT, round(price, 2), qty))
            
        return orders

    def generate_random_order(self) -> Order:
        """Simulates a live market event (a new trader entering)."""
        # 80% chance it's a Limit Order (providing liquidity)
        # 20% chance it's a Market Order (taking liquidity)
        is_limit = random.random() < 0.8
        side = Side.BUY if random.random() < 0.5 else Side.SELL
        qty = random.randint(1, 50) * 10
        
        if is_limit:
            # Randomize price near the current market price
            price_shift = random.randint(-5, 5) * self.tick_size
            price = round(self.current_price + price_shift, 2)
            return Order(self._get_next_id(), side, OrderType.LIMIT, price, qty)
        else:
            # Market orders don't specify a price, they just take the best available
            return Order(self._get_next_id(), side, OrderType.MARKET, 0.0, qty)