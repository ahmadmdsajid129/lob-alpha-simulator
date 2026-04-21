from collections import deque
from core.order import Order, Side, OrderType

class LimitOrderBook:
    def __init__(self):
        self.bids = {}  
        self.asks = {}  
        self.best_bid = 0.0
        self.best_ask = float('inf')

    def process_order(self, order: Order):
        """Main routing engine for incoming orders."""
        if order.type == OrderType.MARKET:
            self._match_order(order)
            
        elif order.type == OrderType.LIMIT:
            # 1. Does this Limit Order cross the spread? If yes, execute it instantly.
            if (order.side == Side.BUY and order.price >= self.best_ask) or \
               (order.side == Side.SELL and order.price <= self.best_bid):
                self._match_order(order, limit_price=order.price)
            
            # 2. If it didn't cross, or if it partially filled and has shares left, rest in the book
            if order.quantity > 0:
                self._add_limit_order(order)

    def _add_limit_order(self, order: Order):
        """Places a resting order in the book."""
        if order.side == Side.BUY:
            if order.price not in self.bids:
                self.bids[order.price] = deque()
            self.bids[order.price].append(order)
            if order.price > self.best_bid:
                self.best_bid = order.price
        else: # SELL
            if order.price not in self.asks:
                self.asks[order.price] = deque()
            self.asks[order.price].append(order)
            if order.price < self.best_ask:
                self.best_ask = order.price

    def _match_order(self, order: Order, limit_price: float = None):
        """Aggressively eats resting orders. Respects limit_price if provided."""
        remaining_qty = order.quantity
        
        if order.side == Side.BUY:
            while remaining_qty > 0 and self.asks:
                best_ask_price = min(self.asks.keys())
                
                # Stop executing if the cheapest seller is more expensive than our limit
                if limit_price is not None and best_ask_price > limit_price:
                    break
                    
                queue = self.asks[best_ask_price]
                while queue and remaining_qty > 0:
                    resting_order = queue[0]
                    if resting_order.quantity <= remaining_qty:
                        remaining_qty -= resting_order.quantity
                        queue.popleft()
                    else:
                        resting_order.quantity -= remaining_qty
                        remaining_qty = 0
                        
                if not queue:
                    del self.asks[best_ask_price]
                    
        else: # SELL
            while remaining_qty > 0 and self.bids:
                best_bid_price = max(self.bids.keys())
                
                # Stop executing if the highest buyer is cheaper than our limit
                if limit_price is not None and best_bid_price < limit_price:
                    break
                    
                queue = self.bids[best_bid_price]
                while queue and remaining_qty > 0:
                    resting_order = queue[0]
                    if resting_order.quantity <= remaining_qty:
                        remaining_qty -= resting_order.quantity
                        queue.popleft()
                    else:
                        resting_order.quantity -= remaining_qty
                        remaining_qty = 0
                        
                if not queue:
                    del self.bids[best_bid_price]
                    
        # Recalculate spread bounds
        self.best_bid = max(self.bids.keys()) if self.bids else 0.0
        self.best_ask = min(self.asks.keys()) if self.asks else float('inf')
        
        # Update the incoming order's quantity so the system knows what's left
        order.quantity = remaining_qty

    def print_book(self, levels: int = 5):
        """Visualizes market depth."""
        print("\n=== LIMIT ORDER BOOK ===")
        sorted_asks = sorted(self.asks.keys(), reverse=True)
        for p in sorted_asks[-levels:]:
            vol = sum(o.quantity for o in self.asks[p])
            print(f"ASK: {vol:4} @ ₹{p:.2f}")
            
        print("-" * 24)
        
        sorted_bids = sorted(self.bids.keys(), reverse=True)
        for p in sorted_bids[:levels]:
            vol = sum(o.quantity for o in self.bids[p])
            print(f"BID: {vol:4} @ ₹{p:.2f}")
            
        mid_price = (self.best_bid + self.best_ask) / 2 if (self.best_bid > 0 and self.best_ask < float('inf')) else 0.0
        print(f"\nMid-Price: ₹{mid_price:.2f}")
        print("========================\n")