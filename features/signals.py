from core.engine import LimitOrderBook

class MicrostructureFeatures:
    @staticmethod
    def calculate_spread(book: LimitOrderBook) -> float:
        """The gap between the highest buyer and lowest seller. A tightening spread means incoming volatility."""
        if book.best_bid > 0 and book.best_ask < float('inf'):
            return book.best_ask - book.best_bid
        return 0.0

    @staticmethod
    def calculate_imbalance(book: LimitOrderBook) -> float:
        """
        Order Book Imbalance (OBI). 
        Ranges from -1.0 to 1.0.
        +1.0 = Massive buy pressure (Price likely to rise).
        -1.0 = Massive sell pressure (Price likely to fall).
        """
        if book.best_bid not in book.bids or book.best_ask not in book.asks:
            return 0.0
            
        bid_vol = sum(o.quantity for o in book.bids[book.best_bid])
        ask_vol = sum(o.quantity for o in book.asks[book.best_ask])
        
        if bid_vol + ask_vol == 0:
            return 0.0
            
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    @staticmethod
    def calculate_micro_price(book: LimitOrderBook) -> float:
        """
        Volume-weighted mid-price. 
        Instead of a simple average, it pulls the "fair value" toward the side with more liquidity.
        """
        if book.best_bid not in book.bids or book.best_ask not in book.asks:
            return 0.0
            
        bid_vol = sum(o.quantity for o in book.bids[book.best_bid])
        ask_vol = sum(o.quantity for o in book.asks[book.best_ask])
        
        if bid_vol + ask_vol == 0:
            return (book.best_bid + book.best_ask) / 2
            
        # Notice the cross-weighting: Ask volume weights the Bid price, and Bid volume weights the Ask price.
        return (book.best_bid * ask_vol + book.best_ask * bid_vol) / (bid_vol + ask_vol)