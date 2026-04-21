import time
from core.engine import LimitOrderBook
from data.feed import MarketDataFeed
from features.signals import MicrostructureFeatures
from models.alpha import DataLogger, AlphaModel

def main():
    print("Initializing High-Frequency Market Simulator & ML Engine...")
    
    engine = LimitOrderBook()
    feed = MarketDataFeed(starting_price=100.0, tick_size=0.10)
    logger = DataLogger()
    ml_model = AlphaModel()
    
    # 1. Seed the Market
    print("Market Makers seeding the order book...")
    for order in feed.seed_market(depth=10):
        engine.process_order(order)
        
    # 2. Simulate High-Frequency Trading & Log Data
    num_events = 5000
    print(f"Simulating {num_events} live market events and capturing signals...")
    
    for i in range(num_events):
        new_order = feed.generate_random_order()
        engine.process_order(new_order)
        
        # Only log data if the market is stable (has both buyers and sellers)
        if engine.best_bid > 0 and engine.best_ask < float('inf'):
            spread = MicrostructureFeatures.calculate_spread(engine)
            imb = MicrostructureFeatures.calculate_imbalance(engine)
            micro = MicrostructureFeatures.calculate_micro_price(engine)
            mid = (engine.best_bid + engine.best_ask) / 2
            
            # Record this exact millisecond into the Pandas dataset
            logger.log_tick(spread, imb, micro, mid)

    print("\n--- DATA COLLECTION COMPLETE ---")
    
    # 3. Build Dataset and Train AI
    print("Structuring time-series DataFrame...")
    df = logger.build_dataframe()
    print(f"Dataset generated with {len(df)} market snapshots.")
    
    print("\nTraining Alpha Model (Logistic Regression)...")
    accuracy = ml_model.train(df)
    
    print("\n=========================================")
    print("🏆 ALPHA MODEL RESULTS")
    print("=========================================")
    print(f"Predictive Accuracy: {accuracy * 100:.2f}%")
    if accuracy > 0.50:
        print("Status: Edge Detected. Model beats random guessing.")
    else:
        print("Status: No Edge. Model is performing worse than a coin flip.")
    print("=========================================")

if __name__ == "__main__":
    main()