import time
import matplotlib.pyplot as plt
from core.engine import LimitOrderBook
from data.feed import MarketDataFeed
from features.signals import MicrostructureFeatures
from models.alpha import DataLogger, AlphaModel
from backtest.engine import MarketMakerBacktester

def main():
    print("Initializing XGBoost Alpha Engine & Market Maker Simulator...")
    
    engine = LimitOrderBook()
    feed = MarketDataFeed(starting_price=100.0, tick_size=0.10)
    logger = DataLogger()
    ml_model = AlphaModel()
    
    backtester = MarketMakerBacktester(initial_capital=10000.0, maker_fee_bps=0.0) 
    
    print("Market Makers seeding the order book...")
    for order in feed.seed_market(depth=10):
        engine.process_order(order)
        
    num_events = 5000
    print(f"Simulating {num_events} live market events...")
    
    for i in range(num_events):
        new_order = feed.generate_random_order()
        engine.process_order(new_order)
        
        if engine.best_bid > 0 and engine.best_ask < float('inf'):
            spread = MicrostructureFeatures.calculate_spread(engine)
            imb = MicrostructureFeatures.calculate_imbalance(engine)
            micro = MicrostructureFeatures.calculate_micro_price(engine)
            mid = (engine.best_bid + engine.best_ask) / 2
            logger.log_tick(spread, imb, micro, mid, engine.best_bid, engine.best_ask)

    df = logger.build_dataframe()
    ml_metrics, test_df = ml_model.train_and_evaluate(df)
    
    if ml_metrics is None:
        return

    print("\n=========================================")
    print("🧠 XGBOOST METRICS (Strict Time Split)")
    print("=========================================")
    print(f"Accuracy:  {ml_metrics['accuracy'] * 100:.2f}%")
    print(f"F1-Score:  {ml_metrics['f1'] * 100:.2f}%")
    
    print("\n=========================================")
    print("📈 REALISTIC MARKET MAKING BACKTEST")
    print("=========================================")
    print("Executing with Adverse Selection Frictions (Low Fill Rate on Winners)...")
    
    bt_results = backtester.run_strategy(test_df, confidence_threshold=0.60)
    
    print(f"Trades Attempted:   {bt_results['trades_attempted']}")
    print(f"Trades Filled:      {bt_results['trades_filled']} (Missed {bt_results['trades_attempted'] - bt_results['trades_filled']} winners)")
    print(f"Win Rate (Net):     {bt_results['win_rate_after_fees'] * 100:.2f}%")
    print(f"Total Net PnL:      ${bt_results['total_pnl']:.2f}")
    print(f"Sharpe Ratio:       {bt_results['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown:   {bt_results['max_drawdown'] * 100:.2f}%")
    
    print("\nGenerating Performance Chart...")
    plot_df = bt_results['df']
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(plot_df)), plot_df['portfolio_value'], color='#10b981', linewidth=2)
    plt.title('Realistic Market Making Strategy (Adverse Selection Included)', fontsize=14, color='white')
    plt.xlabel('Time (Ticks)', fontsize=12, color='lightgray')
    plt.ylabel('Portfolio Value ($)', fontsize=12, color='lightgray')
    plt.grid(color='#1e293b', linestyle='--', linewidth=0.5)
    
    ax = plt.gca()
    ax.set_facecolor('#0f172a')
    plt.gcf().set_facecolor('#0f172a')
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.tick_params(colors='lightgray')
    
    plt.tight_layout()
    plt.savefig('strategy_performance.png', facecolor='#0f172a')
    print("✅ Graph saved to your folder as 'strategy_performance.png'!")

if __name__ == "__main__":
    main()