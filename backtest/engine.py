import numpy as np
import pandas as pd

class MarketMakerBacktester:
    def __init__(self, initial_capital: float = 10000.0, maker_fee_bps: float = 0.0):
        self.capital = initial_capital
        self.maker_fee = maker_fee_bps / 10000.0

    def run_strategy(self, test_df: pd.DataFrame, confidence_threshold: float = 0.60) -> dict:
        df = test_df.copy()
        df['position'] = 0
        
        # 1. SIGNAL-ASSISTED QUOTING
        df.loc[df['prob_up'] > confidence_threshold, 'position'] = 1     
        df.loc[df['prob_up'] < (1 - confidence_threshold), 'position'] = -1 
        
        # 2. ADVERSE SELECTION SIMULATION (The Reality of HFT)
        # Did the price actually go up or down?
        df['price_went_up'] = df['future_mid_price'] > df['mid_price']
        
        np.random.seed(42) # For reproducible noise
        rand_array = np.random.rand(len(df))
        df['is_filled'] = False
        
        # LONG QUOTES (Buying at the Bid):
        # Right Prediction: Sellers didn't hit us. Low chance of fill.
        long_right = (df['position'] == 1) & df['price_went_up']
        # Wrong Prediction: Price crashed, sellers smashed our bid. 100% fill.
        long_wrong = (df['position'] == 1) & ~df['price_went_up']
        
        df.loc[long_right, 'is_filled'] = rand_array[long_right] < 0.35 # Only catch 35% of winning waves
        df.loc[long_wrong, 'is_filled'] = True # 100% chance to catch a falling knife
        
        # SHORT QUOTES (Selling at the Ask):
        short_right = (df['position'] == -1) & ~df['price_went_up']
        short_wrong = (df['position'] == -1) & df['price_went_up']
        
        df.loc[short_right, 'is_filled'] = rand_array[short_right] < 0.35 
        df.loc[short_wrong, 'is_filled'] = True 
        
        # 3. PASSIVE EXECUTION RETURNS (Only calculate if filled)
        long_return = (df['future_bid'] - df['best_bid']) / df['best_bid']
        short_return = (df['best_ask'] - df['future_ask']) / df['best_ask']
        
        df['raw_trade_return'] = 0.0
        df.loc[(df['position'] == 1) & df['is_filled'], 'raw_trade_return'] = long_return
        df.loc[(df['position'] == -1) & df['is_filled'], 'raw_trade_return'] = short_return
        
        # 4. APPLY FEES
        df['net_trade_return'] = np.where(df['is_filled'], df['raw_trade_return'] - (2 * self.maker_fee), 0)
        
        df['cumulative_return'] = (1 + df['net_trade_return']).cumprod()
        df['portfolio_value'] = self.capital * df['cumulative_return']
        
        # --- CALCULATE REALISTIC METRICS ---
        total_pnl = df['portfolio_value'].iloc[-1] - self.capital
        trades_attempted = (df['position'] != 0).sum()
        trades_filled = df['is_filled'].sum()
        
        win_rate = (df[df['is_filled']]['net_trade_return'] > 0).mean() if trades_filled > 0 else 0
        
        mean_ret = df['net_trade_return'].mean()
        std_ret = df['net_trade_return'].std()
        sharpe_ratio = 0.0 if std_ret == 0 else (mean_ret / std_ret) * np.sqrt(len(df))
        
        running_max = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_pnl': total_pnl,
            'trades_attempted': trades_attempted,
            'trades_filled': trades_filled,
            'win_rate_after_fees': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': df['portfolio_value'].iloc[-1],
            'df': df
        }