import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DataLogger:
    def __init__(self):
        self.history = []

    def log_tick(self, spread: float, imbalance: float, micro_price: float, mid_price: float, best_bid: float, best_ask: float):
        """Records a single snapshot of the market state including executable bid/ask prices."""
        self.history.append({
            'spread': spread,
            'imbalance': imbalance,
            'micro_mid_diff': micro_price - mid_price,
            'mid_price': mid_price,
            'best_bid': best_bid,   
            'best_ask': best_ask
        })

    def build_dataframe(self) -> pd.DataFrame:
        """Converts the logged history into a time-series Machine Learning dataset."""
        df = pd.DataFrame(self.history)
        
        # Look 5 ticks into the future for our exit prices
        df['future_mid_price'] = df['mid_price'].shift(-5)
        df['future_bid'] = df['best_bid'].shift(-5)
        df['future_ask'] = df['best_ask'].shift(-5)
        
        # Target = 1 if the future mid-price is strictly greater than current mid-price
        df['target_price_up'] = (df['future_mid_price'] > df['mid_price']).astype(int)
        
        # Drop the last few rows since they don't have a known "future" yet
        return df.dropna()

class AlphaModel:
    def __init__(self):
        # XGBoost is vastly superior for capturing non-linear market microstructure
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            eval_metric='logloss',
            random_state=42
        )

    def train_and_evaluate(self, df: pd.DataFrame):
        """Trains the ML model and returns advanced metrics + out-of-sample predictions."""
        if len(df) < 100:
            return None, None
            
        X = df[['spread', 'imbalance', 'micro_mid_diff']]
        y = df['target_price_up']

        # CRITICAL FIX: shuffle=False ensures we train on the past and test on the future. No data leakage.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        # Get the actual probability of the price going UP
        probabilities = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0)
        }
        
        # Reconstruct the test dataframe so the Backtester knows the actual prices to trade on
        test_df = df.loc[X_test.index].copy()
        test_df['prediction'] = predictions
        test_df['prob_up'] = probabilities
        
        return metrics, test_df