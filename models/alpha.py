import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DataLogger:
    def __init__(self):
        self.history = []

    def log_tick(self, spread: float, imbalance: float, micro_price: float, mid_price: float):
        """Records a single snapshot of the market state."""
        self.history.append({
            'spread': spread,
            'imbalance': imbalance,
            'micro_mid_diff': micro_price - mid_price, # The difference is a massive ML signal
            'mid_price': mid_price
        })

    def build_dataframe(self) -> pd.DataFrame:
        """Converts the logged history into a Machine Learning dataset."""
        df = pd.DataFrame(self.history)
        
        # SUPERVISED LEARNING TARGET: Did the price go up in the next 5 ticks?
        # We shift the mid_price backwards by 5 steps to let the model look into the "future"
        df['future_mid_price'] = df['mid_price'].shift(-5)
        
        # Target = 1 if the future price is higher than the current price, else 0
        df['target_price_up'] = (df['future_mid_price'] > df['mid_price']).astype(int)
        
        # Drop the last 5 rows since they don't have a known "future" yet
        return df.dropna()

class AlphaModel:
    def __init__(self):
        # Logistic Regression is the standard baseline for binary classification (Up/Down)
        self.model = LogisticRegression()
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        """Trains the ML model on the generated market data."""
        if len(df) < 100:
            return 0.0 # Not enough data
            
        # Features (X) are the signals we engineered
        X = df[['spread', 'imbalance', 'micro_mid_diff']]
        # Target (y) is whether the price actually went up
        y = df['target_price_up']

        # Split data: 80% to train the AI, 20% to test it on data it has never seen
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Test the model and return the accuracy score
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy