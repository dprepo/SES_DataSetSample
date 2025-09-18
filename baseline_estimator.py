import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

class BaselineEstimator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.model = LinearRegression()
        self.is_fitted = False
    
    def fit(self, descriptions, story_points):
        """Train baseline model"""
        X = self.vectorizer.fit_transform(descriptions)
        self.model.fit(X, story_points)
        self.is_fitted = True
    
    def predict(self, descriptions):
        """Predict story points"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X = self.vectorizer.transform(descriptions)
        predictions = self.model.predict(X)
        return np.round(predictions).astype(int)
    
    def evaluate_metrics(self, y_true, y_pred):
        """Calculate baseline metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAPE': mape
        }