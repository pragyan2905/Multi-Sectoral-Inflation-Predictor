import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class InflationPredictor:
    """Base class for inflation prediction models"""
    
    def __init__(self, model_type="overall"):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_type = model_type
        self.is_trained = False
        
    def preprocess_data(self, data):
        """Preprocess data for training or prediction"""
        raise NotImplementedError("Subclasses must implement preprocess_data")
    
    def train(self, X, y):
        """Train the model"""
        raise NotImplementedError("Subclasses must implement train")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        return metrics, y_pred

class OverallInflationPredictor(InflationPredictor):
    """Overall inflation predictor using economic indicators"""
    
    def __init__(self):
        super().__init__("overall")
        self.feature_names = [
            'gdp', 'gdp_per_capita', 'unemployment_rate', 'real_interest_rate',
            'gdp_deflator', 'gdp_growth', 'current_account_balance', 'gov_expense',
            'gov_revenue', 'tax_revenue', 'gni', 'public_debt',
            'debt_to_gdp_ratio', 'gdp_growth_per_capita', 'gov_balance'
        ]
    
    def preprocess_data(self, df):
        """Preprocess World Bank economic data"""
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        
        # Rename columns to standard format
        column_mapping = {
            'country_name': 'country',
            'country_id': 'country_code',
            'year': 'year',
            'inflation_(cpi_%)': 'cpi',
            'gdp_(current_usd)': 'gdp',
            'gdp_per_capita_(current_usd)': 'gdp_per_capita',
            'unemployment_rate_(%)': 'unemployment_rate',
            'interest_rate_(real,_%)': 'real_interest_rate',
            'inflation_(gdp_deflator,_%)': 'gdp_deflator',
            'gdp_growth_(%_annual)': 'gdp_growth',
            'current_account_balance_(%_gdp)': 'current_account_balance',
            'government_expense_(%_of_gdp)': 'gov_expense',
            'government_revenue_(%_of_gdp)': 'gov_revenue',
            'tax_revenue_(%_of_gdp)': 'tax_revenue',
            'gross_national_income_(usd)': 'gni',
            'public_debt_(%_of_gdp)': 'public_debt',
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Drop unnecessary columns
        if 'country_code' in df.columns:
            df.drop(['country_code'], axis=1, inplace=True)
        
        # Handle missing values
        for col in df.select_dtypes(include=np.number).columns:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Feature engineering
        df['debt_to_gdp_ratio'] = df['public_debt'] / df['gdp']
        df['gdp_growth_per_capita'] = df['gdp_growth'] / df['gdp_per_capita']
        df['gov_balance'] = df['gov_revenue'] - df['gov_expense']
        
        # Handle infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill new feature NaNs with median
        for col in ['debt_to_gdp_ratio', 'gdp_growth_per_capita', 'gov_balance']:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def train(self, df, target_col='cpi'):
        """Train the overall inflation model"""
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model with optimized parameters
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=0.7,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        metrics, y_pred = self.evaluate(X_test_scaled, y_test)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': metrics
        }
    
    def predict_single(self, gdp, gdp_per_capita, unemployment_rate, real_interest_rate,
                      public_debt, gov_expense, gov_revenue, gni):
        """Make a single prediction with user inputs"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create feature vector
        debt_to_gdp_ratio = public_debt / gdp if gdp > 0 else 0
        gdp_growth_per_capita = 0.02  # Default assumption
        gov_balance = gov_revenue - gov_expense
        
        features = np.array([[
            gdp, gdp_per_capita, unemployment_rate, real_interest_rate,
            0.02,  # gdp_deflator (default)
            0.03,  # gdp_growth (default)
            0.01,  # current_account_balance (default)
            gov_expense, gov_revenue,
            gov_revenue * 0.8,  # tax_revenue (estimate)
            gni, public_debt,
            debt_to_gdp_ratio, gdp_growth_per_capita, gov_balance
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)
        return prediction[0]

class FoodInflationPredictor(InflationPredictor):
    """Food inflation predictor using time-series features"""
    
    def __init__(self):
        super().__init__("food")
        self.feature_names = [
            'year', 'month', 'quarter', 'lag_inflation_1',
            'rolling_avg_3', 'monthly_change'
        ]
    
    def preprocess_data(self, df):
        """Preprocess food price data"""
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean country names
        if 'country' in df.columns:
            df['country'] = df['country'].str.strip()
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Inflation', 'country', 'date'])
        
        # Create time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Create lag features
        df['lag_inflation_1'] = df.groupby('country')['Inflation'].shift(1)
        
        # Create rolling average
        df['rolling_avg_3'] = df.groupby('country')['Inflation'].rolling(
            window=3, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Create monthly change
        df['monthly_change'] = df['Inflation'] - df['lag_inflation_1']
        
        # Remove rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def train(self, df, target_col='Inflation'):
        """Train the food inflation model"""
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost model with optimized parameters
        best_params = {
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'lambda': 0.8396091364711157,
            'alpha': 1.0698943990722283,
            'colsample_bytree': 0.8298188609033732,
            'subsample': 0.704028238502165,
            'learning_rate': 0.2855054211647671,
            'n_estimators': 685,
            'max_depth': 14,
            'min_child_weight': 4
        }
        
        self.model = xgb.XGBRegressor(**best_params)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        metrics, y_pred = self.evaluate(X_test, y_test)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': metrics
        }
    
    def predict_single(self, month, quarter, lag_inflation, rolling_avg, monthly_change, year=2024):
        """Make a single prediction with user inputs"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = np.array([[
            year, month, quarter, lag_inflation, rolling_avg, monthly_change
        ]])
        
        prediction = self.model.predict(features)
        return prediction[0]

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    # Generate sample overall inflation data
    np.random.seed(42)
    n_samples = 1000
    
    overall_data = {
        'country': np.random.choice(['USA', 'Germany', 'Brazil', 'India', 'China'], n_samples),
        'year': np.random.choice(range(2010, 2024), n_samples),
        'cpi': np.random.normal(3, 2, n_samples),
        'gdp': np.random.lognormal(10, 1, n_samples),
        'gdp_per_capita': np.random.normal(35000, 15000, n_samples),
        'unemployment_rate': np.random.exponential(5, n_samples),
        'real_interest_rate': np.random.normal(2, 3, n_samples),
        'gdp_deflator': np.random.normal(2.5, 1.5, n_samples),
        'gdp_growth': np.random.normal(2.5, 2, n_samples),
        'current_account_balance': np.random.normal(0, 5, n_samples),
        'gov_expense': np.random.normal(25, 8, n_samples),
        'gov_revenue': np.random.normal(22, 7, n_samples),
        'tax_revenue': np.random.normal(18, 5, n_samples),
        'gni': np.random.lognormal(9.8, 1, n_samples),
        'public_debt': np.random.exponential(60, n_samples)
    }
    
    overall_df = pd.DataFrame(overall_data)
    
    # Generate sample food inflation data
    dates = pd.date_range('2018-01-01', periods=500, freq='M')
    countries = np.random.choice(['USA', 'Germany', 'Brazil', 'India', 'China'], 500)
    
    food_data = {
        'date': dates,
        'country': countries,
        'Inflation': np.random.normal(4, 3, 500)
    }
    
    food_df = pd.DataFrame(food_data)
    
    return overall_df, food_df

def create_feature_importance_plot(model, feature_names):
    """Create feature importance visualization"""
    try:
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    except AttributeError:
        return None

def format_metrics(metrics):
    """Format metrics for display"""
    return {
        'RMSE': f"{metrics['rmse']:.4f}",
        'MSE': f"{metrics['mse']:.4f}",
        'MAE': f"{metrics['mae']:.4f}",
        'R² Score': f"{metrics['r2']:.4f}"
    }

def save_model(model, filename):
    """Save trained model to file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False

def load_model(filename):
    """Load trained model from file"""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_prediction_confidence_interval(prediction, model_type="overall"):
    """Generate confidence interval for predictions"""
    if model_type == "overall":
        # Based on RMSE of 7.4563
        margin = 7.4563 * 1.96  # 95% confidence interval
    else:  # food
        # Based on RMSE of 3.3386
        margin = 3.3386 * 1.96  # 95% confidence interval
    
    lower = prediction - margin
    upper = prediction + margin
    
    return lower, upper

def interpret_inflation_prediction(prediction, model_type="overall"):
    """Provide interpretation for inflation predictions"""
    if model_type == "overall":
        if prediction < 0:
            return "🔽 Deflation - Prices are decreasing"
        elif prediction < 2:
            return "📉 Low inflation - Below target range"
        elif prediction <= 4:
            return "✅ Moderate inflation - Healthy range"
        elif prediction <= 8:
            return "📈 High inflation - Above comfort zone"
        else:
            return "🚨 Very high inflation - Economic concern"
    else:  # food
        if prediction < 0:
            return "🔽 Food deflation - Food prices declining"
        elif prediction <= 3:
            return "✅ Normal food inflation - Stable prices"
        elif prediction <= 8:
            return "📈 Elevated food inflation - Monitor closely"
        else:
            return "🚨 Food crisis potential - Significant price increases"
