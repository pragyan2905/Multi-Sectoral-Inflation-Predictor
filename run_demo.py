#!/usr/bin/env python3
"""
Demo script to test the Multi-Sectoral Inflation Predictor Streamlit app
This script demonstrates the functionality of both prediction models
"""

import pandas as pd
import numpy as np
from utils import OverallInflationPredictor, FoodInflationPredictor, load_sample_data

def test_overall_inflation_model():
    """Test the overall inflation prediction model"""
    print("🏗️ Testing Overall Inflation Model...")
    print("-" * 50)
    
    # Create model instance
    model = OverallInflationPredictor()
    
    # Load sample data
    overall_df, _ = load_sample_data()
    
    # Train model
    print("📊 Training model with sample data...")
    results = model.train(overall_df)
    
    # Display metrics
    metrics = results['metrics']
    print(f"✅ Model Training Complete!")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    
    # Test single prediction
    print("\n🔮 Testing single prediction...")
    prediction = model.predict_single(
        gdp=1000,  # billion USD
        gdp_per_capita=35000,  # USD
        unemployment_rate=5.0,  # %
        real_interest_rate=2.0,  # %
        public_debt=60.0,  # % of GDP
        gov_expense=25.0,  # % of GDP
        gov_revenue=22.0,  # % of GDP
        gni=950  # billion USD
    )
    
    print(f"   Predicted Inflation: {prediction:.2f}%")
    
    return model, results

def test_food_inflation_model():
    """Test the food inflation prediction model"""
    print("\n🍎 Testing Food Inflation Model...")
    print("-" * 50)
    
    # Create model instance
    model = FoodInflationPredictor()
    
    # Load sample data
    _, food_df = load_sample_data()
    
    # Train model
    print("📊 Training model with sample data...")
    results = model.train(food_df)
    
    # Display metrics
    metrics = results['metrics']
    print(f"✅ Model Training Complete!")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    
    # Test single prediction
    print("\n🔮 Testing single prediction...")
    prediction = model.predict_single(
        month=6,  # June
        quarter=2,  # Q2
        lag_inflation=3.2,  # Previous month %
        rolling_avg=3.0,  # 3-month average %
        monthly_change=0.2,  # Monthly change %
        year=2024
    )
    
    print(f"   Predicted Food Inflation: {prediction:.2f}%")
    
    return model, results

def generate_sample_predictions():
    """Generate sample predictions for demonstration"""
    print("\n📈 Generating Sample Predictions...")
    print("-" * 50)
    
    # Overall inflation scenarios
    scenarios = [
        {"name": "Low Growth Economy", "gdp": 800, "unemployment": 8.0, "debt": 90},
        {"name": "Healthy Economy", "gdp": 1200, "unemployment": 4.0, "debt": 50},
        {"name": "High Growth Economy", "gdp": 1500, "unemployment": 3.0, "debt": 40}
    ]
    
    print("📊 Overall Inflation Scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"   {i}. {scenario['name']}:")
        print(f"      GDP: ${scenario['gdp']}B, Unemployment: {scenario['unemployment']}%, Debt: {scenario['debt']}%")
    
    # Food inflation scenarios
    food_scenarios = [
        {"name": "Normal Period", "lag": 2.5, "change": 0.1},
        {"name": "Supply Disruption", "lag": 8.2, "change": 2.5},
        {"name": "Deflationary Period", "lag": -1.0, "change": -0.8}
    ]
    
    print("\n🍎 Food Inflation Scenarios:")
    for i, scenario in enumerate(food_scenarios, 1):
        print(f"   {i}. {scenario['name']}:")
        print(f"      Last Month: {scenario['lag']}%, Monthly Change: {scenario['change']}%")

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking Dependencies...")
    print("-" * 50)
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scikit-learn', 'xgboost', 'streamlit', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Missing!")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def main():
    """Main demo function"""
    print("🚀 Multi-Sectoral Inflation Predictor Demo")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    try:
        # Test overall inflation model
        overall_model, overall_results = test_overall_inflation_model()
        
        # Test food inflation model
        food_model, food_results = test_food_inflation_model()
        
        # Generate sample predictions
        generate_sample_predictions()
        
        print("\n🎉 Demo Complete!")
        print("=" * 60)
        print("🌐 To run the Streamlit app:")
        print("   streamlit run app.py")
        print("\n📖 For more information, see STREAMLIT_README.md")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("   Check your data files and dependencies")

if __name__ == "__main__":
    main()
