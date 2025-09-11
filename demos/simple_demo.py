#!/usr/bin/env python3
"""
Simple demo script for the Multi-Sectoral Inflation Predictor Streamlit app
This demo shows the app functionality without complex model training
"""

import numpy as np
import pandas as pd

def check_basic_dependencies():
    """Check if basic dependencies are installed"""
    print("🔍 Checking Basic Dependencies...")
    print("-" * 50)
    
    basic_packages = ['pandas', 'numpy', 'matplotlib', 'streamlit', 'plotly']
    missing_packages = []
    
    for package in basic_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Missing!")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All basic dependencies are installed!")
        return True

def simulate_predictions():
    """Simulate inflation predictions for demonstration"""
    print("\n🔮 Simulating Inflation Predictions...")
    print("-" * 50)
    
    # Overall inflation simulation
    print("📊 Overall Inflation Scenarios:")
    scenarios = [
        {"name": "Low Growth Economy", "gdp": 800, "unemployment": 8.0, "debt": 90},
        {"name": "Healthy Economy", "gdp": 1200, "unemployment": 4.0, "debt": 50},
        {"name": "High Growth Economy", "gdp": 1500, "unemployment": 3.0, "debt": 40}
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        # Simple simulation formula
        predicted_inflation = 2.5 + (scenario['unemployment'] * 0.3) + (scenario['debt'] * 0.02) - (scenario['gdp'] * 0.001)
        predicted_inflation = max(0, predicted_inflation)
        
        print(f"   {i}. {scenario['name']}:")
        print(f"      GDP: ${scenario['gdp']}B, Unemployment: {scenario['unemployment']}%, Debt: {scenario['debt']}%")
        print(f"      → Predicted Inflation: {predicted_inflation:.2f}%")
    
    # Food inflation simulation
    print("\n🍎 Food Inflation Scenarios:")
    food_scenarios = [
        {"name": "Normal Period", "lag": 2.5, "change": 0.1},
        {"name": "Supply Disruption", "lag": 8.2, "change": 2.5},
        {"name": "Deflationary Period", "lag": -1.0, "change": -0.8}
    ]
    
    for i, scenario in enumerate(food_scenarios, 1):
        # Simple simulation formula
        predicted_food_inflation = scenario['lag'] + scenario['change'] * 2
        predicted_food_inflation = max(-10, min(50, predicted_food_inflation))
        
        print(f"   {i}. {scenario['name']}:")
        print(f"      Last Month: {scenario['lag']}%, Monthly Change: {scenario['change']}%")
        print(f"      → Predicted Food Inflation: {predicted_food_inflation:.2f}%")

def generate_sample_data():
    """Generate sample data for visualization"""
    print("\n📊 Generating Sample Data...")
    print("-" * 50)
    
    # Create sample economic data
    countries = ['USA', 'Germany', 'Brazil', 'India', 'China']
    years = list(range(2020, 2024))
    
    data = []
    for country in countries:
        for year in years:
            row = {
                'country': country,
                'year': year,
                'cpi': np.random.normal(3, 2),
                'gdp': np.random.lognormal(10, 0.5),
                'unemployment': np.random.exponential(5),
                'debt': np.random.normal(60, 20)
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    print(f"   Generated {len(df)} rows of economic data")
    print(f"   Countries: {', '.join(countries)}")
    print(f"   Years: {min(years)}-{max(years)}")
    
    # Show sample statistics
    print("\n📈 Sample Statistics:")
    print(f"   Average CPI Inflation: {df['cpi'].mean():.2f}%")
    print(f"   Average Unemployment: {df['unemployment'].mean():.2f}%")
    print(f"   Average Public Debt: {df['debt'].mean():.2f}% of GDP")
    
    return df

def show_streamlit_instructions():
    """Show instructions for running the Streamlit app"""
    print("\n🚀 Streamlit App Instructions")
    print("=" * 60)
    print("To run the Streamlit web application:")
    print()
    print("1. Make sure you're in the virtual environment:")
    print("   source venv/bin/activate")
    print()
    print("2. Start the Streamlit app:")
    print("   streamlit run app.py")
    print()
    print("3. The app will open in your browser at:")
    print("   http://localhost:8501")
    print()
    print("4. Navigate through different pages:")
    print("   🏠 Home - Project overview")
    print("   📊 Overall Inflation Predictor - Economic model details")
    print("   🍎 Food Inflation Predictor - Food price model details")
    print("   🔮 Make Predictions - Interactive prediction interface")
    print("   📈 Data Insights - Visualizations and analytics")
    print()
    print("📖 For detailed documentation, see STREAMLIT_README.md")

def main():
    """Main demo function"""
    print("🚀 Multi-Sectoral Inflation Predictor - Simple Demo")
    print("=" * 60)
    
    try:
        # Check dependencies
        if not check_basic_dependencies():
            print("\n⚠️  Please install missing dependencies:")
            print("   pip install -r requirements.txt")
            return
        
        # Simulate predictions
        simulate_predictions()
        
        # Generate sample data
        sample_df = generate_sample_data()
        
        # Show Streamlit instructions
        show_streamlit_instructions()
        
        print("\n🎉 Simple Demo Complete!")
        print("=" * 60)
        print("✨ The Streamlit app includes:")
        print("   • Interactive web interface")
        print("   • Real-time predictions")
        print("   • Beautiful visualizations")
        print("   • Comprehensive analytics")
        print("   • Model performance metrics")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("   This is a simplified demo - check the full app for complete functionality")

if __name__ == "__main__":
    main()
