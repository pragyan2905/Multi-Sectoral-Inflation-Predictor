# Multi-Sectoral Inflation Predictor 📈

## Overview
The Multi-Sectoral Inflation Predictor project provides advanced machine learning models to forecast inflation rates across multiple sectors of the economy. It includes models for predicting overall Consumer Price Index (CPI) inflation as well as food-specific inflation rates. The project features an interactive Streamlit web application for visualization, analysis, and real-time predictions.

## Features

### 1. Overall Inflation Predictor
- Predicts general CPI inflation using economic indicators sourced from the World Bank.
- Utilizes an optimized XGBoost regression model with hyperparameter tuning.
- Key economic indicators include GDP, unemployment rate, public debt, government expenses, and more.
- Achieves an R² score of 0.7521 and RMSE of 7.4563 on test data.

### 2. Food Inflation Predictor
- Predicts inflation rates specifically for food products using time-series and feature engineering.
- Employs an optimized XGBoost model with high accuracy.
- Achieves an R² score of 0.9876 and RMSE of 3.3386 on test data.

## Technologies Used
- Python
- Pandas, NumPy for data manipulation
- Scikit-learn, XGBoost for machine learning
- Optuna for hyperparameter optimization
- Matplotlib, Seaborn, Plotly for visualization
- Streamlit for interactive web app
- Jupyter Notebook for experimentation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/multi-sectoral-inflation-predictor.git
cd multi-sectoral-inflation-predictor
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit Web Application
```bash
streamlit run app.py
```
This launches an interactive dashboard with tabs for:
- Home: Project overview and key features
- Overall Inflation Predictor: Model details and performance
- Food Inflation Predictor: Model details and performance
- Make Predictions: Input custom data for real-time inflation forecasts
- Data Insights: Visualizations and analytics on inflation trends and economic indicators

### Run Prediction Scripts
- Overall Inflation Predictor:
```bash
python overall_inflation_predictor.py
```
- Food Inflation Predictor:
```bash
jupyter notebook food_price_inflation_predictor.ipynb
```

### Simple Demo
To quickly test the app functionality without full model training, run:
```bash
python simple_demo.py
```

## Project Structure
- `app.py`: Streamlit web application source code
- `overall_inflation_predictor.py`: Script for overall inflation prediction model
- `food_price_inflation_predictor.ipynb`: Jupyter notebook for food inflation prediction
- `simple_demo.py`: Simple demo script showcasing app features
- `requirements.txt`: Python dependencies
- `utils.py`: Utility functions
- `run_demo.py`: Additional demo or testing scripts
- `QUICKSTART.md`, `STREAMLIT_README.md`: Additional documentation

## Future Work
- Extend prediction models to other sectors such as energy, housing, and healthcare.
- Incorporate additional data sources including market sentiment.
- Deploy models as a web API for broader accessibility.

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or new features.

## License
This project is licensed under the MIT License.


