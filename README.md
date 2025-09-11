# 🌐 Multi-Sectoral Inflation Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Overview

The **Multi-Sectoral Inflation Predictor** is an advanced machine learning solution that forecasts inflation rates across different economic sectors. This project features two specialized prediction models and an interactive web application, providing accurate inflation forecasts for economic planning and analysis.

### 🎯 Key Highlights
- **Two Specialized Models**: Overall CPI inflation and food-specific inflation predictors
- **High Accuracy**: Food model achieves 98.8% accuracy (R² = 0.9876)
- **Interactive Dashboard**: Beautiful Streamlit web application with real-time predictions
- **Data-Driven Insights**: Comprehensive visualizations and analytics
- **Production Ready**: Optimized models with hyperparameter tuning via Optuna


### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/Multi-Sectoral-Inflation-Predictor.git
cd Multi-Sectoral-Inflation-Predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```



## ✨ Features

### 📈 Prediction Models

#### 1. Overall Inflation Predictor
- **Purpose**: Predicts Consumer Price Index (CPI) inflation rates
- **Data Source**: World Bank economic indicators
- **Algorithm**: XGBoost Regressor with Optuna optimization

- **Features**: GDP, unemployment rate, public debt, interest rates, government finances

#### 2. Food Inflation Predictor
- **Purpose**: Specialized predictions for food price inflation
- **Data Source**: WLD_RTFP real-time food price data
- **Algorithm**: Time-series optimized XGBoost

- **Features**: Lag variables, rolling averages, seasonal patterns

### 🎨 Interactive Web Application

The Streamlit application provides:

- **🏠 Home Dashboard**: Project overview and model comparisons
- **📊 Model Details**: In-depth exploration of each prediction model
- **🔮 Live Predictions**: Interactive interface for real-time forecasting
- **📈 Data Analytics**: Comprehensive visualizations and insights
- **🌍 Global Trends**: Regional inflation pattern analysis

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/Multi-Sectoral-Inflation-Predictor.git
cd Multi-Sectoral-Inflation-Predictor
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Application**
```bash
streamlit run app/app.py
```

## 📁 Project Structure

```
Multi-Sectoral-Inflation-Predictor/
│
├── app/
│   ├── app.py                          # Main Streamlit application
│   ├── utils.py                        # Utility functions and model classes
│   ├── overall_inflation_predictor.py  # Overall inflation model
│   └── food_price_inflation_predictor.ipynb  # Food inflation notebook
│
├── data/
│   ├── world_bank_data_2025.csv       # Economic indicators data
│   └── WLD_RTFP_country_2023-10-02.csv # Food price data
│
├── models/
│   ├── overall_inflation_model.pkl     # Trained overall model
│   └── food_inflation_model.pkl        # Trained food model
│
├── demos/
│   ├── run_demo.py                     # Full demo script
│   └── simple_demo.py                  # Quick demo script
│
├── docs/
│   ├── README.md                       # Project documentation
│   ├── QUICKSTART.md                   # Quick start guide
│   └── STREAMLIT_README.md            # Streamlit app documentation
│
├── notebooks/
│   ├── overall_inflation_predictor.ipynb
│   └── food_price_inflation_predictor.ipynb
│
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## 💻 Usage

### Running the Web Application

```bash
streamlit run app/app.py
```

Navigate through the application tabs:
1. **Home** - Overview and introduction
2. **Overall Inflation Predictor** - Economic model details
3. **Food Inflation Predictor** - Food price model information
4. **Make Predictions** - Interactive prediction interface
5. **Data Insights** - Analytics and visualizations

### Making Predictions


### Running Demo Scripts

```bash
# Quick demo without full model training
python demos/simple_demo.py

# Full demo with model training
python demos/run_demo.py
```

## 📊 Model Performance

### Overall Inflation Model
| Metric | Value |
|--------|-------|
| R² Score | 0.7521 |
| RMSE | 7.4563 |
| Algorithm | XGBoost |
| Features | 15 economic indicators |
| Optimization | 150 Optuna trials |

### Food Inflation Model
| Metric | Value |
|--------|-------|
| R² Score | 0.9876 |
| RMSE | 3.3386 |
| MAE | 0.77 |
| MSE | 11.15 |
| Algorithm | XGBoost |
| Features | 6 time-series features |

## 🔧 Technologies Used

### Machine Learning
- **XGBoost**: Gradient boosting framework
- **Scikit-learn**: ML utilities and preprocessing
- **Optuna**: Hyperparameter optimization
- **Pandas & NumPy**: Data manipulation

### Visualization
- **Streamlit**: Interactive web application
- **Plotly**: Interactive charts
- **Matplotlib & Seaborn**: Statistical visualizations

### Data Sources
- **World Bank**: Economic indicators
- **WLD_RTFP**: Real-time food prices

## 📈 Key Features Explained

### Economic Indicators (Overall Model)
- GDP and GDP per capita
- Unemployment rate
- Real interest rate
- Public debt (% of GDP)
- Government expense and revenue
- Tax revenue
- Gross national income

### Time-Series Features (Food Model)
- Lagged inflation (previous month)
- 3-month rolling average
- Monthly change rate
- Seasonal indicators (month, quarter)

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app/app.py --server.port 8501
```

### Cloud Deployment Options

#### Streamlit Cloud
1. Push to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click



## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🔮 Future Enhancements

- [ ] Additional sector models (energy, housing, healthcare)
- [ ] Real-time data integration
- [ ] API endpoints for programmatic access
- [ ] Mobile application
- [ ] Advanced forecasting with LSTM/Prophet
- [ ] Multi-language support
- [ ] User authentication and saved predictions







## 🙏 Acknowledgments

- World Bank for economic data
- Streamlit team for the amazing framework
- XGBoost developers for the powerful ML library
- Open source community for continuous support


---

<div align="center">
  
**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ using Python, Streamlit, and XGBoost

</div>
