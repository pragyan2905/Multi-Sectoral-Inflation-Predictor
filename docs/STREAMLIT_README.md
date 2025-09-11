# 🚀 Multi-Sectoral Inflation Predictor - Streamlit Web App

## 📋 Overview

This interactive Streamlit web application showcases advanced machine learning models for predicting inflation rates across different economic sectors. The app provides an intuitive interface to explore inflation prediction models, visualize data insights, and make real-time predictions.

## ✨ Features

### 🏠 Homepage
- **Project Overview**: Complete introduction to the inflation prediction models
- **Model Comparison**: Side-by-side performance metrics
- **Interactive Navigation**: Easy access to all features

### 📊 Overall Inflation Predictor
- **Model Details**: XGBoost with 75.2% accuracy (R² = 0.7521)
- **Economic Indicators**: Uses 15+ economic features from World Bank data
- **Visualizations**: Correlation heatmaps, feature distributions, performance charts
- **Methodology**: Step-by-step explanation of the modeling process

### 🍎 Food Inflation Predictor
- **High Accuracy**: 98.8% accuracy (R² = 0.9876) for food price predictions
- **Time-Series Features**: Lag variables, rolling averages, seasonal patterns
- **Feature Importance**: Visual breakdown of most influential factors
- **Real-time Data**: Based on WLD_RTFP food price dataset

### 🔮 Interactive Predictions
- **Overall Inflation**: Input economic indicators for CPI inflation forecasts
- **Food Inflation**: Enter food market data for specific predictions
- **Confidence Intervals**: 95% confidence bounds for all predictions
- **Smart Interpretations**: Automatic analysis of prediction results

### 📈 Data Insights & Analytics
- **Global Trends**: Regional inflation patterns over time
- **Economic Impact**: How different indicators affect inflation
- **Correlation Analysis**: Detailed relationships between variables
- **Time Series Patterns**: Seasonal trends and decomposition analysis

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/Multi-Sectoral-Inflation-Predictor.git
cd Multi-Sectoral-Inflation-Predictor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App
```bash
streamlit run app.py
```

### Step 4: Access the Application
The app will automatically open in your browser at `http://localhost:8501`

## 📦 Dependencies

```
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
optuna>=3.0.0
streamlit>=1.28.0
plotly>=5.15.0
```

## 🎯 Usage Guide

### Making Predictions

#### Overall Inflation Prediction
1. Navigate to "📊 Overall Inflation Predictor"
2. Go to "🔮 Make Predictions" page
3. Select "📊 Overall Inflation"
4. Input economic indicators:
   - GDP (Billion USD)
   - GDP per Capita (USD)
   - Unemployment Rate (%)
   - Real Interest Rate (%)
   - Public Debt (% of GDP)
   - Government Expense/Revenue (% of GDP)
   - Gross National Income (Billion USD)
5. Click "🔮 Predict Overall Inflation"
6. View prediction with confidence interval and interpretation

#### Food Inflation Prediction
1. Navigate to "🔮 Make Predictions"
2. Select "🍎 Food Inflation"
3. Input food market indicators:
   - Country selection
   - Month and Quarter
   - Previous Month Inflation (%)
   - 3-Month Rolling Average (%)
   - Monthly Change (%)
4. Click "🔮 Predict Food Inflation"
5. Analyze results with confidence bounds

### Exploring Data Insights
1. Go to "📈 Data Insights"
2. Choose analysis type:
   - **🌍 Global Trends**: Regional inflation patterns
   - **📊 Economic Indicators**: Impact analysis
   - **🔗 Correlation Analysis**: Variable relationships
   - **📈 Time Series Patterns**: Seasonal trends

## 🏗️ Application Architecture

```
Multi-Sectoral-Inflation-Predictor/
├── app.py                          # Main Streamlit application
├── utils.py                        # Utility functions and model classes
├── requirements.txt                # Python dependencies
├── STREAMLIT_README.md            # This file
├── overall_inflation_predictor.py  # Original overall model script
├── food_price_inflation_predictor.ipynb  # Original food model notebook
└── README.md                      # Project overview
```

### Key Components

#### `app.py`
- Main Streamlit interface
- Page navigation and routing
- Interactive visualizations
- User input handling

#### `utils.py`
- `InflationPredictor`: Base class for prediction models
- `OverallInflationPredictor`: Economic indicators model
- `FoodInflationPredictor`: Time-series food model
- Data preprocessing functions
- Visualization utilities

## 🔬 Model Details

### Overall Inflation Model
- **Algorithm**: XGBoost Regressor
- **Features**: 15 economic indicators
- **Training Data**: World Bank economic data
- **Performance**: RMSE 7.46, R² 0.752
- **Optimization**: Hyperparameter tuning with Optuna

### Food Inflation Model
- **Algorithm**: XGBoost Regressor
- **Features**: 6 time-series features
- **Training Data**: WLD_RTFP food price data
- **Performance**: RMSE 3.34, R² 0.988
- **Optimization**: 50 trials with Optuna

## 📊 Visualization Features

### Interactive Charts
- **Plotly Integration**: Responsive, interactive visualizations
- **Time Series Plots**: Trend analysis with zoom/pan
- **Correlation Heatmaps**: Clickable variable relationships
- **Feature Importance**: Ranked predictor importance
- **Geographic Patterns**: Regional inflation mapping

### Customization Options
- **Color Schemes**: Professional color palettes
- **Responsive Design**: Mobile-friendly layouts
- **Export Options**: Download charts as images
- **Data Tables**: Sortable, filterable data views

## 🚀 Deployment Options

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Deploy with one click

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port $PORT" > Procfile

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

#### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

## 🔧 Customization Guide

### Adding New Models
1. Create new model class in `utils.py`:
```python
class NewInflationPredictor(InflationPredictor):
    def __init__(self):
        super().__init__("new_model")
        # Initialize model-specific attributes
    
    def preprocess_data(self, df):
        # Implement data preprocessing
        pass
    
    def train(self, df, target_col):
        # Implement model training
        pass
```

2. Add new page in `app.py`:
```python
elif page == "🆕 New Model":
    st.markdown("### New Model Interface")
    # Implement UI for new model
```

### Styling Customization
- Modify CSS in `app.py` for custom styling
- Update color schemes in Plotly visualizations
- Customize metrics display cards

### Data Source Integration
- Update data loading functions in `utils.py`
- Add new data preprocessing pipelines
- Integrate with external APIs

## 📈 Performance Optimization

### Caching Strategy
- `@st.cache_data` for data loading
- Model caching for faster predictions
- Visualization caching for responsive UI

### Memory Management
- Efficient data structures
- Lazy loading for large datasets
- Garbage collection optimization

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Update dependencies
pip install --upgrade -r requirements.txt
```

#### Port Conflicts
```bash
# Use different port
streamlit run app.py --server.port 8502
```

#### Memory Issues
- Reduce dataset size for local development
- Use data sampling for large files
- Clear Streamlit cache: `st.cache_data.clear()`

### Debug Mode
```bash
streamlit run app.py --logger.level debug
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test locally
4. Submit pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include type hints where applicable
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 📞 Support

- **Issues**: GitHub Issues page
- **Documentation**: This README and code comments
- **Email**: your-email@domain.com

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Additional economic sectors
- [ ] Advanced forecasting models
- [ ] User authentication system
- [ ] Data export functionality
- [ ] Mobile app version
- [ ] API endpoints for predictions

---

**Built with ❤️ using Streamlit, XGBoost, and Python**
