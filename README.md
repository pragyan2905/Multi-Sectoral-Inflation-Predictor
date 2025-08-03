
# Multi-sectoral Inflation Predictor 📈

## Description

This project provides a framework for predicting inflation rates across different economic sectors. It utilizes historical data and various economic indicators to generate forecasts for both overall (CPI) inflation and sector-specific inflation, such as food prices. The objective is to provide a quantitative tool for economic analysis and forecasting. 📊

## Features

The project currently includes the following predictive models:

1.  **Overall Inflation Predictor**: Forecasts the general Consumer Price Index (CPI) inflation rate.
2.  **Food Inflation Predictor**: Forecasts the inflation rate for food products.

---

### 1. Overall Inflation Predictor

This model predicts the overall inflation rate using a set of economic indicators sourced from the World Bank.

#### Data
The model is trained on the `world_bank_data_2025.csv` dataset, which includes the following key features:
* GDP (Current USD)
* GDP per Capita (Current USD)
* Unemployment Rate (%)
* Real Interest Rate (%)
* Public Debt (% of GDP)
* Government Expense & Revenue (% of GDP)
* Gross National Income (USD)

#### Methodology
The process involves the following stages:
1.  **Data Preprocessing**: Cleaning the raw data by handling missing values via median imputation and standardizing column names.
2.  **Exploratory Data Analysis (EDA)**: Using visualizations such as time-series plots, correlation heatmaps, and histograms to analyze relationships between economic indicators and inflation.
3.  **Feature Engineering**: Generating new features, including the debt-to-GDP ratio and government balance, to serve as additional predictive inputs.
4.  **Model Training**: Training and evaluating several regression models, including Linear Regression, Random Forest, and XGBoost.
5.  **Hyperparameter Tuning**: Optimizing the XGBoost model's hyperparameters with Optuna to minimize the Root Mean Squared Error (RMSE).

#### Results 🎯
The final tuned XGBoost model achieved the following metrics on the test set:
* **RMSE:** 7.4563
* **R² Score:** 0.7521

---

### 2. Food Inflation Predictor

This model is designed to predict the inflation rate specifically for food products, likely using a combination of general economic indicators and food-specific data.

#### Results 🎯
The model's performance metrics on the test set are as follows:
* **RMSE:** 3.3386
* **MSE:** 11.1464
* **MAE:** 0.7688
* **R² Score:** 0.9876

---

## Technologies Used 💻

* **Python**
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.
* **Scikit-learn**: For building and evaluating machine learning models.
* **XGBoost**: For implementing the gradient boosting model.
* **Optuna**: For hyperparameter optimization.
* **Matplotlib** & **Seaborn**: For data visualization.
* **Jupyter Notebook**: For development and experimentation.

## Installation and Usage

To get started with this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/multi-sectoral-inflation-predictor.git](https://github.com/your-username/multi-sectoral-inflation-predictor.git)
    cd multi-sectoral-inflation-predictor
    ```

2.  **Run the prediction scripts:**
    * To run the overall inflation predictor, execute the Python script:
        ```bash
        python overall_inflation_predictor.py
        ```
    * Alternatively, you can explore the analysis and models in the Jupyter Notebook:
        ```bash
        jupyter notebook overall_inflation_predictor.ipynb
        ```

## Future Work 🚀

* Develop and integrate predictors for other sectors (e.g., energy, housing, healthcare).
* Incorporate additional data sources, such as market sentiment data.
* Deploy the models via a web application or API.

## Contributing

Contributions are welcome. Please feel free to submit a pull request or open an issue to discuss potential changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
