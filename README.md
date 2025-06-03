# 🔌 Electricity Demand Analysis

An interactive machine learning project that analyzes electricity demand patterns in conjunction with weather conditions. The project uses clustering and time-series forecasting techniques and is deployed via a Streamlit web app.

---

## 🧾 Overview

This project explores how environmental factors influence electricity demand across cities. The goals were to:

- Analyze electricity demand in relation to weather variables (e.g., temperature, humidity).
- Group similar demand patterns using clustering.
- Forecast future electricity needs via regression modeling.
- Deliver insights through an interactive dashboard.

---

## 📊 Dataset and Objective

We used a merged dataset containing:

- Electricity demand
- Weather variables (temperature, humidity, etc.)
- Location and datetime

**Objectives:**

1. Understand the correlation between demand and environment.
2. Cluster similar demand profiles using KMeans.
3. Forecast future demand using historical data.

---

## 🧰 Methodology

### 1. Data Loading & Preprocessing

- Loaded and cleaned time-series data with weather and demand.
- Handled missing values using mean imputation.
- Normalized numeric features using `StandardScaler`.
- Extracted datetime components for analysis.

### 2. Feature Engineering

- Dynamically selected numeric features.
- Created lag features for supervised time-series modeling.

```python
for i in range(1, look_back + 1):
    df[f'demand_lag_{i}'] = df['demand'].shift(i)
```

### 3. Clustering Analysis

- Applied KMeans clustering on user-selected features.
- Visualized clusters in 2D using PCA and Plotly.

```python
model = KMeans(n_clusters=k)
cluster_labels = model.fit_predict(data_scaled)
```

### 4. Time-Series Forecasting

- Used lag features as input to a `LinearRegression` model.
- Evaluated using RMSE, MAE, and R² metrics.
- Visualized predictions vs. actuals.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### 5. Streamlit Web App

- Fully interactive dashboard for:
  - City selection
  - Feature and clustering controls
  - Forecasting configuration
- Built with Streamlit + Plotly for interactivity.

---

## 📈 Results

- Clustering revealed demand behavior patterns by city and weather.
- Time-series model achieved:
  - **RMSE**: 20.45  
  - **MAE**: 16.32  
  - **R² Score**: 0.82

---

## 💬 Discussion

### ✅ Strengths

- Modular & extendable codebase.
- User-friendly interactive dashboard.
- Clear and dynamic visualizations.

### ❗ Limitations

- Linear models underperform on complex demand peaks.
- Simple imputation might not capture real-world trends.
- No modeling of seasonality.

### 🚀 Potential Improvements

- Add LSTM or XGBoost for more accurate forecasting.
- Integrate weather forecast data.
- Use walk-forward validation instead of static splits.

---

## 🖥️ Run Locally

1. **Clone the repo:**
```bash
git clone https://github.com/tahahasan01/Electricity_Demand_Analysis.git
cd Electricity_Demand_Analysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Electricity_Demand_Analysis/
│
├── app.py                  # Streamlit app
├── notebook.ipynb          # Development notebook
├── data/                   # Input datasets
├── models/                 # (Optional) Saved models
├── README.md               # Project description
└── requirements.txt        # Dependencies
```

---

## 📜 License

This project is for academic and research purposes. Feel free to fork or contribute!

---

## 🙌 Acknowledgements

Developed by Syed Taha Hasan as part of a final year project in electricity load forecasting and smart grid analytics.
