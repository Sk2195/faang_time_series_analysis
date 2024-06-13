import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

# Function to fit ARIMA model and forecast
def fit_arima(series, order):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

# Function to perform differencing and return the differenced series
def difference_series(series):
    return series.diff().dropna()

# Function to invert differencing
def invert_difference(original_series, diff_series):
    forecast = diff_series.copy()
    forecast.iloc[0] = original_series.iloc[-1] + forecast.iloc[0]
    for i in range(1, len(forecast)):
        forecast.iloc[i] = forecast.iloc[i-1] + diff_series.iloc[i]
    return forecast

# Function to plot forecasts
def plot_forecast(actual, forecast, column_name):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual')
    plt.plot(forecast, label='Forecast', color='red')
    plt.title(f"Forecast vs Actual for {column_name}")
    plt.legend()
    st.pyplot(plt)

# Load the forecasts from the pickle file
pickle_file_path = r'C:\users\chimi\Desktop\Python Data Science Projects\stock_market_faang_analysis\task2-time_series_analysis\faang_forecasts.pkl'
with open(pickle_file_path, 'rb') as f:
    faang_forecasts = pickle.load(f)

# Load the original data from the CSV file
csv_file_path = r'C:\users\chimi\Desktop\Python Data Science Projects\stock_market_faang_analysis\task1dataextraction\faang_stock.csv'
faang_stock_df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)

# Print available columns to verify
st.write("Available columns:", faang_stock_df.columns)

# Streamlit UI
st.title('FAANG Stock Market Prediction')

# User selects stock
selected_stock = st.selectbox('Select Stock', faang_forecasts.keys())

# User selects the date range for forecasting
start_date = st.date_input('Start date', value=datetime(2023, 1, 1))
end_date = st.date_input('End date', value=datetime(2023, 12, 31))

if selected_stock and start_date and end_date:
    if selected_stock in faang_stock_df.columns:
        actual_series = faang_stock_df[selected_stock]
        forecast_period = (end_date - start_date).days
        forecast_series = faang_forecasts[selected_stock].head(forecast_period)

        st.subheader(f'Forecast for {selected_stock} from {start_date} to {end_date}')
        plot_forecast(actual_series[start_date:end_date], forecast_series, selected_stock)
        
        st.subheader('Forecast Data')
        st.write(forecast_series)
    else:
        st.error(f"Column '{selected_stock}' not found in the data.")

# Allow users to upload new data for prediction
st.subheader('Upload New Data for Prediction')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    st.write(new_data)

    selected_column = st.selectbox('Select Column for Prediction', new_data.columns)

    if st.button('Predict'):
        new_series = difference_series(new_data[selected_column])
        new_model_fit = fit_arima(new_series, (1, 1, 1))
        forecast_period = 30  # Assuming a fixed forecast period of 30 days
        new_forecast_diff = new_model_fit.forecast(steps=forecast_period)
        new_forecast = invert_difference(new_data[selected_column], new_forecast_diff)
        
        st.subheader(f'Forecast for {selected_column} for the next {forecast_period} days')
        plot_forecast(new_data[selected_column], new_forecast, selected_column)
        
        st.subheader('New Forecast Data')
        st.write(new_forecast)

