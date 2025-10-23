import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time

# Fetch data from Yahoo Finance
def fetch_yahoo_data(ticker, period="5y"):
    for attempt in range(3):  # Retry up to 3 times
        try:
            print(f"Fetching data from Yahoo Finance for {ticker} (Attempt {attempt + 1}/3)...")
            data = yf.download(ticker, period=period)
            if not data.empty:
                # If columns are multi-indexed, flatten them
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = ['_'.join(col).strip() for col in data.columns.values]
                return data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retrying
    print("Failed to fetch data from Yahoo Finance after 3 attempts.")
    return pd.DataFrame()

# Add technical indicators: Moving Average and RSI
def add_technical_indicators(data, ticker):
    # Dynamically use the correct column name
    close_column = f"Close_{ticker}"
    data['MA50'] = data[close_column].rolling(window=50).mean()
    
    # Add 200-day moving average
    data['MA200'] = data[close_column].rolling(window=200).mean()

    # Add Relative Strength Index (RSI)
    delta = data[close_column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

# Prepare the dataset
def prepare_data(data, ticker):
    print(f"Data columns: {data.columns}")
    print(data.head())

    # Add technical indicators
    data = add_technical_indicators(data, ticker)
    
    # Drop rows with NaN values
    data = data.dropna()

    # Define features (X) and target (y)
    close_column = f"Close_{ticker}"
    feature_columns = [close_column, 'MA50', 'MA200', 'RSI']
    X = data[feature_columns].values
    y = data[close_column].shift(-1).dropna().values  # Use the next day's closing price as the prediction target
    X = X[:-1]  # Ensure that X and y have the same length
    
    predict_input = data[feature_columns].iloc[-1:].values
    return X, y, predict_input

# Train model and predict
def train_and_predict(X, y, predict_input, days=30):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse:.2f}")

    # Sequentially predict for the next `days` days
    predictions = []
    current_input = scaler.transform(predict_input)  # Scale the initial input
    for _ in range(days):
        next_prediction = model.predict(current_input)[0]
        predictions.append(next_prediction)
        # Update input with the predicted value for the next day
        current_input = np.array([[next_prediction, next_prediction, next_prediction, next_prediction]])

    return predictions, model

# Visualize the results with Candlestick chart and predicted prices
def visualize(data, predictions, ticker):
    # Plotting the candlestick chart using Plotly
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data[f'Open_{ticker}'],
        high=data[f'High_{ticker}'],
        low=data[f'Low_{ticker}'],
        close=data[f'Close_{ticker}'],
        name="Candlesticks"
    ))

    # Add moving averages
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['MA50'], 
        mode='lines', 
        name='50-Day MA', 
        line={'color': 'orange'}
    ))
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['MA200'], 
        mode='lines', 
        name='200-Day MA', 
        line={'color': 'blue'}
    ))

    # Plot predictions for the next 30 days
    future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=predictions, 
        mode='lines', 
        name='Predicted Prices', 
        line={'color': 'green'}
    ))

    # Customize layout
    fig.update_layout(
        title=f"Stock Price Prediction for {ticker} with Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    # Show the chart
    fig.show()

# Main function
def main():
    ticker = input("Enter stock ticker (e.g., AAPL for Apple): ").upper()
    data = fetch_yahoo_data(ticker)

    if data.empty:
        print("No data found for the ticker. Exiting.")
        return

    X, y, predict_input = prepare_data(data, ticker)

    if X is None or y is None:
        print("Failed to prepare data. Exiting.")
        return

    predictions, model = train_and_predict(X, y, predict_input, days=30)

    print("Predictions for the next 30 days starting from tomorrow:")
    print(predictions)

    visualize(data, predictions, ticker)

if __name__ == "__main__":
    main()
