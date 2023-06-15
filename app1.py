import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64
from datetime import datetime, timedelta

st.set_option('deprecation.showPyplotGlobalUse', False)

# Set the base URL of the FastAPI server
BASE_URL = "http://localhost:8000"

# Load the logo image
logo_image = "Logo.png"

# Create the layout
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo_image, width=100)
with col2:
    st.title("Stock Price Prediction")
   


# Stock selection
stock_name = st.selectbox("Select a stock", ("Reliance", "Zomato", "Hdfc", "Tcs"))

# Calculate the date range
today = datetime.now().date()
min_date = today - timedelta(days=3)
max_date = today + timedelta(days=3)

# Date range selection
start_date = st.date_input("Select start date", min_value=min_date, max_value=max_date)
end_date = st.date_input("Select end date", min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.error("Error: The start date must be before or the same as the end date.")
else:
        if st.button("Predict"):
            # Prepare the request payload
            payload = {"stock_name": stock_name, "start_date": str(start_date), "end_date": str(end_date)}

            # Send the request to the server
            response = requests.post(f"{BASE_URL}/LSTM_Predict", json=payload)

            if response.status_code == 200:
                # Display predicted prices
                data = response.json()
                if "future_prediction" in data:
                    predict_prices = data["future_prediction"]
                    future_dates = data["future_dates"]
                    column_names = data["column_names"]

                    st.write(f"Predicted prices for {stock_name} between {start_date} and {end_date}:")
                    table_data = [["Date"] + column_names]  # Include "Date" as the first element in the first row
                    for date, prices in zip(future_dates, predict_prices):
                        row = [date] + prices
                        table_data.append(row)

                    # Create a DataFrame and remove the column index label
                    df = pd.DataFrame(table_data[1:], columns=table_data[0]).rename_axis(None, axis=1)

                    st.table(df.style.set_table_attributes("style='width:auto;'"))

                    csv_data = df.to_csv(index=True)
                    b64 = base64.b64encode(csv_data.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_prices.csv">Download CSV</a>'

                    st.markdown(href, unsafe_allow_html=True)

                    # Create line plot chart
                    plt.figure(figsize=(10, 6))
                    for i, column_name in enumerate(column_names):
                        plt.plot(future_dates, [price[i] for price in predict_prices], label=column_name)
                    plt.xlabel("Date")
                    plt.ylabel("Price")
                    plt.title(f"Stock Price Trend for {stock_name}")
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot()

                    # Create scatter plot chart
                    plt.figure(figsize=(10, 6))
                    for i, column_name in enumerate(column_names):
                        plt.scatter(future_dates, [price[i] for price in predict_prices], label=column_name)
                    plt.xlabel("Date")
                    plt.ylabel("Price")
                    plt.title(f"Stock Price Scatter Plot for {stock_name}")
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot()

                else:
                    st.error("Error: Failed to retrieve predicted prices.")
            else:
                st.error("Error: Market is closed today.")