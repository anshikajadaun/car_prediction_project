import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Page settings
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# Title
st.title("🚗 Car Price Prediction")
st.write("Enter car details to predict the price.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("car_prediction_project.csv")


try:
    df = load_data()
except Exception:
    st.error("Dataset load nahi hui. CSV filename check karo.")
    st.stop()


# Required columns
required_columns = ["Engine Size", "Year", "Mileage", "Price"]

if not all(col in df.columns for col in required_columns):
    st.error(
        "Dataset mein Engine Size, Year, Mileage aur Price "
        "columns hone chahiye."
    )
    st.stop()


# Prepare data
data = df[required_columns].copy()

for col in required_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

if len(data) < 5:
    st.error("Model train karne ke liye sufficient data nahi hai.")
    st.stop()


# Features and target
X = data[["Engine Size", "Year", "Mileage"]]
y = data["Price"]


# Train model
model = LinearRegression()
model.fit(X, y)


# User input
st.subheader("Enter Car Details")

engine_size = st.number_input(
    "Engine Size",
    min_value=0.1,
    max_value=20.0,
    value=2.3,
    step=0.1
)

year = st.number_input(
    "Manufacturing Year",
    min_value=1900,
    max_value=2100,
    value=2022,
    step=1
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=2000000,
    value=114832,
    step=1000
)


# Predict button
if st.button("Predict Car Price", type="primary",
             use_container_width=True):

    new_car = pd.DataFrame({
        "Engine Size": [engine_size],
        "Year": [year],
        "Mileage": [mileage]
    })

    predicted_price = model.predict(new_car)[0]

    st.success("Prediction completed!")

    st.metric(
        label="Predicted Car Price",
        value=f"{predicted_price:,.2f}"
    )
