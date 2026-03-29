import streamlit as st
import numpy as np
import pickle

# --- Load model ---
@st.cache_resource
def load_model():
    with open("xgb.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- Page config ---
st.set_page_config(page_title="Delivery Time Predictor", layout="centered")

st.title("🚚 Delivery Time Prediction")
st.write("Predict **Time_taken (minutes)** using delivery and order details")

# --- Inputs ---
age = st.number_input("Delivery_person_Age", 18, 60, 25)
rating = st.number_input("Delivery_person_Ratings", 1.0, 5.0, 4.2)
distance = st.number_input("distance_km", 0.0, 50.0, 5.0)

min_time = st.number_input("min_cooking_time", 0.0, 120.0, 10.0)
median_time = st.number_input("median_cooking_time", 0.0, 120.0, 20.0)
mean_time = st.number_input("mean_cooking_time", 0.0, 120.0, 25.0)

num_orders = st.number_input("num_orders", 0, 20, 3)

# --- Categorical (converted to one-hot) ---
vehicle = st.selectbox(
    "Type_of_vehicle",
    ["electric_scooter", "motorcycle", "scooter"]
)

order = st.selectbox(
    "Type_of_order",
    ["Drinks", "Meal", "Snack"]
)

# --- One-hot encoding (IMPORTANT: match training columns exactly) ---
Type_of_vehicle_electric_scooter = 1 if vehicle == "electric_scooter" else 0
Type_of_vehicle_motorcycle = 1 if vehicle == "motorcycle" else 0
Type_of_vehicle_scooter = 1 if vehicle == "scooter" else 0

Type_of_order_Drinks = 1 if order == "Drinks" else 0
Type_of_order_Meal = 1 if order == "Meal" else 0
Type_of_order_Snack = 1 if order == "Snack" else 0

# --- Prediction ---
if st.button("Predict Time"):
    
    features = np.array([[
        age,
        rating,
        distance,
        min_time,
        median_time,
        mean_time,
        num_orders,
        Type_of_vehicle_electric_scooter,
        Type_of_vehicle_motorcycle,
        Type_of_vehicle_scooter,
        Type_of_order_Drinks,
        Type_of_order_Meal,
        Type_of_order_Snack
    ]])

    try:
        prediction = model.predict(features)
        st.success(f"⏱️ Estimated Time_taken: {round(prediction[0], 2)} minutes")
    except Exception as e:
        st.error(f"Prediction failed: {e}")