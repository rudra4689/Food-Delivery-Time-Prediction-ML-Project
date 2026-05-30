"""
app.py — Hugging Face Spaces entry point.
Uses pickle to load artifacts. xgboost is explicitly imported
at the top so pickle.load() can reconstruct the model correctly.
"""

import os
import pickle
import xgboost          # ← must be imported before pickle.load() to avoid ModuleNotFoundError
import numpy as np
import pandas as pd
import gradio as gr


# ── Haversine distance ────────────────────────────────────────────────────────
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = (np.sin(dLat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


# ── Load / auto-train artifacts ───────────────────────────────────────────────
def load_artifacts():
    needed = ["xgb_model.pkl", "cooking_stats.pkl", "feature_columns.pkl"]

    if not all(os.path.exists(p) for p in needed):
        print("Artifact files not found — training from data.xlsx ...")
        if not os.path.exists("data.xlsx"):
            raise FileNotFoundError(
                "Neither model artifacts nor data.xlsx found. "
                "Upload xgb_model.pkl, cooking_stats.pkl, feature_columns.pkl "
                "(or data.xlsx) to your Space."
            )
        from train import preprocess_and_train
        preprocess_and_train()

    print("Loading pre-trained artifacts ...")
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("cooking_stats.pkl", "rb") as f:
        cooking_stats = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    print(f"Model ready. Features: {feature_columns}")
    return model, cooking_stats, feature_columns


model, cooking_stats, feature_columns = load_artifacts()

# Global fallback stats (average across all order types)
_fallback = {
    k: float(np.mean([v[k] for v in cooking_stats.values()]))
    for k in ["min_cooking_time", "median_cooking_time", "mean_cooking_time", "num_orders"]
}


# ── Prediction function ───────────────────────────────────────────────────────
def predict_delivery_time(
    age: int,
    ratings: float,
    rest_lat: float,
    rest_lon: float,
    del_lat: float,
    del_lon: float,
    order_type: str,
    vehicle_type: str,
) -> str:
    distance_km = haversine(rest_lat, rest_lon, del_lat, del_lon)
    stats = cooking_stats.get(order_type, _fallback)

    features = {
        "Delivery_person_Age":      age,
        "Delivery_person_Ratings":  ratings,
        "distance_km":              distance_km,
        "min_cooking_time":         stats["min_cooking_time"],
        "median_cooking_time":      stats["median_cooking_time"],
        "mean_cooking_time":        stats["mean_cooking_time"],
        "num_orders":               stats["num_orders"],
    }

    # One-hot dummies — baselines: bicycle (vehicle), Buffet (order)
    for v in ["electric_scooter", "motorcycle", "scooter"]:
        features[f"Type_of_vehicle_{v}"] = int(vehicle_type == v)
    for o in ["Drinks", "Meal", "Snack"]:
        features[f"Type_of_order_{o}"] = int(order_type == o)

    input_df = pd.DataFrame([features]).reindex(columns=feature_columns, fill_value=0)
    minutes = float(model.predict(input_df)[0])

    mins = int(round(minutes))
    hrs, rem = divmod(mins, 60)
    time_str = f"{hrs}h {rem}min" if hrs else f"{rem} min"

    return (
        f"## ⏱️ Estimated Delivery Time: {time_str}\n\n"
        f"| Detail | Value |\n"
        f"|---|---|\n"
        f"| 📍 Distance | {distance_km:.2f} km |\n"
        f"| 🛵 Vehicle | {vehicle_type.replace('_', ' ').title()} |\n"
        f"| 🍽️ Order Type | {order_type} |\n"
        f"| ⭐ Rider Rating | {ratings} |\n"
        f"| 🧑 Rider Age | {age} yrs |"
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="Food Delivery Time Predictor") as demo:
    gr.Markdown(
        """
        # 🍔 Food Delivery Time Predictor
        **Predict how long your delivery will take** — powered by XGBoost trained on real Indian delivery data.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🧑 Delivery Person")
            age = gr.Slider(minimum=18, maximum=40, value=28, step=1, label="Age")
            ratings = gr.Slider(minimum=1.0, maximum=5.0, value=4.5, step=0.1, label="Ratings (1 – 5)")
            vehicle_type = gr.Dropdown(
                choices=["bicycle", "electric_scooter", "motorcycle", "scooter"],
                value="motorcycle",
                label="Vehicle Type",
            )
            gr.Markdown("### 🍽️ Order")
            order_type = gr.Dropdown(
                choices=["Buffet", "Drinks", "Meal", "Snack"],
                value="Meal",
                label="Order Type",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📍 Restaurant Location")
            rest_lat = gr.Number(value=12.972793, label="Latitude")
            rest_lon = gr.Number(value=80.249982, label="Longitude")
            gr.Markdown("### 🏠 Delivery Location")
            del_lat = gr.Number(value=13.012793, label="Latitude")
            del_lon = gr.Number(value=80.289982, label="Longitude")

    predict_btn = gr.Button("🔮 Predict Delivery Time", variant="primary", size="lg")
    output = gr.Markdown()

    predict_btn.click(
        fn=predict_delivery_time,
        inputs=[age, ratings, rest_lat, rest_lon, del_lat, del_lon, order_type, vehicle_type],
        outputs=output,
    )

    gr.Markdown(
        """
        ---
        ### 📖 How to use
        1. Set the delivery person's **age, ratings, and vehicle type**.
        2. Choose the **order type** (Buffet / Drinks / Meal / Snack).
        3. Enter the **restaurant** and **delivery location** coordinates.
        4. Click **Predict**.

        > **Tip:** Right-click any spot on Google Maps → copy the lat/lon shown.
        """
    )

if __name__ == "__main__":
    demo.launch()
