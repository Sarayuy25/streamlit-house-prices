import streamlit as st
import pandas as pd
import joblib
import mysql.connector
import plotly.express as px
import os
from datetime import datetime

# ----------------------
# Load Model & Dataset
# ----------------------
model = joblib.load("house_price_model.pkl")
df = pd.read_csv("listings.csv")
pd.set_option("display.float_format", "{:,.0f}".format)

# ----------------------
# MySQL Connection
# ----------------------
def create_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="farrelbias25", 
            database="house_prices"
        )
        return conn
    except:
        return None

def save_prediction(conn, bedroom, bathroom, land, building, price):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if conn:  # MySQL available
        cursor = conn.cursor()
        sql = """
        INSERT INTO prediksi (bedroom, bathroom, land_clean, building_clean, price_pred)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (bedroom, bathroom, land, building, price))
        conn.commit()
        cursor.close()
    else:  # fallback CSV
        fallback_file = "prediksi_fallback.csv"
        data = {
            "timestamp": [timestamp],
            "bedroom": [bedroom],
            "bathroom": [bathroom],
            "land_clean": [land],
            "building_clean": [building],
            "price_pred": [price]
        }
        df_fallback = pd.DataFrame(data)
        if os.path.exists(fallback_file):
            df_fallback.to_csv(fallback_file, mode='a', header=False, index=False)
        else:
            df_fallback.to_csv(fallback_file, index=False)
        st.info(f"Koneksi MySQL gagal, prediksi disimpan ke {fallback_file}")

# ----------------------
# Sidebar Input User
# ----------------------
st.sidebar.header("Input Data Rumah")
bedroom = st.sidebar.number_input("Jumlah Kamar Tidur", 1, 10, 3)
bathroom = st.sidebar.number_input("Jumlah Kamar Mandi", 1, 10, 2)
land_clean = st.sidebar.number_input("Luas Tanah (m²)", 10, 1000, 120)
building_clean = st.sidebar.number_input("Luas Bangunan (m²)", 10, 1000, 80)

locations = df["location"].unique()
selected_locations = st.sidebar.multiselect("Filter Lokasi Scatter Plot", options=locations, default=locations)

# ----------------------
# Prediksi Harga
# ----------------------
st.subheader("Prediksi Harga Rumah")
if st.button("Prediksi Harga"):
    input_data = [[bedroom, bathroom, land_clean, building_clean]]
    pred_price = model.predict(input_data)[0]
    st.success(f"Harga prediksi: Rp {int(pred_price):,}")

    conn = create_connection()
    save_prediction(conn, bedroom, bathroom, land_clean, building_clean, int(pred_price))
    if conn:
        st.info("Prediksi berhasil disimpan ke MySQL ✅")

# ======================
# History Prediksi
# ======================
st.subheader("History Prediksi Terakhir")
fallback_file = "prediksi_fallback.csv"
try:
    if os.path.exists(fallback_file):
        df_history = pd.read_csv(fallback_file)
        st.dataframe(df_history.sort_values(by="timestamp", ascending=False).head(20))
    else:
        st.info("Belum ada prediksi fallback CSV")
except:
    st.warning("Gagal membaca file prediksi fallback")

# ======================
# Summary Statistik Dataset
# ======================
st.subheader("Summary Statistik Rumah")
st.dataframe(df.describe().T)

st.subheader("Rata-rata Harga per Lokasi")
avg_price = df.groupby("location")["price_clean"].mean().sort_values(ascending=False)
st.bar_chart(avg_price)

# ======================
# Distribusi Harga
# ======================
st.subheader("Distribusi Harga Rumah")
fig_hist = px.histogram(df, x="price_clean", nbins=50, title="Distribusi Harga Rumah")
fig_hist.update_layout(xaxis_title="Harga (Rp)", yaxis_title="Jumlah Listing")
st.plotly_chart(fig_hist)

# ======================
# Scatter Plot Interaktif
# ======================
st.subheader("Scatter Plot Harga vs Luas Bangunan")
filtered_df = df[df["location"].isin(selected_locations)]
fig_building = px.scatter(
    filtered_df,
    x="building_clean",
    y="price_clean",
    color="location",
    hover_data=["bedroom","bathroom","land_clean"],
    title="Harga vs Luas Bangunan"
)
st.plotly_chart(fig_building)

st.subheader("Scatter Plot Harga vs Luas Tanah")
fig_land = px.scatter(
    filtered_df,
    x="land_clean",
    y="price_clean",
    color="location",
    hover_data=["bedroom","bathroom","building_clean"],
    title="Harga vs Luas Tanah"
)
st.plotly_chart(fig_land)

# ======================
# Feature Importance
# ======================
st.subheader("Feature Importance - Random Forest")
importances = model.feature_importances_
features = ["bedroom","bathroom","land_clean","building_clean"]
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance"
)
st.plotly_chart(fig_importance)





