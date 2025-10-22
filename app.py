# ============================================================
# STREAMLIT APP: Analisis & Prediksi Ketersediaan Air Minum Jawa Barat
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import logging
import os
from typing import Dict, Any
from sklearn.cluster import KMeans
import plotly.express as px

# ============================================================
# 1. SETUP LOGGING DAN PAGE
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Analisis Air Minum Jawa Barat", layout="wide", page_icon="💧")
st.title("💧 Pemetaan dan Analisis Ketersediaan Air Minum Jawa Barat")

st.markdown("""
Aplikasi ini membantu menganalisis ketersediaan air minum yang layak berdasarkan data survei per desa di Jawa Barat.  
Gunakan sidebar untuk navigasi antara **visualisasi data** dan **tinjauan wilayah**.
""")

# ============================================================
# 2. FUNGSI UTILITAS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_ml_artifacts() -> Dict[str, Any]:
    """Load model, scaler, dan encoders dari file .pkl."""
    try:
        model = joblib.load("model.pkl")
        artifacts = {"model": model}
        logger.info("✅ Model berhasil dimuat dari model.pkl.")
        return artifacts
    except Exception as e:
        st.error(f"❌ Gagal memuat model.pkl: {e}")
        st.stop()

# ============================================================
# 3. LOAD MODEL
# ============================================================
with st.spinner("🚀 Sedang memuat model..."):
    artifacts = load_ml_artifacts()

model = artifacts.get("model")
if not model:
    st.error("❌ Model tidak ditemukan. Pastikan file model.pkl ada di folder yang sama.")
    st.stop()

st.success("✅ Model berhasil dimuat.")

# ============================================================
# 4. NAVIGASI
# ============================================================
mode = st.sidebar.radio("Navigasi", ["📊 Visualisasi & Analisis", "🔎 Tinjauan Wilayah"])
st.sidebar.markdown("---")

# ============================================================
# 5. VISUALISASI & ANALISIS
# ============================================================
if mode == "📊 Visualisasi & Analisis":
    st.header("📊 Visualisasi & Analisis Data")

    uploaded_file = st.file_uploader("📁 Unggah file data.csv", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data berhasil diunggah.")
            st.dataframe(df.head())

            st.subheader("Pilih Jenis Analisis")
            sub_option = st.radio(
                "Pilih tampilan:",
                ("Analisis Wilayah Rawan", "Faktor Paling Berpengaruh")
            )

            # --- Submenu 1: Analisis Wilayah Rawan ---
            if sub_option == "Analisis Wilayah Rawan":
                st.write("### 🗺️ Analisis Wilayah Rawan Kekurangan Air")
                try:
                    X = df.select_dtypes('number').dropna()
                    if X.empty or len(X.columns) < 2:
                        st.warning("⚠️ K-Means belum dijalankan karena data belum lengkap atau belum numerik.")
                    else:
                        kmeans = KMeans(n_clusters=3, random_state=42)
                        df['Cluster'] = kmeans.fit_predict(X)
                        cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
                        st.dataframe(cluster_summary)
                        st.success("✅ Analisis wilayah rawan berhasil dibuat menggunakan K-Means.")

                        fig = px.scatter(
                            df,
                            x=X.columns[0],
                            y=X.columns[1],
                            color=df["Cluster"].astype(str),
                            title="Visualisasi Klaster Wilayah",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Gagal menjalankan K-Means: {e}")

            # --- Submenu 2: Faktor Paling Berpengaruh ---
            elif sub_option == "Faktor Paling Berpengaruh":
                st.write("### 🌿 Faktor Paling Berpengaruh terhadap Kelayakan Air")
                try:
                    if hasattr(model, "feature_importances_"):
                        num_cols = df.select_dtypes('number').columns.tolist()
                        if len(num_cols) != len(model.feature_importances_):
                            st.warning("⚠️ Jumlah kolom fitur tidak sesuai dengan model.")
                        else:
                            importance_df = pd.DataFrame({
                                "Feature": num_cols,
                                "Importance": model.feature_importances_
                            }).sort_values(by="Importance", ascending=False)
                            st.dataframe(importance_df)

                            fig = px.bar(
                                importance_df,
                                x="Importance",
                                y="Feature",
                                orientation="h",
                                title="Faktor Paling Berpengaruh terhadap Ketersediaan Air"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.success("✅ Faktor-faktor paling berpengaruh berhasil ditampilkan.")
                    else:
                        st.info("Model tidak memiliki atribut feature_importances_.")
                except Exception as e:
                    st.warning(f"⚠️ Gagal menampilkan faktor paling berpengaruh: {e}")

        except Exception as e:
            st.error(f"❌ Terjadi error saat memproses file: {e}")
            logger.error(f"Error saat memproses file unggahan: {e}")
    else:
        st.info("📤 Unggah file CSV untuk mulai menampilkan visualisasi dan analisis data.")

# ============================================================
# 6. TINJAUAN WILAYAH
# ============================================================
elif mode == "🔎 Tinjauan Wilayah":
    st.header("🔎 Tinjauan Ketersediaan Air Minum Layak per Desa")
    st.markdown("Pilih kabupaten/kota dan kecamatan untuk melihat status sumber air yang tersedia.")

    uploaded_file_pred = st.file_uploader("Unggah file data.csv (untuk tinjauan wilayah)", type=["csv"], key="pred_upload")
    if uploaded_file_pred:
        try:
            df_pred = pd.read_csv(uploaded_file_pred)
            st.success("✅ Data berhasil dimuat untuk analisis wilayah.")

            kabupaten_list = sorted(df_pred["bps_nama_kabupaten_kota"].dropna().unique())
            kabupaten = st.selectbox("Pilih Kabupaten/Kota", kabupaten_list)
            kecamatan_list = sorted(df_pred[df_pred["bps_nama_kabupaten_kota"] == kabupaten]["bps_nama_kecamatan"].dropna().unique())
            kecamatan = st.selectbox("Pilih Kecamatan", kecamatan_list)

            if st.button("🔍 Analisis Sumber Air"):
                df_filtered = df_pred[
                    (df_pred["bps_nama_kabupaten_kota"] == kabupaten) &
                    (df_pred["bps_nama_kecamatan"] == kecamatan)
                ]

                if df_filtered.empty:
                    st.error("❌ Tidak ada data untuk kabupaten dan kecamatan ini.")
                else:
                    sumber_cols = [c for c in df_filtered.columns if "ketersediaan_air_minum_sumber" in c]
                    for col in sumber_cols:
                        df_filtered[col] = df_filtered[col].map({'ADA': 1, 'TIDAK': 0}).fillna(0).astype(int)

                    st.subheader(f"Status Sumber Air di {kabupaten} - {kecamatan}")
                    for col in sumber_cols:
                        avg = df_filtered[col].mean()
                        status = "✅ Ada" if avg > 0.5 else "❌ Tidak Ada"
                        st.write(f"- {col.replace('ketersediaan_air_minum_sumber_', '').replace('_', ' ').title()}: {status}")

                    layak_cols = [
                        "ketersediaan_air_minum_sumber_kemasan",
                        "ketersediaan_air_minum_sumber_ledeng_meteran",
                        "ketersediaan_air_minum_sumber_ledeng_tanpa_meteran",
                        "ketersediaan_air_minum_sumber_mata_air",
                    ]
                    layak_count = sum(1 for col in layak_cols if col in sumber_cols and df_filtered[col].mean() > 0.5)

                    if layak_count >= 4:
                        st.success("✅ Aman: Semua sumber air layak tersedia.")
                    elif layak_count == 1:
                        st.warning("⚠️ Perlu Ditinjau: Hanya 1 sumber air layak tersedia.")
                    else:
                        st.info(f"ℹ️ Jumlah sumber air layak: {layak_count}. Evaluasi lebih lanjut diperlukan.")
        except Exception as e:
            st.error(f"❌ Error memuat data: {e}")
    else:
        st.info("📤 Unggah data CSV untuk mulai meninjau wilayah.")

# ============================================================
# 7. FOOTER
# ============================================================
st.sidebar.write("👩🏻‍💻 *Developed by Kelompok 11*")
