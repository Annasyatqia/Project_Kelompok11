import streamlit as st
import pandas as pd
import joblib
import logging
import os
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 1. SETUP PAGE
# ============================================================
st.set_page_config(page_title="Analisis Air Minum Jawa Barat", layout="wide", page_icon="💧")
st.title("💧 Analisis & Prediksi Ketersediaan Air Minum Jawa Barat")
st.markdown("""
Aplikasi ini membantu menganalisis dan memprediksi ketersediaan air minum yang layak berdasarkan data survei per desa di Jawa Barat.  
Gunakan sidebar untuk navigasi antara **visualisasi data** dan **prediksi wilayah**.
""")

# ============================================================
# 2. LOAD ARTIFACTS DENGAN CACHING (LANGSUNG DARI .PKL)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_ml_artifacts() -> Dict[str, Any]:
    """Load model, scaler, dan encoders langsung dari file .pkl dengan caching untuk performa."""
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        artifacts = {"model": model, "scaler": scaler, **encoders}
        logger.info("✅ Artifacts berhasil dimuat dari .pkl.")
        return artifacts
    except Exception as e:
        st.error(f"❌ Gagal memuat artifacts: {e}. Pastikan file model.pkl, scaler.pkl, dan encoders.pkl ada di repository.")
        st.stop()

with st.spinner("🚀 Sedang memuat model dan resource... (sekitar 10–30 detik)"):
    artifacts = load_ml_artifacts()

model = artifacts.get("model")
scaler = artifacts.get("scaler")
encoders = {k: v for k, v in artifacts.items() if k.startswith("label_")}
if not model or not scaler:
    st.error("❌ Model atau scaler tidak ditemukan. Pastikan file .pkl sudah diunggah di repository.")
    st.stop()
st.success("✅ Model dan resource berhasil dimuat.")

# ============================================================
# 3. NAVIGASI MODE
# ============================================================
mode = st.sidebar.radio("Navigasi", ["📊 Visualisasi Data", "Tinjauan Wilayah"])

# ============================================================
# 4. VISUALISASI DATA (VERSI REVISI & INTERAKTIF)
# ============================================================
if mode == "📊 Visualisasi Data":
    st.header("📊 Visualisasi Distribusi Sumber Air Minum")
    st.markdown("""
    Unggah file **CSV** hasil survei per desa untuk melihat distribusi sumber air di setiap kabupaten/kota.  
    Pastikan file mengandung kolom seperti `bps_nama_kabupaten_kota`, `latitude`, `longitude`, dan kolom sumber air.
    """)

    uploaded_file = st.file_uploader("📁 Unggah file data.csv", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data berhasil diunggah.")
            st.dataframe(df.head())

            # ===============================
            # SUB-MENU DALAM TAB VISUALISASI
            # ===============================
            st.subheader("📊 Eksplorasi & Analisis Data")

            sub_option = st.radio(
                "Pilih tampilan analisis:",
                ("Visualisasi Data", "Analisis Wilayah Rawan", "Faktor Paling Berpengaruh")
            )

            if sub_option == "Visualisasi Data":
                st.write("Tampilkan grafik, distribusi data, atau heatmap di sini.")
                # Contoh placeholder visualisasi (kalau belum ada)
                st.area_chart(df.select_dtypes('number'))

            elif sub_option == "Analisis Wilayah Rawan":
                st.write("### 🗺️ Analisis Wilayah Rawan Kekurangan Air")

                try:
                    from sklearn.cluster import KMeans
                    X = df.select_dtypes('number')
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    df['Cluster'] = kmeans.fit_predict(X)

                    cluster_summary = df.groupby('Cluster').mean()
                    st.dataframe(cluster_summary)

                    st.success("Analisis wilayah rawan berhasil dibuat menggunakan pendekatan K-Means.")
                except Exception as e:
                    st.warning("K-Means belum dijalankan karena data belum tersedia atau belum diproses.")
                    st.text(e)

            elif sub_option == "Faktor Paling Berpengaruh":
                st.write("### 🌿 Faktor Paling Berpengaruh terhadap Kelayakan Air")

                try:
                    import pickle
                    model = pickle.load(open("model.pkl", "rb"))

                    # Coba ambil feature importance dari model RandomForest
                    if hasattr(model, "feature_importances_"):
                        feature_importances = pd.DataFrame({
                            "Feature": df.drop(columns=['target']).columns,
                            "Importance": model.feature_importances_
                        }).sort_values(by="Importance", ascending=False)

                        st.bar_chart(feature_importances.set_index("Feature"))
                        st.success("Berikut faktor-faktor yang paling berpengaruh terhadap ketersediaan air.")
                    else:
                        st.info("Model tidak memiliki atribut feature_importances_.")
                except Exception as e:
                    st.warning("Gagal menampilkan faktor paling berpengaruh.")
                    st.text(e)

            # Validasi kolom
            if "bps_nama_kabupaten_kota" not in df.columns:
                st.error("❌ Kolom 'bps_nama_kabupaten_kota' tidak ditemukan di dataset.")
                st.stop()

            # Dropdown kabupaten
            kabupaten_list = sorted(df["bps_nama_kabupaten_kota"].dropna().unique())
            kabupaten = st.selectbox("🏙️ Pilih Kabupaten/Kota", kabupaten_list)

            # Filter data berdasarkan kabupaten
            df_filtered = df[df["bps_nama_kabupaten_kota"] == kabupaten]

            # Batasi data agar tidak terlalu berat
            if len(df_filtered) > 1000:
                df_filtered = df_filtered.sample(1000)
                st.warning("⚠️ Data besar, hanya menampilkan 1000 titik pertama untuk efisiensi.")

            # Cari kolom sumber air
            sumber_cols = [c for c in df_filtered.columns if "ketersediaan_air_minum_sumber" in c]

            if sumber_cols:
                st.subheader(f"💦 Distribusi Sumber Air di {kabupaten}")

                # Ubah nilai 'ADA' → 1, 'TIDAK' → 0 untuk keperluan grafik
                df_numeric = df_filtered.copy()
                for col in sumber_cols:
                    df_numeric[col] = df_numeric[col].map({'ADA': 1, 'TIDAK': 0}).fillna(0)

                # Hitung jumlah desa yang punya sumber air tertentu
                chart_data = df_numeric[sumber_cols].sum().sort_values(ascending=True)

                # Plot interaktif horizontal bar
                import plotly.express as px
                fig = px.bar(
                    chart_data,
                    x=chart_data.values,
                    y=chart_data.index,
                    orientation="h",
                    labels={"x": "Jumlah Desa", "y": "Jenis Sumber Air"},
                    title=f"Distribusi Sumber Air Minum di {kabupaten}",
                    color=chart_data.values,
                    color_continuous_scale="Blues"
                )
                fig.update_layout(yaxis=dict(title="", tickfont=dict(size=11)))
                st.plotly_chart(fig, use_container_width=True)

                # Peta jika kolom koordinat tersedia
                if {"latitude", "longitude"}.issubset(df_filtered.columns):
                    st.subheader(f"🗺️ Peta Sumber Air di {kabupaten}")
                    m = plot_simple_map(df_filtered, lat_col="latitude", lon_col="longitude", popup_col="bps_nama_desa_kelurahan")
                    st.components.v1.html(m._repr_html_(), height=500)
                else:
                    st.info("ℹ️ Tidak ada kolom 'latitude' dan 'longitude' — hanya menampilkan grafik distribusi.")
            else:
                st.warning("Tidak ditemukan kolom sumber air di data yang diunggah.")

        except Exception as e:
            st.error(f"❌ Terjadi error saat memproses file: {e}")
            logger.error(f"Error saat memproses file unggahan: {e}")
    else:
        st.info("📤 Unggah file CSV untuk mulai menampilkan visualisasi data.")

# ============================================================
# 5. TINJAUAN WILAYAH (HANYA DROPDOWN - ANALISIS SUMBER AIR)
# ============================================================
elif mode == "Tinjauan Wilayah":
    st.header("Tinjauan Ketersediaan Air Minum Layak per Desa")
    st.markdown("Pilih kabupaten/kota dan kecamatan dari dropdown untuk melihat status sumber air yang tersedia.")

    # Upload data untuk dropdown
    uploaded_file_pred = st.file_uploader("Unggah file data.csv (untuk dropdown)", type=["csv"], key="pred_upload")
    df_pred = None
    if uploaded_file_pred:
        try:
            df_pred = pd.read_csv(uploaded_file_pred)
            st.success("✅ Data untuk dropdown berhasil diunggah.")
        except Exception as e:
            st.error(f"❌ Error memuat data: {e}")

    if df_pred is None:
        st.warning("⚠️ Unggah data CSV terlebih dahulu untuk menggunakan dropdown.")
    else:
        # Dropdown Kabupaten
        kabupaten_list = sorted(df_pred["bps_nama_kabupaten_kota"].dropna().unique())
        kabupaten = st.selectbox("Pilih Kabupaten/Kota", kabupaten_list)

        # Dropdown Kecamatan (filtered)
        kecamatan_list = sorted(df_pred[df_pred["bps_nama_kabupaten_kota"] == kabupaten]["bps_nama_kecamatan"].dropna().unique())
        kecamatan = st.selectbox("Pilih Kecamatan", kecamatan_list)

        if st.button("🔍 Analisis Sumber Air"):
            # Filter data berdasarkan kabupaten dan kecamatan
            df_filtered = df_pred[(df_pred["bps_nama_kabupaten_kota"] == kabupaten) & (df_pred["bps_nama_kecamatan"] == kecamatan)]

            if df_filtered.empty:
                st.error("❌ Tidak ada data untuk kabupaten dan kecamatan ini.")
            else:
                # Map kolom sumber air ke binary (0/1) jika masih string
                sumber_cols = [c for c in df_filtered.columns if "ketersediaan_air_minum_sumber" in c]
                for col in sumber_cols:
                    df_filtered[col] = df_filtered[col].map({'ADA': 1, 'TIDAK': 0}).fillna(0).astype(int)

                if not sumber_cols:
                    st.warning("Tidak ditemukan kolom sumber air.")
                else:
                    # Hitung rata-rata atau status (asumsi binary: 1=ada, 0=tidak)
                    sumber_status = {}
                    for col in sumber_cols:
                        avg = df_filtered[col].mean()  # Rata-rata (jika 0/1)
                        sumber_status[col] = "✅ Ada" if avg > 0.5 else "❌ Tidak Ada"

                    st.subheader(f"Status Sumber Air di {kabupaten} - {kecamatan}")
                    for col, status in sumber_status.items():
                        st.write(f"- {col.replace('ketersediaan_air_minum_sumber_', '').replace('_', ' ').title()}: {status}")

                    # Logika analisis: Sumber layak = kemasan, ledeng_meteran, ledeng_tanpa_meteran, mata_air
                    layak_cols = [
                        "ketersediaan_air_minum_sumber_kemasan",
                        "ketersediaan_air_minum_sumber_ledeng_meteran",
                        "ketersediaan_air_minum_sumber_ledeng_tanpa_meteran",
                        "ketersediaan_air_minum_sumber_mata_air",
                    ]
                    layak_count = sum(1 for col in layak_cols if col in sumber_cols and df_filtered[col].mean() > 0.5)

                    if layak_count >= 4:
                        st.success("✅ Aman: Semua sumber air layak tersedia. Tidak perlu ditinjau lagi.")
                    elif layak_count == 1:
                        st.warning("⚠️ Perlu Ditinjau: Hanya 1 sumber air layak tersedia.")
                    else:
                        st.info(f"ℹ️ Jumlah sumber air layak: {layak_count}. Evaluasi lebih lanjut diperlukan.")

# ============================================================
# 6. FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.write("👩🏻‍💻 *Developed by Kelompok 11*")
