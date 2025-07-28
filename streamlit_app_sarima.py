# streamlit_app_sarima.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy.stats import ttest_rel
from scipy.stats import t as tdist
import io
import warnings
warnings.filterwarnings('ignore')

# Custom CSS untuk mempercantik sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .sidebar-box {
        background-color: white;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 0 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigasi
with st.sidebar:
    st.markdown('<div class="sidebar-title">📌 Navigation Menu</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    selected_menu = st.radio("", [
        "📊 Data & Visualization",
        "🌧️ Rainfall Distribution",
        "📉 Decomposition",
        "📈 ACF/PACF & ADF Test",
        "🔍 T Significance Test",
        "🛠️ Train SARIMA"
    ], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# --- LOAD DATA ---
DATA_PATH = 'data/data percobaan surabaya FINAL 3.xlsx'

if os.path.exists(DATA_PATH):
    df = pd.read_excel(DATA_PATH)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.set_index('Tanggal')
    st.sidebar.success(f"✅ Data ditemukan: {DATA_PATH}")
else:
    st.sidebar.error(f"🚫 Data tidak ditemukan di path {DATA_PATH}")
    st.stop()

# --- MAIN CONTENT ---
st.title("📈 SARIMA Dashboard Monthly Rainfall Forecast in Surabaya")

# Halaman: Data & Visualisasi
if selected_menu == "📊 Data & Visualization":
    st.title("📊 Dashboard SARIMA Forecasting")
    st.header("📄 Preview Data")
    df_display = df.copy()
    df_display.index = df_display.index.strftime('%Y-%m-%d')  # Hilangkan jam
    st.dataframe(df_display, height=400, use_container_width=True)


    st.header("📊 Time Series Visualization")
    fig, ax = plt.subplots(figsize=(12, 4))
    df['RR'].plot(ax=ax)
    ax.set_title("Time Series of Rainfall in Surabaya City 2020-2024")
    ax.set_ylabel("Rainfall")
    ax.set_xlabel("Year")
    st.pyplot(fig)

    # Halaman Distribusi Curah Hujan
elif selected_menu == "🌧️ Rainfall Distribution":
    st.header("📈 Distribution of Monthly Rainfall in Surabaya")

    # Hitung total curah hujan per bulan
    monthly_rainfall = df['RR'].resample('M').sum()

    # Plot Line Chart
    fig4, ax = plt.subplots(figsize=(12,6))
    ax.plot(monthly_rainfall.index, monthly_rainfall.values, marker='o', linestyle='-', color='blue')
    ax.set_title('Distribution of Monthly Rainfall in Surabaya', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Rainfall (mm)', fontsize=12)
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    st.header("📊 Monthly Rainfall Boxplot")

    # Kolom nama bulan dari index
    df['Bulan'] = df.index.strftime('%b')
    df['Bulan'] = pd.Categorical(df['Bulan'], categories=[
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ], ordered=True)

    # Plot Boxplot
    fig5, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Bulan', y='RR', data=df, palette='Set3', ax=ax)
    ax.set_title('Monthly Rainfall Boxplot')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Curah Hujan (mm)')
    st.pyplot(fig5)

# Halaman: Decomposition
elif selected_menu == "📉 Decomposition":
    st.title("📉 Decomposition Time Series")
    decomposition = seasonal_decompose(df['RR'], model='additive', period=365)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    decomposition.observed.plot(ax=ax1, title='Observed')
    decomposition.trend.plot(ax=ax2, title='Trend')
    decomposition.seasonal.plot(ax=ax3, title='Seasonal')
    decomposition.resid.plot(ax=ax4, title='Residual')
    st.pyplot(fig)

elif selected_menu == "📈 ACF/PACF & ADF Test":
    st.header("📈 ACF & PACF Plot - Before Seasonal Differencing")
    fig1, ax = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(df['RR'], ax=ax[0])
    plot_pacf(df['RR'], ax=ax[1])
    ax[0].set_title('ACF Before Seasonal Differencing')
    ax[1].set_title('PACF Before Seasonal Differencing')
    st.pyplot(fig1)

    st.header("🧪 ADF Test - Before Seasonal Differencing")
    adf_result = adfuller(df['RR'])
    adf_df = pd.DataFrame({
        'Statistik ADF': [adf_result[0]],
        'p-value': [adf_result[1]],
        'Critical Value 1%': [adf_result[4]['1%']],
        'Critical Value 5%': [adf_result[4]['5%']],
        'Critical Value 10%': [adf_result[4]['10%']]
    })
    st.dataframe(adf_df)

    alpha = 0.05
    if adf_result[1] < alpha:
        st.success(f"✅ p-value < {alpha}: Data sudah stasioner di rata-rata")
    else:
        st.warning(f"❌ p-value ≥ {alpha}: Data belum stasioner di rata-rata")

    st.divider()

    # User options
    st.subheader("⚙️ Konfigurasi Analisis")
    diff_type = st.radio("Pilih Jenis Differencing:", ["Non-Seasonal (d)", "Seasonal (D)"])
    split_option = st.selectbox("Pilih Rasio Data Training:", {
        "90:10": 1644,
        "80:20": 1462,
        "70:30": 1279,
    }.keys())
    split_index = {
        "90:10": 1644,
        "80:20": 1462,
        "70:30": 1279,
    }[split_option]

    # Apply differencing
    if diff_type == "Non-Seasonal (d)":
        df_diff = df['RR'].diff().dropna()
        title = "Non Seasonal Differencing (d)"
    else:
        df_diff = df['RR'].diff(12).dropna()
        title = "Seasonal Differencing (D)"

    st.header(f"📉 Time Series Plot - After {title}")
    fig_ts, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df_diff, ax=ax)
    ax.set_title(f'Time Series After {title}')
    ax.set_ylabel('Δ Curah Hujan')
    ax.set_xlabel('Year')
    ax.grid(True)
    st.pyplot(fig_ts)

    train_data = df_diff.iloc[:split_index]

    st.header(f"📈 ACF & PACF Plot - After {title}")
    fig_acf, ax = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(train_data, lags=30, ax=ax[0])
    plot_pacf(train_data, lags=30, ax=ax[1])
    ax[0].set_title(f'ACF After {title}')
    ax[1].set_title(f'PACF After {title}')
    st.pyplot(fig_acf)

    st.header(f"🧪 ADF Test - After {title}")
    adf_result = adfuller(train_data)
    adf_df = pd.DataFrame({
        'Statistik ADF': [adf_result[0]],
        'p-value': [adf_result[1]],
        'Critical Value 1%': [adf_result[4]['1%']],
        'Critical Value 5%': [adf_result[4]['5%']],
        'Critical Value 10%': [adf_result[4]['10%']]
    })
    st.dataframe(adf_df)

    if adf_result[1] < alpha:
        st.success(f"✅ p-value < {alpha}: Data sudah stasioner di rata-rata")
    else:
        st.warning(f"❌ p-value ≥ {alpha}: Data belum stasioner di rata-rata")

    st.header("🔍 Kandidat Model SARIMA berdasarkan ACF & PACF")

    model_data = {
        "Model": ["(2,0,0)(2,1,0)[12]", "(0,0,2)(0,1,2)[12]", "(2,0,2)(2,1,2)[12]"],
        "Short Description": [
            "Strong AR(2) and seasonal AR(2) indicated by PACF",
            "Strong MA(2) and seasonal MA(2) indicated by ACF",
            "Combination of ARMA(2,2), seasonal ARMA(2,2), and Seasonal Differencing D(1)"
        ]
    }
    model_df = pd.DataFrame(model_data)
    st.table(model_df)

# Halaman Uji T
elif selected_menu == "🔍 T Significance Test":
    st.header("🔍 T Significance Test")

    try:
        from scipy.stats import t as t_dist

        # --- Pilihan rasio pembagian data training ---
        st.subheader("⚙️ Konfigurasi Data Training")
        split_option = st.selectbox("Pilih Rasio Data Training:", {
            "90:10": 1644,
            "80:20": 1462,
            "70:30": 1279,
        }.keys())
        split_index = {
            "90:10": 1644,
            "80:20": 1462,
            "70:30": 1279
        }[split_option]

        # --- Differencing musiman ---
        df_diff_seasonal = df['RR'].diff(12).dropna()
        y_train = df_diff_seasonal.iloc[:split_index]

        # --- Daftar Model SARIMA ---
        kandidat_model = {
            "(2,0,0)(2,1,0)[12]": {"order": (2,0,0), "seasonal_order": (2,1,0,12)},
            "(0,0,2)(0,1,2)[12]": {"order": (0,0,2), "seasonal_order": (0,1,2,12)},
            "(2,0,2)(2,1,2)[12]": {"order": (2,0,2), "seasonal_order": (2,1,2,12)},
        }

        for model_name, params in kandidat_model.items():
            st.markdown(f"### 🔍 Model {model_name}")

            # --- Latih model ---
            model = SARIMAX(y_train, order=params["order"], seasonal_order=params["seasonal_order"],
                            enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)

            # --- Hitung t-statistik dan p-value ---
            st.markdown("#### 📊 T Significance Test Table")
            params = model_fit.params
            bse = model_fit.bse
            t_stat = params / bse
            n = len(y_train)
            k = len(params)
            df_t = n - k
            p_values = [2 * (1 - t_dist.cdf(abs(ti), df_t)) for ti in t_stat]
            signifikan = ["Significant" if p < 0.05 else "Not Significant" for p in p_values]

            t_test_df = pd.DataFrame({
                "Parameter": params.index,
                "Koefisien": np.round(params.values, 4),
                "Std Error": np.round(bse.values, 4),
                "t-Statistik": np.round(t_stat.values, 4),
                "p-Value": np.round(p_values, 5),
                "Signifikansi": signifikan
            })

            st.dataframe(t_test_df.style.format({
                "Koefisien": "{:.4f}",
                "Std Error": "{:.4f}",
                "t-Statistik": "{:.2f}",
                "p-Value": "{:.4f}"
            }), use_container_width=True)

            st.markdown("---")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat evaluasi model: {e}")

# Halaman Train SARIMA
elif selected_menu == "🛠️ Train SARIMA":
    st.header("🛠️ Train Model SARIMA")

    kandidat_model = {
        "(2,0,0)(2,1,0)[12]": {"order": (2,0,0), "seasonal_order": (2,1,0,12)},
        "(0,0,2)(0,1,2)[12]": {"order": (0,0,2), "seasonal_order": (0,1,2,12)},
        "(2,0,2)(2,1,2)[12]": {"order": (2,0,2), "seasonal_order": (2,1,2,12)},
    }

    model_options = ["Choose model"] + list(kandidat_model.keys())
    selected_model = st.selectbox("📌 Select SARIMA Model", options=model_options)

    split_ratio = st.selectbox("🔀 Select Train-Test Split", options=["90:10", "80:20", "70:30"])
    split_map = {"90:10": 0.90, "80:20": 0.80, "70:30": 0.70}

    if selected_model != "Choose model" and st.button("🚀 Train and Forecast"):
        try:
            params = kandidat_model[selected_model]
            df.index = pd.to_datetime(df.index)
            y = df['RR'].resample('MS').mean()

            # === SPLIT DATA ===
            ratio = split_map[split_ratio]
            split_index = int(len(y) * ratio)
            y_train = y.iloc[:split_index]
            y_test = y.iloc[split_index:]

            # === Fit SARIMA ===
            model = SARIMAX(
                y_train,
                order=params['order'],
                seasonal_order=params['seasonal_order'],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)

            # === Forecast Hingga Desember 2025 ===
            last_train_date = y_train.index[-1]
            forecast_end = pd.to_datetime("2025-12-01")
            forecast_steps = (forecast_end.year - last_train_date.year) * 12 + (forecast_end.month - last_train_date.month)

            forecast_result = model_fit.get_forecast(steps=forecast_steps)
            forecast_mean = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()

            # === Visualisasi ===
            fig, ax = plt.subplots(figsize=(18, 9))

            ax.plot(y.index, y.values, label='Training Data (Historical)', color='blue', linewidth=2)
            ax.plot(y_test.index, y_test.values, label='Testing Data', color='orange', linewidth=2)
            ax.plot(forecast_mean.index, forecast_mean.values, label='Forecast', color='green', linewidth=2)
            ax.fill_between(forecast_mean.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                            color='green', alpha=0.3, label='Confidence Interval')
            ax.axvline(x=y_test.index[0], color='red', linestyle='--', linewidth=2, label='Start Forecasting')

            ax.set_title(f'SARIMA Forecast {selected_model} with {split_ratio} Data Split Until December 2025', fontsize=18)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('Average Rainfall', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right', fontsize=10)
            plt.xlim(pd.to_datetime('2020-01-01'), forecast_mean.index[-1] + pd.DateOffset(months=1))
            plt.tight_layout()

            st.pyplot(fig)

            # === Evaluasi ===
            forecast_for_eval = forecast_mean[:len(y_test)]
            mae = mean_absolute_error(y_test, forecast_for_eval)
            rmse = np.sqrt(mean_squared_error(y_test, forecast_for_eval))
            mse = mean_squared_error(y_test, forecast_for_eval)
            bic_value = model_fit.bic if not np.isnan(model_fit.bic) else "N/A"
            
            st.subheader("📋 Evaluation")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("MAE", f"{mae:.2f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
            with col3:
                st.metric("MSE", f"{mse:.2f}")
            with col4:
                st.metric("📊 AIC", f"{model_fit.aic:.2f}")
            with col5:
                st.metric("📊 BIC", f"{bic_value}" if isinstance(bic_value, str) else f"{bic_value:.2f}")

            st.session_state['model_fit'] = model_fit
            st.session_state['trained_model_name'] = selected_model
            st.success(f"✅ Model {selected_model} trained and evaluated successfully with {split_ratio} split!")

        except Exception as e:
            st.error(f"❌ Failed to train and evaluate the model {selected_model}: {e}")
    elif selected_model == "Choose model":
        st.info("📌 Please select the model first before training.")













