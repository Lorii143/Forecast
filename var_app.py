# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from statsmodels.tsa.api import VAR
import statsmodels.api as sm

st.set_page_config(page_title="Road Condition Survey: EDA, Reliability, Regression & VAR", layout="wide")
st.title("📊 Road Condition Survey: EDA, Reliability, Regression & VAR")

# -------------------------
# File upload / normalization
# -------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a CSV file (columns: survey area, environmental, traffic, road surface, month, r, l, t).")
    st.stop()

df = pd.read_csv(uploaded_file)

# Normalize columns
df.columns = df.columns.str.strip().str.lower()

st.subheader("Editable Data (you can correct or enter values)")
df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
st.markdown("---")

# Ensure responses exist
expected_responses = ['r', 'l', 't']
for col in expected_responses:
    if col not in df.columns:
        st.error(f"Missing required column: '{col}'.")
        st.stop()

# -------------------------
# Sidebar Navigation
# -------------------------
menu = st.sidebar.radio(
    "Navigate",
    ["EDA", "Reliability", "Regression", "VAR Model"]
)

# -------------------------
# 1. EDA
# -------------------------
if menu == "EDA":
    st.header("1) Exploratory Data Analysis")
    st.write(df.describe(include="all"))

    # Correlation heatmap (exclude month)
    st.write("### Correlation Heatmap (excluding 'month')")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("month", errors="ignore")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns (excluding 'month') for a correlation heatmap.")

# -------------------------
# 2. Reliability (Cronbach's Alpha + Hierarchical Clustering)
# -------------------------
elif menu == "Reliability":
    st.header("2) Reliability Analysis")

    def cronbach_alpha(df_num):
        df_corr = df_num.corr()
        n_items = len(df_corr)
        if n_items < 2:
            return np.nan
        mean_corr = df_corr.values[np.triu_indices(n_items, 1)].mean()
        return (n_items * mean_corr) / (1 + (n_items - 1) * mean_corr)

    numeric_for_reliability = df[expected_responses].apply(pd.to_numeric, errors='coerce').dropna()

    if numeric_for_reliability.shape[1] > 1:
        alpha = cronbach_alpha(numeric_for_reliability)
        st.write(f"**Cronbach's Alpha (R,L,T):** {alpha:.3f}")

        # --- Hierarchical clustering using correlation distance ---
        st.write("#### Hierarchical Clustering of Items (Correlation Distance)")
        try:
            corr_matrix = numeric_for_reliability.corr()
            distance = 1 - corr_matrix

            from scipy.spatial.distance import squareform
            condensed_dist = squareform(distance)

            Z = linkage(condensed_dist, method='average')
            fig, ax = plt.subplots(figsize=(6, 4))
            dendrogram(Z, labels=numeric_for_reliability.columns, leaf_rotation=0, ax=ax)
            ax.set_title("Hierarchical Clustering of Items")
            ax.set_ylabel("Distance (1 - correlation)")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not compute hierarchical clustering: {e}")

        # --- Clustermap heatmap ---
        st.write("#### Hierarchical Clustering Heatmap")
        try:
            g = sns.clustermap(
                numeric_for_reliability.corr(),
                annot=True,
                cmap='coolwarm',
                figsize=(5, 5)
            )
            st.pyplot(g.fig)
        except Exception as e:
            st.warning(f"Could not plot clustermap: {e}")

    else:
        st.warning("Not enough numeric variables for reliability analysis.")

# -------------------------
# 3. Multivariate Regression
# -------------------------
elif menu == "Regression":
    st.header("3) Multivariate Regression")

    possible_cats = ['environmental', 'traffic', 'road surface']
    cats = [c for c in possible_cats if c in df.columns]

    if not cats:
        st.warning("No categorical predictors found.")
    else:
        X = pd.get_dummies(df[cats], drop_first=True)
        y = df[expected_responses].apply(pd.to_numeric, errors='coerce')
        reg_df = pd.concat([X, y], axis=1).dropna()

        if reg_df.shape[0] < 3:
            st.warning("Not enough valid rows for regression.")
        else:
            X_reg = sm.add_constant(reg_df[X.columns].astype(float))
            for col in expected_responses:
                model = sm.OLS(reg_df[col].astype(float), X_reg).fit()
                st.write(f"### Regression results for {col.upper()}")
                st.text(model.summary())

# -------------------------
# 4. Time Series VAR Model
# -------------------------
elif menu == "VAR Model":
    st.header("4) Time Series VAR Model (R, L, T)")

    if "month" not in df.columns:
        st.error("⚠️ 'month' column is required for time series analysis.")
    else:
        try:
            # Prepare time series
            df_ts = df[['month', 'r', 'l', 't']].dropna().sort_values('month')
            df_ts = df_ts.set_index('month')

            # Plot original series
            st.write("### Original Time Series")
            fig, ax = plt.subplots(figsize=(8,4))
            for col in ['r','l','t']:
                ax.plot(df_ts.index, df_ts[col], marker='o', label=col.upper())
            ax.set_xlabel("Month"); ax.set_ylabel("Value"); ax.legend()
            st.pyplot(fig)

            # Stationarity test (ADF)
            st.write("### Augmented Dickey-Fuller Test")
            from statsmodels.tsa.stattools import adfuller
            for col in df_ts.columns:
                result = adfuller(df_ts[col])
                st.write(f"{col.upper()}: ADF={result[0]:.3f}, p-value={result[1]:.3f}")

            # Difference data for stationarity
            df_ts_diff = df_ts.diff().dropna()

            # Fit VAR
            model = VAR(df_ts_diff)
            lag_order = model.select_order(maxlags=5)
            st.write("### Lag Order Selection Criteria")
            st.text(lag_order.summary())

            # Fit with lag chosen by AIC
            selected_lag = lag_order.selected_orders.get("aic", 1)
            model_fitted = model.fit(selected_lag)
            st.write("### VAR Model Summary")
            st.text(model_fitted.summary())

            # Forecast next 3 months (differenced)
            steps = 3
            forecast_diff = model_fitted.forecast(df_ts_diff.values[-model_fitted.k_ar:], steps=steps)
            forecast_df = pd.DataFrame(forecast_diff, columns=['r','l','t'])

            # Convert back to original scale
            last_values = df_ts.iloc[-1]
            forecast_original = forecast_df.cumsum() + last_values

            # Display forecasted table
            st.write("### Forecasted Values (Original Scale)")
            st.dataframe(forecast_original)

            # Plot forecast vs actual
            fig, ax = plt.subplots(figsize=(8,4))
            for col in ['r','l','t']:
                ax.plot(df_ts.index, df_ts[col], label=f"Actual {col.upper()}")
                ax.plot(range(df_ts.index[-1]+1, df_ts.index[-1]+1+steps),
                        forecast_original[col], 'o--', label=f"Forecast {col.upper()}")
            ax.set_xlabel("Month"); ax.set_ylabel("Value"); ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"VAR model error: {e}")
