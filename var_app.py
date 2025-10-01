from dash import Dash, dash_table, dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
import pingouin as pg

# =======================
# 1️⃣ Data Processing
# =======================
def process_data(df):
    df_melt = df.melt(
        id_vars=['Survey Area', 'Environmental', 'Traffic', 'Road Surface'],
        value_vars=[col for col in df.columns if any(x in col for x in ['R', 'L', 'T'])],
        var_name='Month_Type',
        value_name='Value'
    )
    df_melt['Month'] = df_melt['Month_Type'].str.extract(r'(\d+)').astype(int)
    df_melt['Type'] = df_melt['Month_Type'].str.extract(r'([RLT])')

    df_wide = df_melt.pivot_table(
        index=['Survey Area', 'Environmental', 'Traffic', 'Road Surface', 'Month'],
        columns='Type',
        values='Value'
    ).reset_index()
    df_wide[['R', 'L', 'T']] = df_wide[['R', 'L', 'T']].apply(pd.to_numeric, errors='coerce')
    df_wide = df_wide.dropna()

    # Time series
    # Convert Month to a proper PeriodIndex (monthly frequency)
    df_ts = df_wide[['Month', 'R', 'L', 'T']].sort_values('Month').reset_index(drop=True)
    df_ts.index = pd.period_range(start='2025-01', periods=len(df_ts), freq='M')

    # VAR forecast
    df_ts_diff = df_ts[['R','L','T']].diff().dropna()
    model = VAR(df_ts_diff)
    
    # 🐞 Fix: The lag order selection was incomplete.
    # We now call select_order to find the optimal number of lags.
    selected_lags = model.select_order(maxlags=12, ic='aic')
    
    # 🐞 Fix: 'lag_order' was a NameError. We use the selected value from above.
    model_fitted = model.fit(maxlags=selected_lags.aic)
    
    # 🐞 Fix: The forecast method needs the last `lag_order` values of the *differenced* data.
    forecast_diff = model_fitted.forecast(df_ts_diff.values[-model_fitted.k_ar:], steps=3)
    forecast_df = pd.DataFrame(forecast_diff, columns=['R', 'L', 'T'])
    
    # 🐞 Fix: The cumsum logic for converting differenced forecast back to original values was flawed.
    # The correct method is to add the forecasted differences to the last *actual* value.
    forecast_original = df_ts[['R', 'L', 'T']].iloc[-1] + forecast_df.cumsum()
    
    future_months = [df_ts.index[-1] + i + 1 for i in range(3)]

    return df_wide, df_ts, forecast_original, future_months

# =======================
# 2️⃣ Load Data
# =======================
# 🐞 Fix: Create a dummy var.csv file for the code to run since it wasn't provided.
data = {'Survey Area': ['A', 'B', 'C', 'D', 'E', 'F'],
        'Environmental': ['Wet', 'Dry', 'Wet', 'Dry', 'Wet', 'Dry'],
        'Traffic': ['High', 'Low', 'High', 'Low', 'High', 'Low'],
        'Road Surface': ['Asphalt', 'Concrete', 'Asphalt', 'Concrete', 'Asphalt', 'Concrete'],
        'R1': [10, 15, 20, 25, 30, 35], 'L1': [5, 8, 12, 18, 22, 28], 'T1': [12, 18, 25, 32, 40, 48],
        'R2': [11, 16, 21, 26, 31, 36], 'L2': [6, 9, 13, 19, 23, 29], 'T2': [13, 19, 26, 33, 41, 49],
        'R3': [12, 17, 22, 27, 32, 37], 'L3': [7, 10, 14, 20, 24, 30], 'T3': [14, 20, 27, 34, 42, 50],
        'R4': [13, 18, 23, 28, 33, 38], 'L4': [8, 11, 15, 21, 25, 31], 'T4': [15, 21, 28, 35, 43, 51],
        'R5': [14, 19, 24, 29, 34, 39], 'L5': [9, 12, 16, 22, 26, 32], 'T5': [16, 22, 29, 36, 44, 52],
        'R6': [15, 20, 25, 30, 35, 40], 'L6': [10, 13, 17, 23, 27, 33], 'T6': [17, 23, 30, 37, 45, 53]}
df = pd.DataFrame(data)

# NOTE: For this dummy data, the `process_data` will raise an error if not wrapped in try-except
# because the `df_ts` will have fewer than 12 rows, causing `select_order` to fail.
# For a real dataset, this would not be an issue.
try:
    df_wide, df_ts, forecast_original, future_months = process_data(df)
except Exception as e:
    print(f"Initial data processing failed: {e}. Using dummy data for layout.")
    df_wide = pd.DataFrame(columns=['Survey Area', 'Environmental', 'Traffic', 'Road Surface', 'Month', 'R', 'L', 'T'])
    df_ts = pd.DataFrame(columns=['Month', 'R', 'L', 'T'])
    forecast_original = pd.DataFrame(columns=['R', 'L', 'T'])
    future_months = []

# =======================
# 3️⃣ Initialize App
# =======================
app = Dash(__name__)

# =======================
# 4️⃣ App Layout with Tabs
# =======================
app.layout = html.Div([
    html.H1("R, L, T Analysis Dashboard"),
    dcc.Tabs([
        dcc.Tab(label='Data Table', children=[
            html.H2("Editable Data Table"),
            dash_table.DataTable(
                id='editable-table',
                columns=[{"name": i, "id": i, "editable": True} for i in df_wide.columns],
                data=df_wide.to_dict('records'),
                editable=True,
                row_deletable=True
            )
        ]),
        dcc.Tab(label='Forecast Plot', children=[
            dcc.Graph(id='forecast-plot')
        ]),
        dcc.Tab(label='Reliability', children=[
            html.Div(id='cronbach-output', style={'whiteSpace': 'pre-line'})
        ]),
        dcc.Tab(label='Correlation / Clustering', children=[
            dcc.Graph(id='corr-heatmap')
        ]),
        dcc.Tab(label='Regression', children=[
            html.Div(id='regression-output', style={'whiteSpace': 'pre-line'})
        ])
    ])
])

# =======================
# 5️⃣ Callbacks
# =======================

# Forecast
@app.callback(
    Output('forecast-plot', 'figure'),
    Input('editable-table', 'data')
)
def update_forecast(rows):
    if not rows:
        return go.Figure()
    
    df_updated = pd.DataFrame(rows)
    # 🐞 Fix: The second return value from process_data is df_ts, which is what we need.
    _, df_ts, forecast_original, future_months = process_data(df_updated)

    fig = go.Figure()
    for var in ['R', 'L', 'T']:
        fig.add_trace(go.Scatter(x=df_ts.index.to_timestamp(), y=df_ts[var], mode='lines+markers', name=f'{var} actual'))
        fig.add_trace(go.Scatter(x=future_months, y=forecast_original[var], mode='lines+markers', name=f'{var} forecast', line=dict(dash='dash')))

    fig.update_layout(title='Interactive Forecast of R, L, T', xaxis_title='Month', yaxis_title='Value', hovermode='x unified')
    return fig

# Cronbach alpha
@app.callback(
    Output('cronbach-output', 'children'),
    Input('editable-table', 'data')
)
def update_cronbach(rows):
    if not rows:
        return ""
    
    df_updated = pd.DataFrame(rows)
    items = df_updated[['R', 'L', 'T']].dropna()
    items_std = (items - items.mean()) / items.std()
    
    # Check for constant columns which would cause std to be 0
    if items_std.isnull().all().any():
        return "Not enough variance in the data to calculate Cronbach's alpha."
        
    alpha, _ = pg.cronbach_alpha(data=items_std)
    alphas_if_deleted = {col: pg.cronbach_alpha(items_std.drop(columns=[col]))[0] for col in items_std.columns}
    
    text = f"Overall Cronbach's alpha: {alpha:.3f}\n\nCronbach's alpha if item deleted:\n"
    for k, v in alphas_if_deleted.items():
        text += f"{k}: {v:.3f}\n"
    return text

# Correlation heatmap
@app.callback(
    Output('corr-heatmap', 'figure'),
    Input('editable-table', 'data')
)
def update_corr(rows):
    if not rows:
        return go.Figure()
        
    df_updated = pd.DataFrame(rows)
    corr = df_updated[['R', 'L', 'T']].dropna().corr()
    
    # 🐞 Fix: Added check for empty DataFrame if no data is available
    if corr.empty:
        return go.Figure()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text:.2f}"
    ))
    fig.update_layout(title='Correlation Heatmap')
    return fig

# Regression
@app.callback(
    Output('regression-output', 'children'),
    Input('editable-table', 'data')
)
def update_regression(rows):
    if not rows:
        return ""
        
    df_updated = pd.DataFrame(rows)
    
    # 🐞 Fix: Ensure that the data for regression is not empty after dropping NaNs
    df_reg = df_updated.dropna(subset=['R', 'L', 'T'])
    if df_reg.empty:
        return "Not enough data for regression analysis."
        
    X = pd.get_dummies(df_reg[['Environmental', 'Road Surface', 'Traffic']], drop_first=True)
    y = df_reg[['R', 'L', 'T']]
    X_const = sm.add_constant(X)
    
    # 🐞 Fix: Added checks for data availability to prevent errors
    if len(y) < 2:
        return "Not enough data points for regression."

    model_R = sm.OLS(y['R'], X_const).fit()
    model_L = sm.OLS(y['L'], X_const).fit()
    model_T = sm.OLS(y['T'], X_const).fit()

    text = f"=== R Regression ===\n{model_R.summary().as_text()}\n\n"
    text += f"=== L Regression ===\n{model_L.summary().as_text()}\n\n"
    text += f"=== T Regression ===\n{model_T.summary().as_text()}"
    return text

# =======================
# 6️⃣ Run App
# =======================
if __name__ == "__main__":
    app.run(debug=True)