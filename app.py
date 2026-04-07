import matplotlib
matplotlib.use('Agg')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Linear Regression Visualizer", layout="wide")

st.title("📈 Linear Regression Visualizer")

st.write("App is running ✅")

# Sidebar controls
st.sidebar.header("Controls")

n_points = st.sidebar.slider("Number of data points", 10, 200, 50)
noise = st.sidebar.slider("Noise level", 0.0, 50.0, 10.0)
slope = st.sidebar.slider("True slope", -10.0, 10.0, 2.0)
intercept = st.sidebar.slider("True intercept", -50.0, 50.0, 5.0)

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(n_points, 1) * 100
true_y = slope * X + intercept
y = true_y + np.random.randn(n_points, 1) * noise

# Train model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Display model parameters
st.subheader("Model Parameters")
st.write(f"Estimated Slope: {model.coef_[0][0]:.2f}")
st.write(f"Estimated Intercept: {model.intercept_[0]:.2f}")

# Sort values for proper line plotting
sorted_idx = X.flatten().argsort()
X_sorted = X[sorted_idx]
pred_sorted = predictions[sorted_idx]

# Plot
fig, ax = plt.subplots()
ax.scatter(X, y, label="Data Points")
ax.plot(X_sorted, pred_sorted, linewidth=2, label="Regression Line")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

st.pyplot(fig)

# Upload dataset
st.subheader("📂 Upload Your Own Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    columns = df.columns.tolist()

    x_col = st.selectbox("Select X column", columns)
    y_col = st.selectbox("Select Y column", columns)

    X_user = df[[x_col]].values
    y_user = df[[y_col]].values

    model_user = LinearRegression()
    model_user.fit(X_user, y_user)

    preds_user = model_user.predict(X_user)

    st.write(f"Slope: {model_user.coef_[0][0]:.2f}")
    st.write(f"Intercept: {model_user.intercept_[0]:.2f}")

    sorted_idx2 = X_user.flatten().argsort()
    X_user_sorted = X_user[sorted_idx2]
    preds_user_sorted = preds_user[sorted_idx2]

    fig2, ax2 = plt.subplots()
    ax2.scatter(X_user, y_user)
    ax2.plot(X_user_sorted, preds_user_sorted, linewidth=2)
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)

    st.pyplot(fig2)