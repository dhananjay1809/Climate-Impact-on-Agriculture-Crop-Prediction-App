import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt


# 1. Load Data

@st.cache_data
def load_data():
    df = pd.read_csv("sample_climate_agri.csv")
 
    return df

df = load_data()

st.title("🌾 Climate Impact on Agriculture & Crop Prediction App")
st.image("images/image2.0.png")


st.write("""
This app can:
1. Predict the **most suitable crop** based on climate & soil data.
2. Estimate the **expected yield (tons/hectare)** for given conditions.
""")


# 2. Preprocessing for Crop Classification

X_crop = df.drop(["crop", "state", "yield_t_ha"], axis=1)
y_crop = df["crop"]

crop_encoder = LabelEncoder()
y_crop_encoded = crop_encoder.fit_transform(y_crop)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_crop, y_crop_encoded, test_size=0.2, random_state=42
)

crop_model = LogisticRegression(max_iter=500)
crop_model.fit(Xc_train, yc_train)


# 3. Preprocessing for Yield Regression

X_yield = df.drop(["yield_t_ha", "crop", "state"], axis=1)
y_yield = df["yield_t_ha"]

Xy_train, Xy_test, yy_train, yy_test = train_test_split(
    X_yield, y_yield, test_size=0.2, random_state=42
)

yield_model = LinearRegression()
yield_model.fit(Xy_train, yy_train)


# 4. User Input

st.subheader("Enter Climate & Soil Parameters:")

inputs = {}
for col in X_crop.columns:
    if col == "year":
        inputs[col] = st.number_input(f"{col}", min_value=2000, max_value=2100, value=2025)
    else:
        inputs[col] = st.number_input(f"{col}", value=float(df[col].mean()))

input_df = pd.DataFrame([inputs])


# 5. Prediction + Histograms

if st.button("Predict"):
    # Crop prediction
    crop_pred = crop_model.predict(input_df)[0]
    crop_name = crop_encoder.inverse_transform([crop_pred])[0]

    # Yield prediction
    yield_pred = yield_model.predict(input_df)[0]

    st.success(f"🌱 Recommended Crop: **{crop_name}**")
    st.info(f"📊 Expected Yield: **{yield_pred:.2f} tons/hectare**")


    # 6. Histograms

    st.subheader("📊 Data Distribution (Histograms)")


    numeric_cols = []
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        except:
            pass

    import matplotlib.pyplot as plt

    for col in numeric_cols:
        if col in ["year", "state", "crop"]:
            continue

        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=20, color="lightgrey", edgecolor="black")

        if col in inputs.keys():
            ax.axvline(inputs[col], color="red", linestyle="--", label="Your Input")

        if col == "yield_t_ha":
            ax.axvline(yield_pred, color="green", linestyle="--", label="Predicted Yield")

            

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()

        st.pyplot(fig)
