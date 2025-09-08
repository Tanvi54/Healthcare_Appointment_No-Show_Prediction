import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model + encoders
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("data/model/decision_tree_model.pkl")
    label_encoders = joblib.load("data/model/label_encoders.pkl")
    model_features = joblib.load("data/model/model_features.pkl")
    return model, label_encoders, model_features

model, label_encoders, model_features = load_artifacts()

# =========================
# Streamlit App
# =========================
st.title("🩺 Patient No-Show Prediction App")
st.write("Predict whether a patient will **Show** or **NoShow** based on appointment details.")

# --- Mode Selection ---
mode = st.radio("Choose Input Mode:", ["Single Patient", "Multiple Patients"])

# =========================
# SINGLE PATIENT INPUT (Final Compact UI)
# =========================
if mode == "Single Patient":
    st.markdown(
        """
        <style>
        /* Make form compact */
        div[data-testid="stVerticalBlock"] div[role="radiogroup"] {
            flex-direction: row;
        }
        .form-card {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
            margin-bottom: 20px;
        }
        .form-header {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }
        .form-header span {
            margin-left: 8px;
        }
        label {
            font-weight: 600 !important;
            font-size: 13px !important;
            margin-bottom: 2px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """<div class="form-card">
                <div class="form-header">📝 <span>Enter Patient Details</span></div>
            </div>""",
        unsafe_allow_html=True
    )

    # Layout with two columns
    user_input = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(model_features):
        if feature in label_encoders:
            options = list(label_encoders[feature].classes_)
            if i % 2 == 0:
                user_input[feature] = col1.selectbox(f"{feature}", options, key=feature)
            else:
                user_input[feature] = col2.selectbox(f"{feature}", options, key=feature)
        else:
            if i % 2 == 0:
                user_input[feature] = col1.number_input(f"{feature}", min_value=0.0, step=1.0, key=feature)
            else:
                user_input[feature] = col2.number_input(f"{feature}", min_value=0.0, step=1.0, key=feature)

    # Prediction button (centered)
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col = st.columns([1,2,1])[1]
    if btn_col.button("🔮 Predict (Single)"):
        input_df = pd.DataFrame([user_input])
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))
        input_df = input_df[model_features]

        pred = model.predict(input_df)[0]
        pred_label = "Show" if pred == 0 else "NoShow"

        st.success(f"✅ Prediction: Patient will **{pred_label}**")



# =========================
# MULTIPLE PATIENT INPUT
# =========================
else:
    st.subheader("Upload or Paste Multiple Patient Records")

    st.write("👉 Upload a CSV file with the same columns as used in training OR paste tabular data below.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    pasted_data = st.text_area("Or Paste CSV-like data (comma separated):")

    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif pasted_data.strip() != "":
        try:
            df = pd.read_csv(pd.compat.StringIO(pasted_data))
        except Exception:
            st.error("⚠️ Could not parse pasted data. Ensure it is valid CSV format.")

    if df is not None:
        st.write("📋 Preview of Input Data")
        st.dataframe(df.head())

        if st.button("🔮 Predict (Multiple)"):
            # Encode categorical
            for col, le in label_encoders.items():
                if col in df.columns:
                    df[col] = le.transform(df[col].astype(str))

            # Ensure correct order
            df = df[model_features]

            # Predictions
            preds = model.predict(df)
            df["Prediction"] = ["Show" if p == 0 else "NoShow" for p in preds]

            st.subheader("✅ Predictions for Patients")
            st.dataframe(df)

            # Download option
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
