# ============================================================
# CHRONIC KIDNEY DISEASE PREDICTION WEB APP
# ============================================================

import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Small banner image to improve look-and-feel
def _render_banner():
    try:
        # create a horizontal gradient background
        w, h = 1200, 220
        grad = np.linspace(0, 1, w)
        grad = np.tile(grad, (h, 1))

        fig = plt.figure(figsize=(12, 2.2), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(grad, aspect='auto', cmap='plasma')

        # overlay title and subtitle
        ax.text(0.06, 0.52, "🫀", transform=ax.transAxes, fontsize=68, color="white", va="center", ha="center")
        ax.text(0.22, 0.62, "Chronic Kidney Disease Prediction System", transform=ax.transAxes, fontsize=26, fontweight='600', color="white", va="center", ha="left")
        ax.text(0.22, 0.34, "Fast, simple screening using a trained ML model", transform=ax.transAxes, fontsize=14, color="white", va="center", ha="left", alpha=0.95)

        ax.axis('off')

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        # `use_column_width` is deprecated; set an explicit width instead.
        st.image(buf, width=1000)
        plt.close(fig)
    except Exception:
        pass

_render_banner()

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

model = pickle.load(
    open("models/final_ckd_model.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

# ============================================================
# TITLE
# ============================================================

st.title("🩺 Chronic Kidney Disease Prediction System")

st.markdown(
        """
        <div style="background:linear-gradient(90deg,#3b82f6,#06b6d4);padding:18px;border-radius:12px;color:#fff;margin-bottom:18px">
            <h3 style="margin:0 0 6px 0;font-weight:600">This Machine Learning application predicts whether a patient is likely to have Chronic Kidney Disease (CKD)</h3>
            <p style="margin:0;opacity:0.95">Enter patient medical parameters below and click <strong>Predict CKD</strong> to obtain a prediction and associated probability.</p>
        </div>
        """,
        unsafe_allow_html=True,
)

# ============================================================
# INPUT SECTION
# ============================================================

st.header("Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=45
    )

    bp = st.number_input(
        "Blood Pressure",
        min_value=50,
        max_value=200,
        value=80
    )

    sg = st.number_input(
        "Specific Gravity",
        min_value=1.0,
        max_value=1.05,
        value=1.02
    )

    al = st.number_input(
        "Albumin",
        min_value=0,
        max_value=5,
        value=1
    )

    su = st.number_input(
        "Sugar",
        min_value=0,
        max_value=5,
        value=0
    )

    rbc = st.selectbox(
        "Red Blood Cells",
        [0, 1]
    )

    pc = st.selectbox(
        "Pus Cell",
        [0, 1]
    )

    pcc = st.selectbox(
        "Pus Cell Clumps",
        [0, 1]
    )

with col2:

    ba = st.selectbox(
        "Bacteria",
        [0, 1]
    )

    bgr = st.number_input(
        "Blood Glucose Random",
        min_value=50,
        max_value=500,
        value=120
    )

    bu = st.number_input(
        "Blood Urea",
        min_value=1,
        max_value=300,
        value=40
    )

    sc = st.number_input(
        "Serum Creatinine",
        min_value=0.1,
        max_value=20.0,
        value=1.2
    )

    sod = st.number_input(
        "Sodium",
        min_value=100,
        max_value=200,
        value=135
    )

    pot = st.number_input(
        "Potassium",
        min_value=1.0,
        max_value=10.0,
        value=4.5
    )

    hemo = st.number_input(
        "Hemoglobin",
        min_value=1.0,
        max_value=20.0,
        value=13.5
    )

    pcv = st.number_input(
        "Packed Cell Volume",
        min_value=1,
        max_value=60,
        value=40
    )

with col3:

    wc = st.number_input(
        "White Blood Cell Count",
        min_value=1000,
        max_value=30000,
        value=8000
    )

    rc = st.number_input(
        "Red Blood Cell Count",
        min_value=1.0,
        max_value=10.0,
        value=5.0
    )

    htn = st.selectbox(
        "Hypertension",
        [0, 1]
    )

    dm = st.selectbox(
        "Diabetes Mellitus",
        [0, 1]
    )

    cad = st.selectbox(
        "Coronary Artery Disease",
        [0, 1]
    )

    appet = st.selectbox(
        "Appetite",
        [0, 1]
    )

    pe = st.selectbox(
        "Pedal Edema",
        [0, 1]
    )

    ane = st.selectbox(
        "Anemia",
        [0, 1]
    )

# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("Predict CKD"):

    input_data = np.array([[
        age,
        bp,
        sg,
        al,
        su,
        rbc,
        pc,
        pcc,
        ba,
        bgr,
        bu,
        sc,
        sod,
        pot,
        hemo,
        pcv,
        wc,
        rc,
        htn,
        dm,
        cad,
        appet,
        pe,
        ane
    ]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    prediction_proba = model.predict_proba(
        scaled_data
    )

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error(
            "⚠️ Patient is likely to have Chronic Kidney Disease"
        )

    else:

        st.success(
            "✅ Patient is unlikely to have Chronic Kidney Disease"
        )

    st.write(
        "Prediction Probability:",
        prediction_proba
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
        """
        <div style="font-size:13px;color:#6b7280;line-height:1.4">
              <strong style="color:#111">Developed by</strong>:
              <ul style="margin:6px 0 8px 18px;color:#374151">
                 <li>Biplob Kumar</li>
                 <li>Deepak Pal</li>
                 <li>Sonam Gupta</li>
                 <li>Soumya Agnihotri</li>
                 <li>Zaid Ali</li>
              </ul>
              <div style="color:#6b7280;margin-bottom:6px">Affiliation: <strong style="color:#111">CSJM University, Kanpur (UIET)</strong></div>
              <div style="color:#6b7280">Supervision: Dr. Alok Kumar. Dataset: UCI Machine Learning Repository.</div>
        </div>
        """,
        unsafe_allow_html=True,
)