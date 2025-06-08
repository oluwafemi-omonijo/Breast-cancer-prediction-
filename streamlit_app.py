from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("breast_cancer_rf_model.pkl")

logo = Image.open("logo.png")
st.image(logo, width=150)

st.markdown("<h2 style='text-align: center;'>Breast Cancer Risk Prediction</h2>", unsafe_allow_html=True)

st.title("🩺 Breast Cancer Risk Prediction App")

st.markdown("Use this tool to estimate your breast cancer risk based on lifestyle and reproductive factors.")

age = st.slider('Age', 18, 90, 30)

st.markdown("### 🧮 BMI Calculator")
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=165.0)
bmi = round(weight / ((height / 100) ** 2), 1)
st.markdown(f"**Calculated BMI:** {bmi}")

menarche_age = st.slider('Age at First Menstruation', 8, 18, 12)
first_pregnancy_age = st.slider('Age at First Pregnancy', 12, 50, 25)
parity = st.slider('Number of Births (Parity)', 0, 10, 2)

family_history = st.selectbox("Family History of Breast Cancer", ["No", "Yes"])
oral_contraceptive_use = st.selectbox("Used Oral Contraceptives?", ["No", "Yes"])
hormone_therapy_use = st.selectbox("Used Hormone Therapy?", ["No", "Yes"])
urban_residence = st.selectbox("Lives in Urban Area?", ["No", "Yes"])
exposure_to_pollution = st.selectbox("Exposed to Pollution?", ["No", "Yes"])
breastfeeding_history = st.selectbox("Breastfed Children?", ["No", "Yes"])

smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
alcohol_intake = st.slider("Alcohol Intake (units/week)", 0.0, 10.0, 1.0)
physical_activity_level = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
education_level = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])

binary_map = {'Yes': 1, 'No': 0}
multi_map = {
    "Never": 0, "Former": 1, "Current": 2,
    "Low": 0, "Moderate": 1, "High": 2,
    "None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3
}

input_data = np.array([[
    age, bmi, menarche_age, first_pregnancy_age, parity,
    binary_map[family_history],
    binary_map[oral_contraceptive_use],
    binary_map[hormone_therapy_use],
    multi_map[smoking_status],
    alcohol_intake,
    multi_map[physical_activity_level],
    multi_map[education_level],
    binary_map[urban_residence],
    binary_map[exposure_to_pollution],
    binary_map[breastfeeding_history]
]])

if st.button("🧠 Predict Risk"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Predicted Risk: **High** for Breast Cancer")

        labels = [
            'Age', 'BMI', 'Menarche Age', 'First Pregnancy Age', 'Parity',
            'Family History', 'Oral Contraceptives', 'Hormone Therapy',
            'Smoking', 'Alcohol Intake', 'Physical Activity',
            'Education', 'Urban Residence', 'Pollution Exposure', 'Breastfeeding'
        ]
        values = input_data.flatten()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(labels, values, color='salmon')
        ax.set_title("Your Risk Factor Contributions", fontsize=14)
        ax.set_xlabel("Reported Level / Presence of Risk", fontsize=12)
        plt.tight_layout()

        st.pyplot(fig)
        st.markdown("""
            ### 🩺 Recommended Actions:
            - Please consult a healthcare professional for proper screening (e.g., mammogram).
            - Consider lifestyle changes like reducing alcohol, increasing physical activity.
            - Maintain regular self-breast examinations.
            - If you're over 40, follow up with a clinical breast exam and imaging.
            """)

        labels = [
            'Age', 'BMI', 'Menarche Age', 'First Pregnancy Age', 'Parity',
            'Family History', 'Oral Contraceptives', 'Hormone Therapy',
            'Smoking', 'Alcohol Intake', 'Physical Activity',
            'Education', 'Urban Residence', 'Pollution Exposure', 'Breastfeeding'
        ]
        colors = []
        for i, val in enumerate(values):
            if labels[i] in ['Age', 'BMI', 'Family History', 'Hormone Therapy', 'Smoking',
                             'Pollution Exposure'] and val >= 1:
                colors.append('red')  # High concern
            elif val >= 0.5:
                colors.append('orange')  # Moderate
            else:
                colors.append('green')  # Low concern

        # Plot
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(labels, values, color=colors)
        ax.set_title("Your Personal Risk Profile", fontsize=14)
        ax.set_xlabel("Relative Level / Risk Presence", fontsize=12)
        plt.tight_layout()

        st.pyplot(fig)
    else:
        st.success("✅ Predicted Risk: **Low** for Breast Cancer")
        st.markdown("""
            ### 👏 You're Doing Well:
            - Continue regular health checkups and maintain healthy habits.
            - Stay physically active and eat a balanced diet.
            - Know your family history and report changes to your doctor.
            - Do routine breast self-exams and follow screening guidelines based on your age.
            """)

        with st.expander("ℹ️ About the Dataset"):
            st.markdown("""
            ### 📊 Simulated Breast Cancer Risk Dataset by Oluwafemi

            This app is powered by a **simulated dataset** designed to reflect real-world breast cancer risk factors based on:

            - **Clinical guidelines**
            - **Epidemiological research**
            - **Risk prediction frameworks**

            Key features included:
            - Demographics (e.g., age, education)
            - Reproductive history (e.g., age at first pregnancy, parity)
            - Lifestyle (e.g., BMI, smoking, alcohol intake, physical activity)
            - Environmental exposures (e.g., air pollution, urban residence)
            - Medical history (e.g., hormone therapy, oral contraceptive use, family history)

            > ⚠️ **Disclaimer**: This tool is for **educational and research purposes only**. The dataset used does not represent real patient records and should not be used for clinical decision-making.
            """)











