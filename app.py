import streamlit as st
import pandas as pd
import joblib

model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('feature_names.pkl')


st.title("Heart Disease Prediction App")
st.markdown("This app predicts the likelihood of heart disease based on user input features.")

age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("chest pain type",["ATA","NAP","TA","ASY"])
resting_BP = st.number_input("resting blood pressure(mm Hg)", 80,200,120)
cholestrol = st.number_input("cholestrol (mg/dL)",100,600,200)
fasting_bs = st.selectbox("fasting blood sugar>120mg/dL",[0,1])
resting_ecg = st.selectbox("resting ECG",["normal","ST","LVH"])
max_HR = st.slider("max heart rate", 60,220,150)
exercise_angima = st.selectbox("exercise-induced Angima",["y","n"])
old_peak = st.slider("oldpeak (ST Depression)",0.0,6.0,1.0)
ST_slope = st.selectbox("ST slope",["up","flat","down"])


if st.button("predict"):
    raw_input = {
        "age": age,
        "sex": sex,
        "chest_pain": chest_pain,
        "resting_BP": resting_BP,
        "cholestrol": cholestrol,
        "fasting_bs": fasting_bs,
        "resting_ecg": resting_ecg,
        "max_HR": max_HR,
        "exercise_angima": exercise_angima,
        "old_peak": old_peak,
        "ST_slope": ST_slope


    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

    input_df= input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction ==1:
        st.error("High risk of heart disease")

    else:
        st.error("low risk of heart disease")