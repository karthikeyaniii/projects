import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model


scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('model.keras')


label_encoders = {
    'workclass': LabelEncoder(),
    'education': LabelEncoder(),
    'marital-status': LabelEncoder(),
    'occupation': LabelEncoder(),
    'relationship': LabelEncoder(),
    'race': LabelEncoder(),
    'sex': LabelEncoder(),
    'native-country': LabelEncoder()
}


label_encoders['workclass'].fit(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
label_encoders['education'].fit(['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
label_encoders['marital-status'].fit(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
label_encoders['occupation'].fit(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
label_encoders['relationship'].fit(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
label_encoders['race'].fit(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
label_encoders['sex'].fit(['Female', 'Male'])
label_encoders['native-country'].fit(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])


def preprocess_input(input_data):
    for column in label_encoders:
        input_data[column] = label_encoders[column].transform([input_data[column]])[0]
    
    input_df = pd.DataFrame([input_data])
    input_df_scaled = scaler.transform(input_df)
    
    return input_df_scaled


st.title("Income Prediction App")
st.write("Enter the details below to predict whether the income is above or below $50K.")
image_path = r'C:\Users\HP\Desktop\New folder\Gemini_Generated_Image_494b2s494b2s494b.jpeg'
st.image(image_path, caption='Gemini Generated Image', use_column_width=True)


age = st.number_input("Age", min_value=17, max_value=90, value=30)
workclass = st.selectbox("Workclass", options=label_encoders['workclass'].classes_)
education = st.selectbox("Education", options=label_encoders['education'].classes_)
marital_status = st.selectbox("Marital Status", options=label_encoders['marital-status'].classes_)
occupation = st.selectbox("Occupation", options=label_encoders['occupation'].classes_)
relationship = st.selectbox("Relationship", options=label_encoders['relationship'].classes_)
race = st.selectbox("Race", options=label_encoders['race'].classes_)
sex = st.selectbox("Sex", options=label_encoders['sex'].classes_)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
native_country = st.selectbox("Native Country", options=label_encoders['native-country'].classes_)

input_data = {
    'age': age,
    'workclass': workclass,
    'education': education,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'sex': sex,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}

if st.button("Predict"):
    input_df_scaled = preprocess_input(input_data)
    prediction = model.predict(input_df_scaled)
    prediction_class = (prediction > 0.5).astype(int)[0][0]

    
    st.write(f"The predicted income class is: {'>50K' if prediction_class == 1 else '<=50K'}")
