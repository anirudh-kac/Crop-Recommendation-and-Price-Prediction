import streamlit as st
import pickle
import numpy as np

recommender = {}
with open("recommender.pickle", "rb")  as file:
    recommender = pickle.load(file)

rf = recommender["model"]
le = recommender["label_encoder"]
scaler = recommender["standard_scaler"]

with st.sidebar:
    N = float(st.slider("Nitrogen : (N): " , 0 ,140,50))
    P = float(st.slider("Phosphorus : (P) : " , 0 ,140,50))
    K = float(st.slider("Pottasium (K) : " , 0 ,140,50))
    temperature = float(st.slider("Temperature : (Celcius) : " , 0 , 50,25))
    humidity = float(st.slider("Humidity : (mm) " , 0, 100,70))
    ph = float(st.slider("pH : " , 0 , 14 , 6))
    rainfall = float(st.slider("Rainfall (mm) : " , 0,300,100))


st.header("Crop Recommendation")

instance = np.array([N,P,K,temperature,humidity,ph,rainfall]).reshape(1,-1)
instance = scaler.transform(instance)
prediction = rf.predict(instance)
recommendation = le.inverse_transform(prediction)
st.subheader(recommendation[0].upper())
