import streamlit as st
import json
import requests
import pandas as pd


st.title("Iris prediction")

st.write("")
st.write("Enter the flower parameters you want to predict, take the reference unseen data below")
data = pd.read_csv("./test_data.csv")
st.write(data)

sepal_length = st.text_input("Sepal length")
sepal_width = st.text_input("Sepal width")
petal_length = st.text_input("Petal length")
petal_width = st.text_input("Petal width")

inputs = {"sepal_length" : sepal_length, 
          "sepal_width" : sepal_width, 
          "petal_length" : petal_length,
          "petal_width" : petal_width}

if st.button('Prediction'):
    res = requests.post(url = "http://backend:8000/prediction", data=json.dumps(inputs))
    st.subheader("Predicted Results:")
    mystring = res.text.replace('"', "").replace("|", "\n")
    print("---")
    print(mystring)
    st.write(mystring)
    
