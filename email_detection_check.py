import joblib
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
vector=joblib.load('Email_detecion_vector.joblib')
load=joblib.load("Email_detection.joblib")
st.title("FACT CHECKER 📰")
st.header("Write down news here to check: ")
unseen_text=st.text_input("")

if unseen_text:
 unseen_text=unseen_text.lower() 
 unseen_text=re.sub('<br />','',unseen_text) 
 unseen_text=re.sub(',','',unseen_text) 
 unseen_text=vector.transform([unseen_text]) 
 prediction=load.predict(unseen_text) 
 st.write(f"**Yeah, this news is {prediction[0]}**")

else:
 pass