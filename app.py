import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#load in model
pickle_in = open('model.pkl', 'rb') 
model = pickle.load(pickle_in)
pickle_in = open('vectorizer.pkl', 'rb')
cv = pickle.load(pickle_in)

def prediction(message):
    message_count = cv.transform([message])
    prediction = model.predict(message_count)

    return 'spam' if prediction[0] == 1 else 'ham'


def main():
    st.title('Spam Prediction')
    html_temp = """ 
        <div style ="background-color:yellow;padding:13px"> 
        <h1 style ="color:black;text-align:center;">Streamlit Spam Detector ML App </h1> 
        </div> 
    """

    st.markdown(html_temp, unsafe_allow_html = True) 

    message = st.text_input('Spam or Ham', 'Type here')

    if (st.button('Predict')):
        result = prediction(message)
        st.success('The message is {}'.format(result))

if __name__ == '__main__':
    main()