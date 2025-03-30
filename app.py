import streamlit as st
import pickle

import nltk 
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('stopwords')

# Initialize tokenizer and stemmer
tokenizer = TreebankWordTokenizer()
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)  # Use TreebankWordTokenizer instead of word_tokenize
    
    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric words
            y.append(i)
    
    text = y[:] # cloneing the list 
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))  # Apply stemming
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))

model = pickle.load(open('model.pkl','rb'))

st.title("SMS/Email Spam Classifier")

input_sms = st.text_area("Enter the Message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")