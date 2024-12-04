import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def transform_text(data):
    data = data.lower()
    data = nltk.word_tokenize(data)
    y = []
    for i in data:
        if i.isalnum():
            y.append(i)
    data = y[:]
    y.clear()
    for i in data:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    data = y[:]
    y.clear()
    for i in data:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open("model.pkl","rb"))

st.title("Email/SMS/Text spam classifier")
input_sms=st.text_area("Enter the msg:")
if st.button("Predict"):
    #preprossing
    transform_sms=transform_text(input_sms)

    #vectorize
    vector_input=tfidf.transform([transform_sms])

    #pridict
    result=model.predict(vector_input)[0]

    #display
    if result==1:
        st.header("Beware!!It's a Spam message")
    else:
        st.header("Non Spam")

