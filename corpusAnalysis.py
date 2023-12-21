import streamlit as st
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objs as go
import numpy as np
import os

import pickle
from utils.preprocessing import readData, remove_punctuation
from utils.classify import feedbackSentimentAnalysis
from underthesea import word_tokenize
from keras.models import load_model
from copy import deepcopy

# Load Model
model = load_model("models/lstm_model.h5")

with open("utils/words_dict.pkl", "rb") as file:
    words = pickle.load(file)

DESIRED_SEQUENCE_LENGTH = 205

def tokenize_vietnamese_sentence(sentence):
    return word_tokenize(remove_punctuation(sentence.lower()))

def sent2vec(message, word_dict = words):
    tokens = tokenize_vietnamese_sentence(message)
    vectors = []
    
    for token in tokens:
        if token not in word_dict.keys():
            continue
        token_vector = word_dict[token]
        vectors.append(token_vector)
    return np.array(vectors, dtype=float)

def X_to_vectors(sentences):
    all_word_vector_sequences = []
    
    for message in sentences:
      message_as_vector_seq = sent2vec(message)
      if message_as_vector_seq.shape[0] == 0:
        message_as_vector_seq = np.zeros(shape=(1, 200))

      all_word_vector_sequences.append(message_as_vector_seq)
    
    return all_word_vector_sequences

def pad_sequences(X, desired_sequence_length=205):
  X_copy = deepcopy(X)

  for i, x in enumerate(X):
    x_seq_len = x.shape[0]
    sequence_length_difference = desired_sequence_length - x_seq_len
    
    pad = np.zeros(shape=(sequence_length_difference, 200))

    X_copy[i] = np.concatenate([x, pad])
  
  return np.array(X_copy).astype(float)

def predictions(file_path, model = model):
    sentences = readData(file_path)
    vectors = X_to_vectors(sentences)
    sequences = pad_sequences(vectors)
    predictions = (model.predict(sequences))
    
    predicted_labels = np.argmax(predictions, axis=1)
    
    sentiments = [feedbackSentimentAnalysis(label) for label in predicted_labels]
    df = pd.DataFrame({
        "feedback" : sentences,
        "sentiment" : sentiments
    })
    return df

def renderPage():
    st.title("Corpus Analysis")

    # File Upload
    uploaded_file = st.file_uploader("Browse Corpus", type=["csv", "txt"])

    if uploaded_file:
        # st.write("filename: ", uploaded_file.name)
        
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            # Read CSV file
            df = pd.read_csv(uploaded_file)

            # Display DataFrame
            st.set_page_config(layout="wide")
            st.subheader("DataFrame:")
            st.write(df.sample(50))

            # Choose columns for pie charts (fixed options)
            selected_columns = ["sentiments", "topics"]

            # Generate Pie Charts horizontally
            st.subheader("Pie Charts:")

            fig = sp.make_subplots(rows=1, cols=2, subplot_titles=[f"Pie Chart for {col}" for col in selected_columns],
                                   specs=[[{'type': 'domain'}, {'type': 'domain'}]])

            for i, col in enumerate(selected_columns, start=1):
                labels = df[col].value_counts().index
                values = df[col].value_counts().values
                trace = go.Pie(labels=labels, values=values, name=col)
                fig.add_trace(trace, row=1, col=i)
                
        elif file_extension == 'txt':
            path = os.path.join("Data/testForApp", uploaded_file.name)
            df = predictions(path)
        
            # Display DataFrame
            st.subheader("Predictions:")
            st.dataframe(df.sample(50), width=800)
            
            labels = df["sentiment"].value_counts().index
            values = df["sentiment"].value_counts().values
            trace = go.Pie(labels=labels, values=values, name="sentiment")
            
            # Create a layout (optional)
            layout = go.Layout(title="Sentiment Distribution")

            # Create a figure using the trace and layout
            fig = go.Figure(data=[trace], layout=layout)

        fig.update_layout(showlegend=True)

        st.plotly_chart(fig)