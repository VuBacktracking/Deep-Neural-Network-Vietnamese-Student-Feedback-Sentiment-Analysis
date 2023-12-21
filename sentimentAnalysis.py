import streamlit as st
import streamlit.components.v1 as components
from utils.classify import feedbackSentimentAnalysis
from PIL import Image
from keras.models import load_model
import pickle
from underthesea import word_tokenize
from utils.preprocessing import remove_punctuation
import numpy as np
import matplotlib.pyplot as plt

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

def pad_sequence_sentence(sentence):
    array = sent2vec(sentence)
    arr_seq_len = array.shape[0]
    sequence_length_difference = DESIRED_SEQUENCE_LENGTH - arr_seq_len
        
    pad = np.zeros(shape=(sequence_length_difference, 200))

    array = np.array(np.concatenate([array, pad]))
    array = np.expand_dims(array, axis=0)
    return array

def draw_radar_chart(percentages):
    num_categories = len(percentages)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

    # Convert percentages to values in the range [0, 1]
    values = [percentage * 100 for percentage in percentages]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles + angles[:1], values + values[:1], 'o-', color='b', alpha=0.5)

    # Fill the area under the radar chart
    ax.fill(angles + angles[:1], values + values[:1], color='b', alpha=0.2)

    # Set labels for each category
    ax.set_thetagrids(np.degrees(angles), ['Tiêu cực', 'Trung lập', 'Tích cực'])

    # Set the axis limit
    ax.set_ylim(0, 100)
    
    plt.title('Polarity')
    return fig


def getSentiments(userText):
    array = pad_sequence_sentence(userText)
    percentages = model.predict(array)
    status_label = np.argmax(percentages > 0.5)
    status = feedbackSentimentAnalysis(status_label)
    if(status=="Tích Cực"):
        image = Image.open('images/forapp/positive.png')
    elif(status=="Tiêu Cực"):
        image = Image.open('images/forapp/negative.png')
    else:
        image = Image.open('images/forapp/neutral.png')
    print(status_label)
    # st.image(image, caption=status)
    
    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Display the image in the first column
    col1.image(image, caption=status)

    # Display the radar chart in the second column
    radar_chart = draw_radar_chart(list(percentages[0]))
    col2.pyplot(radar_chart)
        

# def renderPage():
#     st.title("Vietnamese Student Feedback Sentiment Analysis 😊😐😕")
#     components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 5px" /> """)
#     # st.markdown("### User Input Text Analysis")
#     st.subheader("Phân tích Feedback của học sinh.")
#     st.text("Phân tích feedback của học sinh, sinh viên và trả về cảm xúc của nó")
#     st.text("")
#     userText = st.text_input('User Input', placeholder='Input text HERE')
#     if st.button('Analyze'):
#         if(userText!="" and type is not None):
#             st.components.v1.html("""
#                                 <h3 style="color: #0284c7; font-family: Source Sans Pro, sans-serif; font-size: 28px; margin-bottom: 10px; margin-top: 50px;">Result</h3>
#                                 """, height=100)
#             getSentiments(userText)

def renderPage():
    st.title("Vietnamese Student Feedback Sentiment Analysis 😊😐😕")
    
    # Add images using HTML and CSS
    st.markdown("""
        <style>
            .image {
                width: 24px;
                height: 24px;
                margin-right: 10px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    decor = Image.open("images/forapp/decoration.png")
    st.image(decor)
    
    st.subheader("Phân tích Feedback của học sinh.")
    st.text("Phân tích feedback của học sinh, sinh viên và trả về cảm xúc của nó")
    st.text("")
    userText = st.text_input('User Input', placeholder='Input text HERE')
    
    if st.button('Analyze'):
        if userText != "" and type is not None:
            st.components.v1.html("""
                <h3 style="color: #0284c7; 
                            font-family: Source Sans Pro, sans-serif; 
                            font-size: 28px; 
                            margin-bottom: 10px; 
                            margin-top: 50px;">
                    Result
                </h3>
            """, height=100)

            # Assuming getSentiments is a function you have defined elsewhere
            getSentiments(userText)
