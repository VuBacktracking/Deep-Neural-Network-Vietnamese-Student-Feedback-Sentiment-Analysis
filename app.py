import sentimentAnalysis as sentimentAnalysis
import corpusAnalysis as corpusAnalysis
import streamlit as st
import sidebar as sidebar

page = sidebar.show()

if page=="Corpus Analysis":
    corpusAnalysis.renderPage()
if page=="Sentiment Analysis":
    sentimentAnalysis.renderPage()