import streamlit as st
import numpy as np
import pandas as pd
# For download buttons
from functionforDownloadButtons import download_button
import os
import json
import re
import pprint
import pickle
import data
import altair as alt

st.set_page_config(
    page_title="Test Question Generator",
    page_icon="📜",
)

st.title("📜 Test Question Generator")
st.header("")

with st.expander("ℹ️ - Quick Start", expanded=True):
    st.write(
        """     
-   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
-   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
	    """
    )
    st.markdown("")

st.markdown("")
st.markdown("## 📌 Edit here")

with st.sidebar:
    st.markdown("# Upload files with the type")
    st.markdown("- PDF format")
    st.markdown("- Microsoft Word Document")
    st.markdown("- Text files")

    with st.form(key="my_form"):
        uploaded_file = st.file_uploader(label="Choose files", type=["pdf", "docx", "txt"], accept_multiple_files=True, help="None")

        # Every form must have a submit button.
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            #upload data
            st.success('Uploaded Successfully!')

with st.form(key="download"):

    txt = st.text_area("", data.s)
    res = txt.split()
    if res:
        st.info(
            "Your text contains "
            + str(res.count('Q:'))
            + " Questions&Answers."
            + " 😊"
        )
    submit_button = st.form_submit_button(label="✨ Get me the data!")
    if submit_button:
        st.success("Downloaded and copy pasted above!")

st.markdown("## 🎈 Check & download results")

c1, c2, c3, cLast = st.columns([1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(txt, "data.csv", "📥 Download (.csv)")
with c2:
    CSVButton2 = download_button(txt, "data.txt", "📥 Download (.txt)")
with c3:
    CSVButton2 = download_button(txt, "data.json", "📥 Download (.json)")

# Evaluation results here

with st.expander("ℹ️ - Evaluation Metrics", expanded=True):
    st.write("To be able to make a statement about the quality of results a question-answering pipeline or any other pipeline in haystack produces, it is important to evaluate it. Furthermore, evaluation allows determining which components of the pipeline can be improved. The results of the evaluation can be saved as CSV files, which contain all the information to calculate additional metrics later on or inspect individual predictions")
    st.markdown("")

st.markdown("""Before training the FARMmodel (reader):
- Retriever - Recall (single relevant document): 0.8
- Retriever - Recall (multiple relevant documents): 0.8
- Retriever - Mean Reciprocal Rank: 0.6247619047619047
- Retriever - Precision: 0.16571428571428576
- Retriever - Mean Average Precision: 0.6257142857142857
- Reader - F1-Score: 0.35396155963876497
- Reader - Exact Match: 0.05714285714285714

After training the FARM model (reader)
- Retriever - Recall (single relevant document): 0.8
- Retriever - Recall (multiple relevant documents): 0.8
- Retriever - Mean Reciprocal Rank: 0.6247619047619047
- Retriever - Precision: 0.16571428571428576
- Retriever - Mean Average Precision: 0.6257142857142857
- Reader - F1-Score: 0.5805476872540989
- Reader - Exact Match: 0.4

#### Overall Score
- 0.61671585""")

source = pd.DataFrame({
    'Score': [0.5805476872540989*100, 0.4*100, 0.61671585*100],
    'Metrics': ['F1-Score', 'Exact Match', 'Overall Score']
    })

bar_chart = alt.Chart(source).mark_bar().encode(
    y='Score',
    x='Metrics',
)

st.altair_chart(bar_chart, use_container_width=True)
