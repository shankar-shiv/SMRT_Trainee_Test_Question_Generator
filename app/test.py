import streamlit as st
import numpy as np
from pandas import DataFrame
import seaborn as sns
# For download buttons
from functionforDownloadButtons import download_button
import os
import json
import re
import pprint
import pickle
import data

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

with st.expander("ℹ️ - Advanced Evaluation Metrics", expanded=True):
    st.write("As an advanced evaluation metric, semantic answer similarity (SAS) can be calculated. This metric takes into account whether the meaning of a predicted answer is similar to the annotated gold answer rather than just doing string comparison")
    st.markdown("")

