import sys
import os
import subprocess
import requests
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from typing import Tuple
import spacy

env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)
import logging
logging.info("hello")
import os
logging.info(os.listdir())

if not spacy.util.is_package("pl_core_news_sm"):
    command = "python3 -m spacy download pl_core_news_sm"
    subprocess.run(command, shell=True)


from src.model_deployment.utils.nlp_utils import process_tokenize_input, get_ort_session, run_inference


st.set_page_config(
    page_title="Polish Hate-Speech Detection",
    page_icon="🚫",
    layout="wide"
)


def download_model():
    file_url = os.getenv("DROPBOX_PATH")

    direct_link = file_url.replace("www.dropbox.com", "dl.dropboxusercontent.com")

    local_file_path = os.getenv("MODEL_PATH")

    response = requests.get(direct_link, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    if response.status_code == 200:
        with open(local_file_path, "wb") as file, tqdm(
            desc=local_file_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))
        print("File downloaded successfully.")
    else:
        print("Failed to download the file.")

download_model()

# Function to classify text
def classify_text(input_text: str) -> Tuple[int, float]:
    ort_session = get_ort_session()
    ort_inputs = process_tokenize_input([input_text, ], preprocess=True)
    output_classes, confidences, _ = run_inference(ort_session, ort_inputs)
    return output_classes[0], round(confidences[0], 2)


def main():
    # Set app title and description
    st.title("Polish Hate-Speech Detection Model")
    st.write("This app classifies text as hate speech or not.")

    # Add repository link
    st.sidebar.write("[Repository](https://dagshub.com/a-s-gorski/polish-hatespeech-sentiment-classification)")

    # Add input text box
    input_text = st.text_area("Input Text:", "")

    if st.button("Classify"):
        if input_text:
            predicted_class, confidence = classify_text(input_text)
            if predicted_class == 1:
                st.error("Hate Speech Detected!")
            else:
                st.success("No Hate Speech Detected.")
            st.info(f"Model Confidence: {confidence}%")
        else:
            st.warning("Please enter some text to classify.")


if __name__ == "__main__":
    main()
