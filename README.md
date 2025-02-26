# RAG App with Ollama DeepSeek
This project implements a Retrieval-Augmented Generation (RAG) system using Streamlit, Ollama DeepSeek, and FAISS for efficient similarity search. Users can upload a dataset (CSV or JSON), ask questions, and get answers generated by the Ollama DeepSeek model based on the retrieved context from the dataset.

## Features
Dataset Upload: Upload CSV or JSON files containing text data.

Retrieval System: Use FAISS to build a fast and efficient similarity search index.

Question Answering: Generate answers using the Ollama DeepSeek model based on the retrieved context.

User-Friendly Interface: Built with Streamlit for an intuitive and interactive experience.

## Technologies Used
Streamlit: For building the web app interface.

Ollama DeepSeek: For generating answers to user queries.

FAISS: For efficient similarity search and retrieval.

Sentence Transformers: For generating embeddings from text data.

Pandas: For dataset manipulation and preprocessing.

NumPy: For numerical operations.

## Installation
Prerequisites
Python 3.8 or higher.

Ollama installed and running locally (with the DeepSeek model downloaded).


ollama pull deepseek-r1:latest
Run the Streamlit app:

streamlit run main.py
Open your browser and navigate to http://localhost:8501.

## Usage
Upload a Dataset:

Click on the "Upload your dataset" button and select a CSV or JSON file.

Ensure the dataset contains a text column for retrieval.

Select Text Column:

Choose the column from the dataset that contains the text data.

Build Retrieval System:

Click the "Build Retrieval System" button to create embeddings and build the FAISS index.

Ask Questions:

Enter your query in the text input box.

The app will retrieve relevant contexts from the dataset and generate an answer using the Ollama DeepSeek model.
