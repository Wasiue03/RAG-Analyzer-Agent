import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama


def load_dataset(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        data = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or JSON file.")
        return None
    return data


def create_faiss_index(data, text_column):
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    texts = data[text_column].astype(str).tolist()  
    embeddings = model.encode(texts, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  
    index.add(np.array(embeddings).astype('float32'))
    return index, texts, model


def retrieve_context(query, index, texts, model, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    relevant_contexts = [texts[i] for i in indices[0]]
    return relevant_contexts


def generate_answer(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = ollama.generate(model='deepseek-r1:latest', prompt=prompt)
    return response['response']


st.title("RAG App with Ollama DeepSeek")
st.write("Upload a dataset and ask questions!")


uploaded_file = st.file_uploader("Upload your dataset (CSV or JSON)", type=["csv", "json"])
if uploaded_file:
    data = load_dataset(uploaded_file)
    if data is not None:
        st.write("Dataset Preview:")
        st.write(data.head())

        text_column = st.selectbox("Select the text column for retrieval", data.columns)
        if st.button("Build Retrieval System"):
            with st.spinner("Creating embeddings and indexing..."):
                index, texts, model = create_faiss_index(data, text_column)
                st.session_state['index'] = index
                st.session_state['texts'] = texts
                st.session_state['model'] = model
                st.success("Retrieval system is ready!")

if 'index' in st.session_state:
    query = st.text_input("Enter your query:")
    if query:
        with st.spinner("Searching for relevant context..."):
            relevant_contexts = retrieve_context(query, st.session_state['index'], st.session_state['texts'], st.session_state['model'])
            st.write("Retrieved Contexts:")
            for i, context in enumerate(relevant_contexts):
                st.write(f"{i+1}. {context}")

        with st.spinner("Generating answer..."):
            combined_context = "\n".join(relevant_contexts)
            answer = generate_answer(query, combined_context)
            st.write("Generated Answer:")
            st.write(answer)