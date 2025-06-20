# chatbot/knowledge_base.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st

@st.cache_resource
def load_chat_models():
    bi_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return bi_model, cross_model

@st.cache_resource
def load_knowledge_base():
    try:
        with open("datasets/text/knowledge_data_uz.json", "r", encoding="utf-8") as f:
            kb = json.load(f)
        questions = [item["question"] for item in kb]
        answers = [item["answer"] for item in kb]
        contexts = [item.get("context", "") for item in kb]
        return kb, questions, answers, contexts
    except FileNotFoundError:
        st.error("Baza fayli topilmadi: datasets/text/knowledge_data_uz.json")
        st.stop()

@st.cache_resource
def build_faiss_index():
    bi_model, _ = load_chat_models()
    kb_data, kb_questions, kb_answers, kb_contexts = load_knowledge_base()
    kb_qa = [f"{q} [SEP] {ctx}" for q, ctx in zip(kb_questions, kb_contexts)]
    embeds = bi_model.encode(kb_qa, convert_to_tensor=False)
    embeds = np.array(embeds).astype("float32")
    embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)
    return index, embeds, kb_questions, kb_answers, kb_contexts

def get_chat_answer(query, top_k=5, threshold=0.5):
    bi_model, cross_model = load_chat_models()
    index, embeds, questions, answers, contexts = build_faiss_index()
    
    query_embed = bi_model.encode([query], convert_to_tensor=False)[0]
    query_embed = query_embed / np.linalg.norm(query_embed)
    query_embed = query_embed.astype('float32')
    
    distances, indices = index.search(np.array([query_embed]), k=top_k)
    candidates = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i] >= threshold:
            candidates.append({
                "question": questions[idx],
                "answer": answers[idx],
                "context": contexts[idx],
                "score": distances[0][i]
            })
    
    if not candidates:
        return {"question": "", "answer": "Hech qanday mos javob topilmadi", "context": "", "score": 0}
    
    cross_inputs = [(query, f"{c['question']} [SEP] {c['context']}") for c in candidates]
    cross_scores = cross_model.predict(cross_inputs)
    for i, s in enumerate(cross_scores):
        candidates[i]["cross_score"] = s
    return sorted(candidates, key=lambda x: x["cross_score"], reverse=True)[0]