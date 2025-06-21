from data.translations import translations

def get_text(key, lang=None):
    import streamlit as st
    if lang is None:
        lang = st.session_state.language
    return translations[lang].get(key, key)