# chatbot/search.py
from duckduckgo_search import DDGS
import streamlit as st

def web_search(query, max_results=5):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return [
                {
                    "title": r["title"],
                    "description": r["body"],
                    "url": r["href"]
                } for r in results if r.get("body") and r.get("href")
            ]
    except Exception as e:
        return [{"error": f"❌ Ошибка при поиске: {str(e)}"}]
