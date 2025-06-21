import streamlit as st
import time
from utils.deepseek import query_deepseek, format_bot_response

def render_chat(pred_class, probs, treatment, api_key, get_text, config):
    st.markdown("---")
    st.subheader(get_text("chat_header"))

    if st.button(get_text("clear_chat_btn")):
        st.session_state.chat_history = []
        st.session_state.chat_context = ""
        st.rerun()

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["bot"])

    with st.form("chat_form"):
        user_input = st.text_input(get_text("chat_input_label"), key="chat_input")
        submitted = st.form_submit_button(get_text("chat_submit_button"))
        if submitted and user_input.strip():
            with st.spinner(get_text("chat_processing")):
                context = (
                    f"{get_text('diagnosed_disease')}: {config.format_class_name(pred_class, st.session_state.language)} "
                    f"({probs[0]*100:.1f}%). "
                    f"{get_text('recommended_treatment')}: {treatment.get('treatment', '')}"
                )
                st.session_state.chat_context += f"\n{context}"
                bot_response = query_deepseek(user_input, api_key, st.session_state.chat_context)
                formatted = format_bot_response(bot_response)
                st.session_state.chat_history.append({"user": user_input, "bot": formatted})

                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full_response = ""
                    for word in formatted.split():
                        full_response += word + " "
                        placeholder.markdown(full_response + "▌")
                        time.sleep(0.05)
                    placeholder.markdown(full_response)
