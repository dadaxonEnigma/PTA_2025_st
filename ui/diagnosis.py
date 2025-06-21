import streamlit as st
from PIL import Image
import io
from web_search.search import web_search
import config
from models.visualization import plot_probabilities, get_heatmap, get_filtered_map

def render_diagnosis(img, session, model, classes, treatment, get_text, config):
    with st.expander(get_text("image_view_expander"), expanded=True):
        st.image(img, caption=get_text("image_caption"), use_container_width=True)

    with st.spinner(get_text("processing_message")):
        pred_class, probs, top_idx = model_predict(img, session, model, classes)

    st.subheader(get_text("results_header"))
    with st.container():
        st.markdown(f"### {get_text('main_diagnosis')}")
        st.markdown(f"**{config.format_class_name(pred_class, st.session_state.language)}** — {probs[top_idx[0]]*100:.1f}%")
        st.markdown(f"**{get_text('description_label')}:** {treatment[pred_class]['description']}")

        if 'symptoms' in treatment[pred_class]:
            st.markdown(f"### {get_text('symptoms_label')}")
            st.markdown(treatment[pred_class]['symptoms'])

        st.markdown(f"### {get_text('treatment_label')}")
        st.markdown(treatment[pred_class].get('treatment', treatment[pred_class].get('recommendation', get_text('no_treatment'))))

        if 'prevention' in treatment[pred_class]:
            st.markdown(f"**{get_text('prevention_label')}:** {treatment[pred_class]['prevention']}")

    render_visualizations(img, model, probs, top_idx, classes, get_text)

    return pred_class, probs, top_idx


def model_predict(img, session, model, classes):
    from models.inference import predict_disease
    return predict_disease(img, session, model, classes)


def render_visualizations(img, model, probs, top_idx, classes, get_text):
    tabs = st.tabs(get_text("visualization_tabs"))

    # --- Таб 1: График вероятностей ---
    with tabs[0]:
        fig = plot_probabilities(probs, classes, top_idx)
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        st.download_button(
            label=get_text("download_graph"),
            data=buf.getvalue(),
            file_name="disease_probabilities.png",
            mime="image/png"
        )

    # --- Таб 2: Тепловая карта ---
    with tabs[1]:
        if st.button(get_text("generate_heatmap")):
            with st.spinner(get_text('processing_message')):
                heatmap_img = get_heatmap(img, model, top_idx[0])
                if heatmap_img:
                    st.image(heatmap_img, caption=get_text("heatmap_caption"), use_container_width=True)
                    buf = io.BytesIO()
                    heatmap_img.save(buf, format="JPEG")
                    st.download_button(
                        label=get_text("download_heatmap"),
                        data=buf.getvalue(),
                        file_name="heatmap.jpg",
                        mime="image/jpeg"
                    )

    # --- Таб 3: Частотная карта ---
    with tabs[2]:
        if st.button(get_text("generate_freq_map")):
            with st.spinner(get_text("processing_message")):
                filtered_img = get_filtered_map(img)
                st.image(filtered_img, caption=get_text("freq_map_caption"), use_container_width=True)
                buf = io.BytesIO()
                filtered_img.save(buf, format="JPEG")
                st.download_button(
                    label=get_text("download_freq_map"),
                    data=buf.getvalue(),
                    file_name="frequency_map.jpg",
                    mime="image/jpeg"
                )
    with tabs[3]:  # Веб-поиск
        st.markdown(f"### {get_text('web_search_header')}")
        if st.button(get_text("web_search_label")):
            if 'pred_class' not in st.session_state:
                st.warning("Сначала сделайте предсказание!")
            else:
                pred_class = st.session_state['pred_class']
                with st.spinner(get_text("chat_processing")):
                    web_results = web_search(f"{config.format_class_name(pred_class, st.session_state.language)} {get_text('treatment_search')}")
                    if web_results and "error" not in web_results[0]:
                        for result in web_results:
                            st.markdown(f"**{result['title']}**")
                            st.markdown(f"{result['description']}")
                            st.markdown(f"[{get_text('source_link')}]({result['url']})")
                            st.markdown("---")
                    else:
                        st.error(get_text("chat_web_error"))