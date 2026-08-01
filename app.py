import os
import json
import time
import cv2
import pandas as pd
from datetime import datetime
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from model import get_predictor_model
from utils import load_settings
from camera_manager import CameraManager, get_camera_sources
from db_analytics import init_db, get_alerts_dataframe

st.set_page_config(page_title="AI Surveillance System", layout="wide")


init_db()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


config = load_config()
settings = load_settings()

@st.cache_resource
def _load_models():
    return get_predictor_model()


with st.spinner("Инициализация (пожалуйста, подождите)..."):
    model = None
    try:
        model = _load_models()
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")


@st.cache_resource
def _init_camera_manager():
    cameras = get_camera_sources(config, settings)
    if not cameras:
        return None
    return CameraManager(cameras, model, settings, config)


camera_manager = _init_camera_manager()


st.sidebar.title("Меню управления")
page = st.sidebar.radio(
    "Переключение режима", ["Мониторинг", "Аналитика и Дашборды"]
)

if page == "Мониторинг":
    st.title("Training—please wait.")

    if camera_manager is None:
        st.error(
            "Не найдено ни одной камеры. Проверьте config.json (rtsp_cameras / webcam_index)."
        )
    else:
        camera_names = camera_manager.get_camera_names()
        cols_per_row = 3
        n_cameras = len(camera_names)

        for row_start in range(0, n_cameras, cols_per_row):
            row_names = camera_names[row_start : row_start + cols_per_row]
            cols = st.columns(len(row_names))

            for idx, name in enumerate(row_names):
                with cols[idx]:
                    stream = camera_manager.get_reader(name)
                    frame = stream.read() if stream else None
                    frame_age = (
                        time.time() - stream.last_frame_time if stream and getattr(stream, "last_frame_time", 0) else 0
                    )
                    status = camera_manager.get_status(name)

                    if frame is not None:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if "Норма" in status["text"]:
                            border_color = [0, 255, 0]
                        else:
                            border_color = [0, 0, 255]

                        h, w = rgb_frame.shape[:2]
                        border_rgb = cv2.copyMakeBorder(
                            rgb_frame, 6, 6, 6, 6,
                            cv2.BORDER_CONSTANT, value=border_color,
                        )

                        st.image(
                            border_rgb,
                            caption=f"**{name}** — {status['text']}",
                            width="stretch",
                            output_format="JPEG",
                        )
                    else:
                        st.warning(f"**{name}**: ожидание сигнала")
elif page == "Аналитика и Дашборды":
    st.title("Аналитика зафиксированных инцидентов")

    df = get_alerts_dataframe()

    if df.empty:
        st.info("В базе данных пока нет зафиксированных инцидентов.")
    else:
        # Гарантируем, что timestamp — это datetime, иначе resample() упадёт
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Всего инцидентов", len(df))
        col2.metric(
            "Самый частый тип",
            df["event_type"].mode()[0] if not df["event_type"].empty else "N/A",
        )
        col3.metric(
            "Средняя уверенность", f"{df['confidence'].mean() * 100:.1f}%"
        )

        st.markdown("---")

        st.subheader("Распределение инцидентов по классам")
        event_counts = df["event_type"].value_counts()
        st.bar_chart(event_counts)

        st.subheader("Динамика инцидентов во времени")
        df_time = (
            df.set_index("timestamp")
            .resample("h")
            .size()
            .reset_index(name="count")
        )
        st.line_chart(df_time, x="timestamp", y="count")

        st.subheader("Журнал всех алертов")
        st.dataframe(df, width="stretch")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Скачать отчёт (CSV)",
            data=csv,
            file_name=f"incidents_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )