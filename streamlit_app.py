import os
import tempfile
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="筋トレフォームAI", page_icon="🏋️", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container {max-width: 850px; padding-top: 1.2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {padding: .65rem; border: 1px solid rgba(128,128,128,.2); border-radius: 12px;}
@media (max-width: 640px) {
 .block-container {padding-left: 1rem; padding-right: 1rem;}
 h1 {font-size: 1.8rem !important;}
 div[data-testid="stMetricValue"] {font-size: 1.45rem;}
}
</style>""", unsafe_allow_html=True)
MAX_UPLOAD_MB = 100


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
CONFIG = {
    "ベンチプレス": {"triples": [(11, 13, 15), (12, 14, 16)], "low": 105, "high": 145, "good_low": 105, "good_high": 145, "joint": "肘", "muscles": "CHEST / TRICEPS"},
    "スクワット": {"triples": [(23, 25, 27), (24, 26, 28)], "low": 120, "high": 155, "good_low": 110, "good_high": 155, "joint": "膝", "muscles": "QUADS / GLUTES"},
    "アームカール": {"triples": [(11, 13, 15), (12, 14, 16)], "low": 90, "high": 135, "good_low": 80, "good_high": 145, "joint": "肘", "muscles": "BICEPS"},
}

@st.cache_resource(show_spinner="姿勢推定モデルを準備中...")
def model_file():
    path = Path(tempfile.gettempdir()) / "form_pose_landmarker_lite.task"
    if not path.exists() or path.stat().st_size < 1000000:
        urllib.request.urlretrieve(MODEL_URL, path)
    return str(path)


def angle3(a, b, c):
    u, v = a - b, c - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-7:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1, 1))))


def reliable(lms, ids, threshold=0.40):
    return all(
        np.isfinite(lms[i].x) and np.isfinite(lms[i].y)
        and lms[i].visibility >= threshold
        for i in ids
    )


def xy(lms, i, w, h):
    return np.array([lms[i].x * w, lms[i].y * h], dtype=float)


def landmark_angle(lms, triple, w, h):
    if not reliable(lms, triple):
        return None
    return angle3(*(xy(lms, i, w, h) for i in triple))


def effort_level(angle, cfg, exercise):
    """動作角度に基づく可視化用の推定。実測の筋負荷ではない。"""
    if angle is None:
        return "UNKNOWN", (190, 190, 190)
    if exercise == "アームカール":
        if angle <= cfg["low"]:
            return "HIGH*", (40, 40, 245)
        if angle < cfg["high"]:
            return "MID*", (0, 220, 255)
        return "LOW*", (55, 215, 65)
    if angle <= cfg["low"]:
        return "HIGH*", (40, 40, 245)
    if angle < cfg["high"]:
        return "MID*", (0, 220, 255)
    return "LOW*", (55, 215, 65)


def stable_load_level(angle, cfg, previous=None, margin=8):
    """Hysteresis for display only; not a physiological load measurement."""
    if angle is None:
        return previous
    low, high = cfg["low"], cfg["high"]
    if previous is None:
        return "HIGH" if angle <= low else "MID" if angle < high else "LOW"
    if previous == "HIGH":
        return "MID" if angle > low + margin else "HIGH"
    if previous == "LOW":
        return "MID" if angle < high - margin else "LOW"
    # メイン画面
st.title("🏋️ 筋トレフォーム判定AI")
st.write("動画から筋トレフォームを解析するアプリです。")

exercise = st.selectbox(
    "筋トレ種目を選択",
    ["ベンチプレス", "スクワット", "アームカール"]
)

uploaded_file = st.file_uploader(
    "筋トレ動画をアップロード",
    type=["mp4", "mov", "m4v"]
)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("フォーム評価を開始"):
        st.info("画面表示は正常です。動画解析機能はこれから接続します。")
