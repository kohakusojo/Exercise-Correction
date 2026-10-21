import csv
import io
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title='筋トレフォーム判定AI', page_icon='🏋️', layout='centered')
st.markdown('''<style>.block-container{max-width:850px;padding-top:1rem;padding-bottom:2rem}div[data-testid="stMetric"]{border:1px solid #9995;border-radius:12px;padding:10px}@media(max-width:640px){.block-container{padding-left:1rem;padding-right:1rem}h1{font-size:1.7rem!important}}</style>''', unsafe_allow_html=True)

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
MAX_MB = 100
MAX_SECONDS = 90
MAX_WIDTH = 640
FPS_LIMIT = 24
CONFIG = {
    'ベンチプレス': {'triples': ((11,13,15),(12,14,16)), 'down': 105, 'up': 150, 'ideal_low': 65, 'ideal_high': 115, 'joint':'肘', 'muscles':'大胸筋・上腕三頭筋', 'tip':'肘を左右均等に曲げ、肩甲骨を安定させましょう。'},
    'スクワット': {'triples': ((23,25,27),(24,26,28)), 'down': 110, 'up': 155, 'ideal_low': 65, 'ideal_high': 115, 'joint':'膝', 'muscles':'大腿四頭筋・大殿筋', 'tip':'膝とつま先の向きをそろえ、背中を丸めずに動きましょう。'},
    'アームカール': {'triples': ((11,13,15),(12,14,16)), 'down': 80, 'up': 145, 'ideal_low': 35, 'ideal_high': 85, 'joint':'肘', 'muscles':'上腕二頭筋', 'tip':'肘を体側で固定し、反動を使わずに動かしましょう。'},
}
# MediaPipe landmark indices: shoulder 11/12, elbow 13/14, wrist 15/16, hip 23/24, knee 25/26, ankle 27/28.
CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]

@st.cache_resource(show_spinner='姿勢推定モデルを準備しています…')
def model_path():
    target = Path(tempfile.gettempdir()) / 'form_pose_landmarker_lite_v1.task'
    if not target.exists() or target.stat().st_size < 100_000:
        tmp = target.with_suffix('.download')
        try:
            urllib.request.urlretrieve(MODEL_URL, str(tmp))
            if tmp.stat().st_size < 100_000:
                raise RuntimeError('モデルファイルが不完全です')
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
    return str(target)


def angle(a, b, c):
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    x, y = a-b, c-b
    den = np.linalg.norm(x)*np.linalg.norm(y)
    if den < 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(x,y)/den, -1, 1))))


def visible(lms, ids, minimum=.45):
    return all(np.isfinite(lms[i].x) and np.isfinite(lms[i].y) and getattr(lms[i], 'visibility', 1.0) >= minimum for i in ids)


def joint_angle(lms, triple, w, h):
    if not visible(lms, triple):
        return None
    return angle(*[(lms[i].x*w, lms[i].y*h) for i in triple])


def load_state(value, cfg, old=None, margin=7):
    if value is None:
        return old or 'UNKNOWN'
    low, high = cfg['down'], cfg['up']
    if old == 'HIGH' and value <= low+margin:
        return 'HIGH'
    if old == 'LOW' and value >= high-margin:
        return 'LOW'
    return 'HIGH' if value <= low else 'MID' if value < high else 'LOW'


def select_side(lms, cfg, w, h, preferred=None):
    options = []
    for side, triple in zip(('左','右'), cfg['triples']):
        value = joint_angle(lms, triple, w, h)
        if value is not None:
            visibility = sum(getattr(lms[i], 'visibility', 1.0) for i in triple)
            options.append((side, value, visibility))
    if not options:
        return preferred, None
    if preferred:
        for side, value, _ in options:
            if side == preferred:
                return side, value
    side, value, _ = max(options, key=lambda item: item[2])
    return side, value


def draw_pose(frame, lms, exercise, value, side, reps, level):
    h, w = frame.shape[:2]
