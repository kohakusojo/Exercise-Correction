import csv
import pickle
import warnings
from collections import Counter
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
st.markdown("""
<style>
.block-container {
    max-width: 850px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

div[data-testid="stMetric"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 10px;
}

@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    h1 {
        font-size: 1.7rem !important;
    }
}
</style>
""", unsafe_allow_html=True)
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
MAX_MB = 100
MAX_SECONDS = 30
MAX_WIDTH = 640
FPS_LIMIT = 12

CONFIG = {
    'ベンチプレス': {
        'triples': ((11, 13, 15), (12, 14, 16)),
        'down': 125,
        'up': 150,
        'ideal_low': 65,
        'ideal_high': 115,
        'joint': '肘',
        'muscles': '大胸筋・上腕三頭筋',
        'tip': '肘を左右均等に曲げ、肩甲骨を安定させましょう。'
    },
    'スクワット': {
        'triples': ((23, 25, 27), (24, 26, 28)),
        'down': 135,
        'up': 155,
        'ideal_low': 65,
        'ideal_high': 115,
        'joint': '膝',
        'muscles': '大腿四頭筋・大殿筋',
        'tip': '膝とつま先の向きをそろえ、背中を丸めずに動きましょう。'
    },
    'アームカール': {
        'triples': ((11, 13, 15), (12, 14, 16)),
        'down': 110,
        'up': 145,
        'ideal_low': 35,
        'ideal_high': 85,
        'joint': '肘',
        'muscles': '上腕二頭筋',
        'tip': '肘を体側で固定し、反動を使わずに動かしましょう。'
    }
}

# MediaPipe landmark indices: shoulder 11/12, elbow 13/14, wrist 15/16, hip 23/24, knee 25/26, ankle 27/28.
# アームカール学習済みモデルの入力順序（元の5.detection.ipynbと同じ）
BICEP_KEYPOINTS = [
    ('nose', 0), ('left_shoulder', 11), ('right_shoulder', 12),
    ('right_elbow', 14), ('left_elbow', 13),
    ('right_wrist', 16), ('left_wrist', 15),
    ('left_hip', 23), ('right_hip', 24),
]
BICEP_FEATURE_NAMES = [
    f'{name}_{axis}' for name, _ in BICEP_KEYPOINTS
    for axis in ('x', 'y', 'z', 'v')
]
BICEP_MODEL_DIR = Path(__file__).resolve().parent / 'core' / 'bicep_model' / 'model'


    @st.cache_resource(show_spinner='アームカールAIモデルを読み込んでいます…')
def load_bicep_ai():
    """信頼できるリポジトリ由来のpklのみ読み込むこと。"""
    model_file = BICEP_MODEL_DIR / 'KNN_model.pkl'
    scaler_file = BICEP_MODEL_DIR / 'input_scaler.pkl'
    if not model_file.is_file() or not scaler_file.is_file():
        raise FileNotFoundError('core/bicep_model/model に KNN_model.pkl と input_scaler.pkl が必要です')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with model_file.open('rb') as f:
            model = pickle.load(f)
        with scaler_file.open('rb') as f:
            scaler = pickle.load(f)
    if getattr(scaler, 'n_features_in_', None) != 36 or getattr(model, 'n_features_in_', None) != 36:
        raise ValueError('学習済みモデルの特徴量数が36ではありません')
    actual_columns = list(getattr(scaler, 'feature_names_in_', []))
    if actual_columns and actual_columns != BICEP_FEATURE_NAMES:
        raise ValueError('学習時と特徴量の順序が異なります')
    version_warning = any('version' in str(w.message).lower() for w in caught)
    return model, scaler, version_warning


def predict_bicep_ai(landmarks, model, scaler):
    """MediaPipe Tasks APIの33点から学習時と同じ36特徴量を抽出。"""
    if any(not np.isfinite(getattr(landmarks[i], a, np.nan))
           for _, i in BICEP_KEYPOINTS for a in ('x', 'y', 'z')):
        return None
    if any(getattr(landmarks[i], 'visibility', 0.0) < .45 for _, i in BICEP_KEYPOINTS):
        return None
    row = []
    for _, i in BICEP_KEYPOINTS:
        lm = landmarks[i]
        row.extend([lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)])
    features = pd.DataFrame([row], columns=BICEP_FEATURE_NAMES)
    transformed = scaler.transform(features)
    label = model.predict(transformed)[0]
    probs = model.predict_proba(transformed)[0]
    confidence = float(np.max(probs))
    return str(label), confidence


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

def muscle_diagram(exercise):
    import cv2
    import numpy as np

    img = np.full((400, 320, 3), 245, dtype=np.uint8)

    # 人体の簡易図
    cv2.circle(img, (160, 55), 25, (160, 160, 160), 2)
    cv2.line(img, (160, 80), (160, 240), (160, 160, 160), 4)
    cv2.line(img, (160, 110), (95, 180), (160, 160, 160), 4)
    cv2.line(img, (160, 110), (225, 180), (160, 160, 160), 4)
    cv2.line(img, (160, 240), (110, 350), (160, 160, 160), 4)
    cv2.line(img, (160, 240), (210, 350), (160, 160, 160), 4)

    blue = (255, 100, 0)

    if exercise == "ベンチプレス":
        cv2.circle(img, (140, 125), 18, blue, -1)
        cv2.circle(img, (180, 125), 18, blue, -1)

    elif exercise == "スクワット":
        cv2.line(img, (145, 245), (110, 310), blue, 14)
        cv2.line(img, (175, 245), (210, 310), blue, 14)

    elif exercise == "アームカール":
        cv2.line(img, (135, 125), (105, 165), blue, 12)
        cv2.line(img, (185, 125), (215, 165), blue, 12)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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

    for side, triple in zip(('左', '右'), cfg['triples']):
        value = joint_angle(lms, triple, w, h)

        if value is not None:
            confidence = min(
                getattr(lms[i], 'visibility', 1.0)
                for i in triple
            )
            options.append((side, value, confidence))

    if not options:
        return preferred, None

    best = max(options, key=lambda x: x[2])

    if preferred is not None:
        for item in options:
            if item[0] == preferred:
                # 現在の側が十分見えている間は維持する
                if item[2] >= 0.65:
                    return item[0], item[1]

                # 切替先が十分に信頼できる場合だけ変更
                if best[2] < 0.75:
                    return preferred, None

    return best[0], best[1]


def smooth_pose_points(lms, previous, w, h):
    """関節座標のブレを抑えつつ、実際の動作には追従する。"""

    current = np.array(
        [[lm.x, lm.y] for lm in lms],
        dtype=np.float32
    )

    confidence = np.array(
        [getattr(lm, 'visibility', 1.0) for lm in lms]
    )

    if previous is None or previous.shape != current.shape:
        return current.copy()

    filtered = previous.copy()

    for i, point in enumerate(current):

        if not np.all(np.isfinite(point)):
            continue

        if confidence[i] < 0.5:
            continue

        if not np.all(np.isfinite(previous[i])):
            filtered[i] = point
            continue

        movement_px = np.linalg.norm(
            (point - previous[i]) * [w, h]
        )

        # 微小なブレは無視する
        if movement_px < 2.5:
            continue

        # 急激な位置変化を抑制する
        if movement_px > 45:
            alpha = 0.15

        # ゆっくりした動作
        elif movement_px < 10:
            alpha = 0.25

        # 通常の筋トレ動作
        else:
            alpha = 0.55

        filtered[i] = (
            previous[i]
            + alpha * (point - previous[i])
        )

    return filtered

def draw_pose(frame, lms, exercise, value, side, reps, level, points):
    h, w = frame.shape[:2]

    colors = {
        'HIGH': (35, 45, 235),
        'MID': (0, 215, 255),
        'LOW': (65, 200, 70)
    }

    load_color = colors.get(level, (160, 160, 160))
    base_color = (100, 180, 150)

    if exercise == 'スクワット':
        active_pairs = {
            (23, 25), (25, 27),
            (24, 26), (26, 28)
        }
    else:
        active_pairs = {
            (11, 13), (13, 15),
            (12, 14), (14, 16)
        }

    for a, b in CONNECTIONS:
        if visible(lms, (a, b), .35):
            p1 = (int(points[a, 0]*w), int(points[a, 1]*h))
            p2 = (int(points[b, 0]*w), int(points[b, 1]*h))

            active = (a, b) in active_pairs
            color = load_color if active else base_color

            cv2.line(
                frame, p1, p2, color,
                7 if active else 3,
                cv2.LINE_AA
            )

    for i in range(11, 29):
        if visible(lms, (i,), .35):
            cv2.circle(
                frame,
                (int(points[i, 0]*w), int(points[i, 1]*h)),
                4, (255, 255, 255), -1
            )

     label = (
        f'ANGLE: {value:.0f} deg'
        if value is not None else 'ANGLE: --'
    )
  
   
    # 回数・角度・負荷レベルの表示
    texts = [
        (f'REPS: {reps}', 30, (255, 255, 255)),
        (label, 59, (255, 255, 255)),
        (f'LEVEL: {level}', 88, load_color)
    ]

    for text, y, color in texts:
        # 黒い縁取り
        cv2.putText(
            frame, text, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 0, 0), 5, cv2.LINE_AA
        )

        # 文字本体
        cv2.putText(
            frame, text, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            color, 2, cv2.LINE_AA
        )

    return frame


def normalize_video(src, dst):
    """ffmpeg ensures iPhone HEVC/MOV inputs can be decoded and played as H264 MP4."""
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(src),'-t',str(MAX_SECONDS),'-vf',f'scale=if(gte(iw,ih),min({MAX_WIDTH},iw),-2):if(gte(iw,ih),-2,min({MAX_WIDTH},ih)),fps={FPS_LIMIT}', '-an','-c:v','libx264','-preset','ultrafast','-crf','28','-pix_fmt','yuv420p','-movflags','+faststart',str(dst)]
    # Use force_original_aspect_ratio instead of fragile ffmpeg expressions.
    cmd[cmd.index('-vf')+1] = f'scale={MAX_WIDTH}:{MAX_WIDTH}:force_original_aspect_ratio=decrease:force_divisible_by=2,fps={FPS_LIMIT}'
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    if p.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError('動画の変換に失敗しました: '+p.stderr[-600:])


# スクワットの既存LRモデル用特徴量（3.detection.ipynbに準拠）
SQUAT_KEYPOINTS = [
    ('nose', 0), ('left_shoulder', 11), ('right_shoulder', 12),
    ('left_hip', 23), ('right_hip', 24),
    ('left_knee', 25), ('right_knee', 26),
    ('left_ankle', 27), ('right_ankle', 28),
]
SQUAT_FEATURE_NAMES = [
    f'{name}_{axis}' for name, _ in SQUAT_KEYPOINTS
    for axis in ('x', 'y', 'z', 'v')
]
SQUAT_MODEL_FILE = Path(__file__).resolve().parent / 'core' / 'squat_model' / 'model' / 'LR_model.pkl'

@st.cache_resource(show_spinner='スクワットAIモデルを読み込んでいます…')
def load_squat_ai():
    if not SQUAT_MODEL_FILE.is_file():
        raise FileNotFoundError('core/squat_model/model/LR_model.pkl がありません')
    # 出所を確認した信頼できるモデルファイルだけを読み込んでください。
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with SQUAT_MODEL_FILE.open('rb') as f:
            model = pickle.load(f)
    if getattr(model, 'n_features_in_', None) != 36:
        raise ValueError('スクワットモデルの入力特徴量数が36ではありません')
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is not None and list(feature_names) != SQUAT_FEATURE_NAMES:
        raise ValueError('スクワットモデルの特徴量名・順序が一致しません')
    version_warning = any('version' in str(w.message).lower() for w in caught)
    return model, version_warning


def predict_squat_ai(landmarks, model):
    if any(not np.isfinite(getattr(landmarks[i], axis, np.nan))
           for _, i in SQUAT_KEYPOINTS for axis in ('x', 'y', 'z')):
        return None
    if any(getattr(landmarks[i], 'visibility', 0.0) < .45 for _, i in SQUAT_KEYPOINTS):
        return None
    row = [value for _, i in SQUAT_KEYPOINTS
           for value in (landmarks[i].x, landmarks[i].y,
                         landmarks[i].z, landmarks[i].visibility)]
    X = pd.DataFrame([row], columns=SQUAT_FEATURE_NAMES)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    classes = list(model.classes_)
    pos = classes.index(pred)
    # 元ノートブックでは 0=down、1=up
    label = 'down（しゃがみ）' if str(pred) == '0' else 'up（立位）' if str(pred) == '1' else str(pred)
    return label, float(prob[pos])


# ベンチプレス・スクワット：学習データから作ったモデルのみ読み込む。
# 学習済みファイルがない場合は従来の角度判定を継続する。
EXTRA_MODEL_DIRS = {
    'ベンチプレス': Path(__file__).resolve().parent / 'core' / 'bench_model' / 'model',
    'スクワット': Path(__file__).resolve().parent / 'core' / 'squat_model' / 'model',
}

@st.cache_resource(show_spinner='種目別AIモデルを準備しています…')
def load_extra_ai(exercise):
    folder = EXTRA_MODEL_DIRS[exercise]
    model_file = folder / 'form_classifier.pkl'
    if not model_file.is_file():
        raise FileNotFoundError(f'{model_file.relative_to(Path(__file__).resolve().parent)} がありません。学習用スクリプトで作成してください。')
    # 自分で学習した信頼できるpickleファイルのみ読み込む。
    with model_file.open('rb') as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or payload.get('schema') != 'pose36-v1':
        raise ValueError('モデルの特徴量定義が対応していません')
    if payload.get('exercise') != exercise:
        raise ValueError('選択種目とモデルの種目が一致しません')
    if payload.get('feature_names') != BICEP_FEATURE_NAMES:
        raise ValueError('学習時と入力特徴量の順序が一致しません')
    return payload['pipeline']


def predict_extra_ai(landmarks, pipeline):
    if any(not np.isfinite(getattr(landmarks[i], axis, np.nan))
           for _, i in BICEP_KEYPOINTS for axis in ('x', 'y', 'z')):
        return None
    features = [[v for _, i in BICEP_KEYPOINTS
                 for v in (landmarks[i].x, landmarks[i].y,
                           landmarks[i].z, landmarks[i].visibility)]]
    frame = pd.DataFrame(features, columns=BICEP_FEATURE_NAMES)
    probs = pipeline.predict_proba(frame)[0]
    idx = int(np.argmax(probs))
    return str(pipeline.classes_[idx]), float(probs[idx])


def analyze(src, out, exercise, progress):
    cfg=CONFIG[exercise]
    ai_model = ai_scaler = None
    ai_warning = None
    ai_counts = Counter()
    ai_confidences = []
    if exercise == 'アームカール':
        try:
            ai_model, ai_scaler, version_warning = load_bicep_ai()
            if version_warning:
                ai_warning = '学習時とscikit-learnのバージョンが異なります。AI判定は参考値です。'
        except Exception as exc:
            ai_warning = f'アームカールAIモデルを読み込めませんでした：{exc}'
    if exercise == 'スクワット':
        try:
            ai_model, version_warning = load_squat_ai()
            if version_warning:
                ai_warning = 'スクワットモデルの学習時とscikit-learnのバージョンが異なります。分類は参考値です。'
        except Exception as exc:
            ai_warning = f'スクワットAIモデルを読み込めませんでした（角度判定は利用可能）：{exc}'
    elif exercise == 'ベンチプレス':
        try:
            ai_model = load_extra_ai(exercise)
        except Exception as exc:
            ai_warning = f'{exercise}のAIモデルは未導入です（角度判定は利用可能）：{exc}'
    cap=cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError('動画を読み込めませんでした')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0:
        fps=FPS_LIMIT
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w<=0 or h<=0:
        cap.release(); raise RuntimeError('動画のサイズを取得できません')
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():
        cap.release(); raise RuntimeError('解析結果の動画を書き出せません')
    options=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path()),running_mode=vision.RunningMode.VIDEO,num_poses=1,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
    reps = 0
    valid = 0
    invalid = 0

    phase = 'up'
    side = None
    prev = None
    smoothed = None

    min_angle = None
    max_angle = None
    angles = []
    rows = []
    last_ms = -1

    frame_id = 0
    visible_frames = 0
    previous_points = None
    missed_pose_frames = 0

    rep_min_angle = None
    down_frames = 0
    up_frames = 0
    missing_angle_frames = 0

    STABLE_FRAMES = 3
    try:
        with vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame=cap.read()
                if not ok:
                    break
                frame_id+=1
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
                ms=max(last_ms+1,int((frame_id-1)*1000/fps)); last_ms=ms
                result=detector.detect_for_video(mp_image,ms)
                value=None; level='UNKNOWN'
                ai_label = None
                ai_confidence = None
                if result.pose_landmarks:
                    lms=result.pose_landmarks[0]
                    missed_pose_frames = 0
                    if ai_model is not None:
                        try:
                            prediction = (predict_bicep_ai(lms, ai_model, ai_scaler)
                                          if exercise == 'アームカール' else
                                          predict_squat_ai(lms, ai_model)
                                          if exercise == 'スクワット' else
                                          predict_extra_ai(lms, ai_model))
                            if prediction is not None:
                                ai_label, ai_confidence = prediction
                                if ai_confidence >= .70:
                                    ai_counts[ai_label] += 1
                                    ai_confidences.append(ai_confidence)
                        except Exception as exc:
                            ai_warning = f'AI推論を停止しました：{exc}'
                            ai_model = None
                            ai_scaler = None
                    previous_points = smooth_pose_points(lms, previous_points, w, h)
                    old_side = side
                    side, raw = select_side(
                        lms, cfg, w, h, side
                    )
                    
                    if old_side is not None and side != old_side:
                        smoothed = None
                        prev = None
                        phase = 'up'
                        rep_min_angle = None
                        down_frames = 0
                        up_frames = 0
                    if raw is not None:
                        visible_frames+=1
                        smoothed=raw if smoothed is None else .65*smoothed+.35*raw
                        value=smoothed
                        angles.append(value)
                        min_angle=value if min_angle is None else min(min_angle,value)
                        max_angle=value if max_angle is None else max(max_angle,value)
                        prev=load_state(value,cfg,prev)
                        level=prev
                        if phase == 'up':
                            if value <= cfg['down']:
                                down_frames += 1
                            else:
                                down_frames = 0

                            if down_frames >= STABLE_FRAMES:
                                phase = 'down'
                                rep_min_angle = value
                                down_frames = 0
                                up_frames = 0

                        elif phase == 'down':
                            rep_min_angle = min(
                                rep_min_angle, value
                            )

                            if value >= cfg['up']:
                                up_frames += 1
                            else:
                                up_frames = 0

                            if up_frames >= STABLE_FRAMES:
                                reps += 1

                                if (
                                    cfg['ideal_low']
                                    <= rep_min_angle
                                    <= cfg['ideal_high']
                                ):
                                    valid += 1
                                else:
                                    invalid += 1

                                phase = 'up'
                                rep_min_angle = None
                                down_frames = 0
                                up_frames = 0
                    draw_pose(frame,lms,exercise,value,side,reps,level,previous_points)
                    if ai_label is not None:
                        ai_text = f'AI: {ai_label} ({ai_confidence:.0%})'
                        cv2.putText(frame, ai_text, (12, 118), cv2.FONT_HERSHEY_SIMPLEX,
                                    .55, (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, ai_text, (12, 118), cv2.FONT_HERSHEY_SIMPLEX,
                                    .55, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    missed_pose_frames += 1
                    if missed_pose_frames >= 5:
                        previous_points = None
                    cv2.putText(frame,f'REPS: {reps} / NO POSE',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
                rows.append({'秒':round((frame_id-1)/fps,2),'角度':round(value,1) if value is not None else None,'側':side,'推定負荷':level,'回数':reps,'AI分類':ai_label,'AI予測確率':round(ai_confidence, 3) if ai_confidence is not None else None})
                writer.write(frame)
                if frame_id%12==0:
                    progress.progress(min(.98,frame_id/max(n,1)),text=f'動画解析中… {frame_id}フレーム')
    finally:
        cap.release(); writer.release()
    if frame_id==0:
        raise RuntimeError('動画のフレームを読み取れませんでした')
    detection_ratio=visible_frames/frame_id
    if visible_frames<max(8,frame_id*.12):
        score=None
    else:
        rep_score=(valid/reps)*100 if reps else 0
        tracking_score=min(100,detection_ratio*120)
        score=round(.8*rep_score+.2*tracking_score) if reps else None
    advice=[]
    if detection_ratio<.65:
        advice.append('関節を十分に検出できていません。全身または腕・脚が画面内に収まるよう、明るい場所で横方向から撮影してください。')
    if reps==0:
        advice.append('1回の動作を検出できませんでした。開始姿勢から十分に曲げ、再び伸ばす動きを映してください。')
    if invalid:
        advice.append(f'{invalid}回で動作の深さが基準範囲外でした。可動域を意識してください。')
    if reps and invalid==0:
        advice.append('今回検出した反復では、関節角度の基準を満たしました。左右差や姿勢の安定も確認してください。')
    advice.append(cfg['tip'])
    return {'reps':reps,'valid':valid,'invalid':invalid,'score':score,'min':min_angle,'max':max_angle,'ratio':detection_ratio,'advice':advice,'csv':pd.DataFrame(rows),'ai_counts':dict(ai_counts),'ai_mean_confidence':float(np.mean(ai_confidences)) if ai_confidences else None,'ai_warning':ai_warning}


st.title('🏋️ 筋トレフォーム判定AI')
st.caption('動画から姿勢を推定し、回数・フォーム・使用筋肉を可視化します。スマホ対応。')
exercise=st.selectbox('種目を選択',list(CONFIG.keys()))
st.info('撮影のコツ：体の関節が隠れないように、明るい場所で横から撮影してください。30秒以内の動画を推奨します。')
upload=st.file_uploader('筋トレ動画をアップロード',type=['mp4','mov','m4v','avi'],help='最大100MB。iPhoneのMOVにも対応します。')
if upload is not None:
    st.write(f'選択したファイル：{upload.name}（{upload.size/1024/1024:.1f}MB）')
    if upload.size>MAX_MB*1024*1024:
        st.error(f'ファイルが大きすぎます。{MAX_MB}MB以下にしてください。')
    else:  
        st.video(upload)
        if st.button('▶ フォーム評価を開始',type='primary',use_container_width=True):
            progress=st.progress(0,text='動画を準備中…')
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    folder=Path(tmp)
                    suffix=Path(upload.name).suffix.lower() or '.mp4'
                    original=folder/('input'+suffix)
                    original.write_bytes(upload.getvalue())
                    normalized=folder/'normalized.mp4'
                    annotated=folder/'annotated.mp4'
                    playable=folder/'result.mp4'
                    normalize_video(original,normalized)
                    progress.progress(.02,text='姿勢推定を開始します…')
                    result=analyze(normalized,annotated,exercise,progress)
                    # Browser-compatible H264 output, not the default mp4v codec.
                    p=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(annotated),'-c:v','libx264','-preset','ultrafast','-crf','28','-pix_fmt','yuv420p','-movflags','+faststart',str(playable)],capture_output=True,text=True,timeout=180)
                    if p.returncode!=0:
                        raise RuntimeError('結果動画の変換に失敗しました: '+p.stderr[-500:])
                    st.session_state['analysis']={'exercise':exercise,'result':result,'video':playable.read_bytes(),'name':upload.name}
                    progress.progress(1.0,text='解析完了')
            except subprocess.TimeoutExpired:
                st.error('動画の処理がタイムアウトしました。短い動画で再試行してください。')
            except Exception as e:
                st.error(f'解析できませんでした：{e}')
                st.caption('Streamlit CloudのManage app → Logsにも詳細が表示されます。')

if 'analysis' in st.session_state:
    data=st.session_state['analysis']
    result=data['result']; cfg=CONFIG[data['exercise']]
    st.divider(); st.subheader('📊 解析結果')
    a,b,c=st.columns(3)
    a.metric('合計回数',f"{result['reps']} 回")
    b.metric('基準達成',f"{result['valid']} 回")
    c.metric('基準未達',f"{result['invalid']} 回")
    if result['score'] is not None:
        st.metric('フォーム評価（参考）',f"{result['score']} / 100 点")
        st.progress(result['score']/100)
    else:
        st.warning('十分な反復動作を検出できなかったため、採点できませんでした。')
    if data['exercise'] in ('アームカール', 'ベンチプレス', 'スクワット'):
        st.subheader(f"🤖 {data['exercise']} AI姿勢分類")
        if result.get('ai_warning'):
            st.warning(result['ai_warning'])
        counts = result.get('ai_counts', {})
        if counts:
            dominant = max(counts, key=counts.get)
            st.write(f'**最頻分類ラベル：{dominant}**')
            st.write(f"AI予測確率の平均：{result['ai_mean_confidence']:.1%}")
            st.bar_chart(pd.Series(counts, name='フレーム数'))
            st.caption('分類ラベルは学習データ由来です。予測確率は正解率ではなく、フォームの安全性も保証しません。')
        else:
            st.info('AI分類結果はまだありません。モデルの有無・入力形式・撮影条件を確認してください。角度判定は継続しています。')
    st.subheader('💡 改善アドバイス')
    for message in result['advice']:
        st.write('・'+message)
    st.subheader('💪 使用する筋肉')
    left,right=st.columns([1,1])
    with left:
        muscle_images = {
            "ベンチプレス": "muscle_bench.png",
            "スクワット": "muscle_squat.png",
            "アームカール": "muscle_curl.png",
        }
        image_path = Path(__file__).resolve().parent / muscle_images[data['exercise']]
        if image_path.is_file():
            st.image(str(image_path), caption="赤色：主に使用する筋肉", use_container_width=True)
        else:
            st.image(muscle_diagram(data['exercise']), caption="使用筋肉の簡易図（参考）", use_container_width=True)
            st.caption(f"画像 {image_path.name} が未配置のため簡易図を表示しています。")
    with right:
        st.write('**主な使用部位：** '+cfg['muscles'])
        st.write('**測定した関節：** '+cfg['joint'])
        st.write('**負荷表示：** HIGH / MID / LOW（関節角度による目安）')
        st.caption('筋肉の活動量や実際の負荷を直接測定しているわけではありません。')
    st.subheader('🎬 姿勢推定付き動画')
    st.video(data['video'])
    st.download_button('解析動画をダウンロード（MP4）',data['video'],file_name='form_analysis.mp4',mime='video/mp4',use_container_width=True)
