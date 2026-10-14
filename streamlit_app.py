
import os
import math
import tempfile
import subprocess
import urllib.request
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st

from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==============================================
# 基本設定
# ==============================================

st.set_page_config(
    page_title="筋トレフォームAI",
    page_icon="🏋️",
    layout="centered"
)

MAX_SIZE = 640
ANALYSIS_FPS = 15
MAX_SECONDS = 60
VISIBILITY = 0.55

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

MODEL_PATH = Path(tempfile.gettempdir()) / "pose_landmarker_lite.task"

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


# ==============================================
# 種目設定
# ==============================================

EXERCISES = {
    "ベンチプレス": {
        "indices": {
            "右": (12, 14, 16),
            "左": (11, 13, 15)
        },
        "joint": "肘",
        "muscle": "大胸筋",
        "min_range": (70, 100),
        "max_range": (155, 175),
        "bend": 110,
        "extend": 145
    },
    "スクワット": {
        "indices": {
            "右": (24, 26, 28),
            "左": (23, 25, 27)
        },
        "joint": "膝",
        "muscle": "大腿四頭筋",
        "min_range": (80, 100),
        "max_range": (165, 180),
        "bend": 115,
        "extend": 155
    },
    "アームカール": {
        "indices": {
            "右": (12, 14, 16),
            "左": (11, 13, 15)
        },
        "joint": "肘",
        "muscle": "上腕二頭筋",
        "min_range": (30, 40),
        "max_range": (150, 160),
        "bend": 60,
        "extend": 140
    }
}


# ==============================================
# モデル・フォント
# ==============================================

@st.cache_resource
def get_model_path():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1000000:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


@lru_cache(maxsize=32)
def get_font(size):
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


# ==============================================
# 計算関数
# ==============================================

def point_xy(lm, width, height):
    return np.array([lm.x * width, lm.y * height], dtype=float)


def reliable(landmarks, indices):
    return all(
        np.isfinite(landmarks[i].x)
        and np.isfinite(landmarks[i].y)
        and landmarks[i].visibility >= VISIBILITY
        for i in indices
    )


def calculate_angle(a, b, c):
    ab = np.asarray(a) - np.asarray(b)
    cb = np.asarray(c) - np.asarray(b)

    length = np.linalg.norm(ab) * np.linalg.norm(cb)
    if length < 1e-8:
        return None

    cosine = np.clip(np.dot(ab, cb) / length, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def calculate_score(angle, target):
    if angle is None:
        return 0
    low, high = target
    difference = max(low - angle, angle - high, 0)
    return max(0, round(100 - 5 * difference))


def get_load(angle, cfg):
    # 色は角度による動作段階の目安
    if angle >= cfg["extend"]:
        return "低", (0, 255, 0)
    if angle >= cfg["bend"]:
        return "中", (0, 255, 255)
    return "高", (0, 0, 255)


def draw_text(frame, items):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    for text, x, y, size, color in items:
        draw.text(
            (int(x), int(y)),
            str(text),
            font=get_font(int(size)),
            fill=(color[2], color[1], color[0])
        )

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


# ==============================================
# 胸上部の追跡
# ==============================================

def estimate_chest(landmarks, width, height):
    ls = point_xy(landmarks[11], width, height)
    rs = point_xy(landmarks[12], width, height)
    lh = point_xy(landmarks[23], width, height)
    rh = point_xy(landmarks[24], width, height)

    shoulder = (ls + rs) / 2
    hip = (lh + rh) / 2

    torso = hip - shoulder
    length = np.linalg.norm(torso)

    if length < 15:
        return None

    center = shoulder + torso * 0.12
    shoulder_width = np.linalg.norm(rs - ls)

    major = max(12, length * 0.22, shoulder_width * 0.38)
    minor = max(6, length * 0.085)

    rotation = math.degrees(math.atan2(torso[1], torso[0])) + 90

    return np.array(
        [center[0], center[1], major, minor, rotation],
        dtype=float
    )


def smooth_chest(previous, current, alpha=0.30):
    if previous is None:
        return current.copy()

    result = previous.copy()
    result[:4] = (1 - alpha) * previous[:4] + alpha * current[:4]

    difference = (current[4] - previous[4] + 90) % 180 - 90
    result[4] = previous[4] + alpha * difference

    return result


def draw_chest(frame, chest, color):
    x, y, major, minor, rotation = chest
    overlay = frame.copy()

    cv2.ellipse(
        overlay,
        (int(x), int(y)),
        (max(1, int(major)), max(1, int(minor))),
        float(rotation),
        0,
        360,
        color,
        -1,
        cv2.LINE_AA
    )

    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)


def draw_limb(frame, points, color):
    a, b, c = [tuple(p.astype(int)) for p in points]

    overlay = frame.copy()
    thickness = max(10, int(frame.shape[1] * 0.025))

    cv2.line(overlay, a, b, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

    cv2.line(frame, b, c, (255, 255, 255), 3, cv2.LINE_AA)

    for p in (a, b, c):
        cv2.circle(frame, p, 5, (255, 255, 255), -1)


# ==============================================
# 動画解析
# ==============================================

def process_video(input_path, output_path, exercise, side, progress):
    cfg = EXERCISES[exercise]
    indices = cfg["indices"][side]

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした")

    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_w <= 0 or source_h <= 0:
        cap.release()
        raise RuntimeError("動画サイズを取得できません")

    if not (1 <= fps <= 240):
        fps = 30.0

    if total > 0 and total / fps > MAX_SECONDS:
        cap.release()
        raise RuntimeError("60秒以内の動画を選択してください")

    scale = min(1.0, MAX_SIZE / max(source_w, source_h))
    width = max(2, int(source_w * scale) // 2 * 2)
    height = max(2, int(source_h * scale) // 2 * 2)

    raw_path = os.path.join(os.path.dirname(output_path), "raw.mp4")

    writer = cv2.VideoWriter(
        raw_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("動画の保存を開始できませんでした")

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=get_model_path()
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    interval = max(1, round(fps / ANALYSIS_FPS))

    count = 0
    good = 0
    bad = 0
    stage = "waiting"

    bend_frames = 0
    extend_frames = 0

    smooth_angle = None
    tracked_chest = None

    overall_min = None
    overall_max = None
    rep_min = None
    rep_max = None

    display_angle = None
    display_points = None
    display_chest = None

    last_result = "測定待ち"
    results = []

    index = 0
    last_timestamp = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(
                frame, (width, height),
                interpolation=cv2.INTER_AREA
            )

            # 姿勢推定は一定間隔で実行
            if index % interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb)
                )

                timestamp = max(
                    int(index * 1000 / fps),
                    last_timestamp + 1
                )
                last_timestamp = timestamp

                detection = landmarker.detect_for_video(
                    mp_image, timestamp
                )

                display_angle = None
                display_points = None
                display_chest = None

                if detection.pose_landmarks:
                    landmarks = detection.pose_landmarks[0]

                    if reliable(landmarks, indices):
                        points = [
                            point_xy(landmarks[i], width, height)
                            for i in indices
                        ]

                        measured = calculate_angle(*points)

                        if measured is not None:
                            smooth_angle = (
                                measured if smooth_angle is None
                                else 0.4 * measured + 0.6 * smooth_angle
                            )

                            angle = smooth_angle
                            display_angle = angle
                            display_points = points

                            overall_min = (
                                angle if overall_min is None
                                else min(overall_min, angle)
                            )
                            overall_max = (
                                angle if overall_max is None
                                else max(overall_max, angle)
                            )

                            if stage != "waiting":
                                rep_min = (
                                    angle if rep_min is None
                                    else min(rep_min, angle)
                                )
                                rep_max = (
                                    angle if rep_max is None
                                    else max(rep_max, angle)
                                )

                            bend_frames = (
                                bend_frames + 1
                                if angle <= cfg["bend"] else 0
                            )
                            extend_frames = (
                                extend_frames + 1
                                if angle >= cfg["extend"] else 0
                            )

                            bent = bend_frames >= 2
                            extended = extend_frames >= 2

                            if stage == "waiting" and extended:
                                stage = "extended"
                                rep_min = angle
                                rep_max = angle

                            elif stage == "extended" and bent:
                                stage = "bent"

                            elif stage == "bent" and extended:
                                count += 1

                                passed = (
                                    rep_min <= cfg["min_range"][1]
                                    and rep_max >= cfg["max_range"][0]
                                )

                                if passed:
                                    good += 1
                                    last_result = "基準達成"
                                else:
                                    bad += 1
                                    last_result = "基準未達"

                                results.append({
                                    "回数": count,
                                    "最小角度": round(rep_min, 1),
                                    "最大角度": round(rep_max, 1),
                                    "最小スコア": calculate_score(
                                        rep_min, cfg["min_range"]
                                    ),
                                    "最大スコア": calculate_score(
                                        rep_max, cfg["max_range"]
                                    ),
                                    "判定": last_result
                                })

                                stage = "extended"
                                rep_min = angle
                                rep_max = angle
                    else:
                        bend_frames = 0
                        extend_frames = 0

                    if (
                        exercise == "ベンチプレス"
                        and display_angle is not None
                        and reliable(landmarks, [11, 12, 23, 24])
                    ):
                        chest = estimate_chest(
                            landmarks, width, height
                        )
                        if chest is not None:
                            tracked_chest = smooth_chest(
                                tracked_chest, chest
                            )
                            display_chest = tracked_chest.copy()
                else:
                    bend_frames = 0
                    extend_frames = 0

            white = (255, 255, 255)
            green = (0, 255, 0)
            red = (0, 0, 255)

            if display_angle is not None:
                load_text, load_color = get_load(
                    display_angle, cfg
                )
                movement = (
                    "伸展" if display_angle >= cfg["extend"]
                    else "屈曲" if display_angle <= cfg["bend"]
                    else "中間"
                )
            else:
                load_text = "測定待ち"
                load_color = white
                movement = "測定待ち"

            # 胸を先に描き、腕を上から描画
            if display_chest is not None:
                draw_chest(frame, display_chest, load_color)

            if display_points is not None:
                draw_limb(frame, display_points, load_color)

            # 評価パネル
            panel_scale = min(1.0, width / 850, height / 750)

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (8, 8),
                (
                    int(620 * panel_scale),
                    int(570 * panel_scale)
                ),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, 0.38, frame, 0.62, 0, frame)

            def fmt(value):
                return f"{value:.1f}°" if value is not None else "--"

            min_score = (
                calculate_score(overall_min, cfg["min_range"])
                if overall_min is not None else "--"
            )
            max_score = (
                calculate_score(overall_max, cfg["max_range"])
                if overall_max is not None else "--"
            )

            labels = [
                (f"{exercise} フォーム評価", 25, 25, 30, white),
                (
                    f"現在の{cfg['joint']}角度：{fmt(display_angle)}",
                    25, 75, 24, white
                ),
                (f"動作状態：{movement}", 25, 118, 24, load_color),
                (
                    f"{cfg['muscle']} 推定負荷：{load_text}",
                    25, 161, 24, load_color
                ),
                (f"最小角度：{fmt(overall_min)}", 25, 216, 24, white),
                (f"最小スコア：{min_score}/100", 25, 258, 24, white),
                (f"最大角度：{fmt(overall_max)}", 25, 310, 24, white),
                (f"最大スコア：{max_score}/100", 25, 352, 24, white),
                (f"総回数：{count}回", 25, 402, 24, white),
                (f"基準達成：{good}回", 25, 443, 24, green),
                (f"基準未達：{bad}回", 25, 484, 24, red),
                (f"前回の判定：{last_result}", 25, 525, 24, white)
            ]

            scaled_labels = [
                (
                    text,
                    int(x * panel_scale),
                    int(y * panel_scale),
                    max(12, int(size * panel_scale)),
                    color
                )
                for text, x, y, size, color in labels
            ]

            frame = draw_text(frame, scaled_labels)
            writer.write(frame)

            index += 1

            if index % 15 == 0 and total > 0:
                progress.progress(
                    min(index / total, 0.94),
                    text=f"解析中：{index}/{total}フレーム"
                )

    finally:
        cap.release()
        writer.release()
        landmarker.close()

    progress.progress(0.95, text="動画を変換中...")

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", raw_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ],
        check=True
    )

    progress.progress(1.0, text="解析完了")

    return {
        "count": count,
        "good": good,
        "bad": bad,
        "minimum": overall_min,
        "maximum": overall_max,
        "results": results
    }


# ==============================================
# Streamlit画面
# ==============================================

st.title("🏋️ 筋トレフォーム評価システム")
st.write("スマホで撮影した動画をAIで解析します。")

exercise = st.selectbox(
    "① 種目を選択",
    list(EXERCISES.keys())
)

side = st.radio(
    "② 評価する側を選択",
    ["右", "左"],
    horizontal=True
)

uploaded = st.file_uploader(
    "③ 動画をアップロード",
    type=["mp4", "mov", "m4v"]
)

st.caption("まずは5〜15秒程度の動画で試してください。")

if uploaded is not None:
    st.video(uploaded)

    if st.button(
        "フォーム評価を開始",
        type="primary",
        use_container_width=True
    ):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                extension = Path(uploaded.name).suffix.lower()
                input_path = os.path.join(tmp, "input" + extension)
                output_path = os.path.join(tmp, "result.mp4")

                with open(input_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                progress = st.progress(0)

                with st.spinner("フォームを解析しています..."):
                    summary = process_video(
                        input_path,
                        output_path,
                        exercise,
                        side,
                        progress
                    )

                with open(output_path, "rb") as f:
                    video_bytes = f.read()

            st.session_state["result"] = {
                "summary": summary,
                "video": video_bytes
            }

        except Exception as e:
            st.error(f"エラー：{e}")


# ==============================================
# 結果表示
# ==============================================

if "result" in st.session_state:
    result = st.session_state["result"]
    summary = result["summary"]

    st.success("フォーム評価が完了しました！")

    c1, c2, c3 = st.columns(3)
    c1.metric("総回数", summary["count"])
    c2.metric("基準達成", summary["good"])
    c3.metric("基準未達", summary["bad"])

    st.subheader("評価動画")
    st.video(result["video"])

    st.download_button(
        "評価動画をダウンロード",
        data=result["video"],
        file_name="form_evaluation.mp4",
        mime="video/mp4",
        use_container_width=True
    )

    if summary["results"]:
        st.subheader("回数ごとの評価")
        df = pd.DataFrame(summary["results"])
        st.dataframe(df, hide_index=True)

        st.download_button(
            "評価結果をCSVで保存",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="evaluation.csv",
            mime="text/csv"
        )
    else:
        st.info("回数が検出されませんでした。")

st.caption(
    "※ 筋肉の色表示は関節角度に基づく推定であり、"
    "実際の筋負荷を測定するものではありません。"
)
