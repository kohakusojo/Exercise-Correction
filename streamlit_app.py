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

st.set_page_config(page_title="筋トレフォームAI", page_icon="🏋️", layout="wide")

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


def paint_segment(frame, a, b, color, thickness=18, alpha=0.48):
    overlay = frame.copy()
    p1 = tuple(np.rint(a).astype(int))
    p2 = tuple(np.rint(b).astype(int))
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)


def paint_chest(frame, lms, w, h, color):
    """肩と腰のランドマークから胸部の概略位置を描画。"""
    if not reliable(lms, [11, 12, 23, 24], threshold=0.32):
        return
    left_sh, right_sh = xy(lms, 11, w, h), xy(lms, 12, w, h)
    left_hip, right_hip = xy(lms, 23, w, h), xy(lms, 24, w, h)
    sh = (left_sh + right_sh) / 2
    hip = (left_hip + right_hip) / 2
    torso = hip - sh
    length = float(np.linalg.norm(torso))
    shoulder_width = float(np.linalg.norm(right_sh - left_sh))
    if length < 12:
        return
    center = sh + torso * 0.22
    # 胸部の位置は簡易推定。横向きの場合は細く描く。
    half_width = max(8, shoulder_width * 0.43)
    half_height = max(6, length * 0.17)
    rotation = float(np.degrees(np.arctan2(torso[1], torso[0])) + 90)
    overlay = frame.copy()
    cv2.ellipse(
        overlay,
        tuple(np.rint(center).astype(int)),
        (int(half_width), int(half_height)),
        rotation, 0, 360, color, -1, cv2.LINE_AA
    )
    cv2.addWeighted(overlay, 0.38, frame, 0.62, 0, frame)


def paint_muscles(frame, lms, exercise, cfg, w, h, display_angle):
    """推定負荷の部位を色付け。姿勢が見えない部位は描かない。"""
    label, color = effort_level(display_angle, cfg, exercise)
    thickness = max(10, int(min(w, h) * 0.035))
    if exercise == "ベンチプレス":
        paint_chest(frame, lms, w, h, color)
        for sh, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
            if reliable(lms, [sh, elbow]):
                paint_segment(frame, xy(lms, sh, w, h), xy(lms, elbow, w, h), color, thickness)
    elif exercise == "スクワット":
        for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
            if reliable(lms, [hip, knee]):
                paint_segment(frame, xy(lms, hip, w, h), xy(lms, knee, w, h), color, thickness + 6)
            if reliable(lms, [hip, 11 if hip == 23 else 12]):
                shoulder = 11 if hip == 23 else 12
                hip_point = xy(lms, hip, w, h)
                sh_point = xy(lms, shoulder, w, h)
                # 臀部を股関節付近の円で示す
                overlay = frame.copy()
                cv2.circle(overlay, tuple(np.rint(hip_point).astype(int)), max(10, thickness // 2), color, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    else:
        for sh, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
            if reliable(lms, [sh, elbow]):
                paint_segment(frame, xy(lms, sh, w, h), xy(lms, elbow, w, h), color, thickness)
    return label, color


def evaluate_video(input_path, output_path, exercise, sensitivity, progress):
    cfg = CONFIG[exercise]
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("動画を読み込めません。MP4形式で再度お試しください。")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < 1 or fps > 240:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0 and total / fps > 90:
        cap.release()
        raise RuntimeError("動画は90秒以内にしてください。")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w <= 0 or orig_h <= 0:
        cap.release()
        raise RuntimeError("動画のサイズを取得できません。")
    factor = min(1.0, 640 / max(orig_w, orig_h))
    w = max(2, int(orig_w * factor) // 2 * 2)
    h = max(2, int(orig_h * factor) // 2 * 2)
    temp_raw = str(Path(output_path).with_name("raw_video.mp4"))
    writer = cv2.VideoWriter(temp_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("出力動画を作成できません。")
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_file()),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    every = max(1, round(fps / 15))
    phase = None
    observed_low = None
    observed_high = None
    count = good = bad = 0
    rows = []
    history = []
    smooth = None
    last_angles = [None, None]
    last_landmarks = None
    last_timestamp = -1
    idx = 0
    invalid_streak = 0
    low_threshold = cfg["low"] + sensitivity
    high_threshold = cfg["high"] - sensitivity
    if high_threshold - low_threshold < 15:
        high_threshold = low_threshold + 15

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if total <= 0 and idx / fps > 90:
                raise RuntimeError("動画は90秒以内にしてください。")
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            if idx % every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                timestamp = max(int(idx * 1000 / fps), last_timestamp + 1)
                last_timestamp = timestamp
                result = detector.detect_for_video(image, timestamp)
                angles = [None, None]
                last_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                if last_landmarks is not None:
                    angles = [landmark_angle(last_landmarks, t, w, h) for t in cfg["triples"]]
                last_angles = angles
                valid = [x for x in angles if x is not None]
                if valid:
                    invalid_streak = 0
                    measured = float(np.mean(valid))
                    smooth = measured if smooth is None else 0.55 * measured + 0.45 * smooth
                    a = smooth
                    history.append({"秒": round(idx / fps, 2), "角度": round(a, 1)})
                    if phase is None:
                        if a <= low_threshold:
                            phase = "bent"
                            observed_low = a
                            observed_high = a
                        elif a >= high_threshold:
                            phase = "extended"
                            observed_low = a
                            observed_high = a
                    else:
                        observed_low = min(observed_low, a)
                        observed_high = max(observed_high, a)
                        if phase == "extended" and a <= low_threshold:
                            phase = "bent"
                        elif phase == "bent" and a >= high_threshold:
                            count += 1
                            passed = observed_low <= cfg["good_low"] and observed_high >= cfg["good_high"]
                            good += int(passed)
                            bad += int(not passed)
                            rows.append({
                                "回数": count,
                                "最小角度(°)": round(observed_low, 1),
                                "最大角度(°)": round(observed_high, 1),
                                "判定": "基準達成" if passed else "基準未達"
                            })
                            phase = "extended"
                            observed_low = a
                            observed_high = a
                else:
                    invalid_streak += 1
                    if invalid_streak > 15:
                        smooth = None
                        phase = None
                        observed_low = None
                        observed_high = None
                        last_landmarks = None

            load_label, load_color = effort_level(smooth, cfg, exercise)
            if last_landmarks is not None and smooth is not None:
                paint_muscles(frame, last_landmarks, exercise, cfg, w, h, smooth)

            overlay = frame.copy()
            panel_w = min(w, 470)
            cv2.rectangle(overlay, (0, 0), (panel_w, 155), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
            angle_label = f"{smooth:.1f}" if smooth is not None else "--"
            cv2.putText(frame, f"ANGLE: {angle_label} deg", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.putText(frame, f"REPS: {count}  OK: {good}  NG: {bad}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 255, 140), 2)
            cv2.putText(frame, f"MUSCLE: {cfg['muscles']}", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2)
            cv2.putText(frame, f"LOAD STAGE: {load_label}", (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, load_color, 2)
            side_label = "BOTH" if all(x is not None for x in last_angles) else "ONE" if any(x is not None for x in last_angles) else "NONE"
            cv2.putText(frame, f"SIDE: {side_label}", (12, 139), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1)
            writer.write(frame)
            idx += 1
            if idx % 30 == 0 and total > 0:
                progress.progress(min(0.90, idx / total * 0.90), text="動画を解析中...")
    finally:
        cap.release()
        writer.release()
        detector.close()

    progress.progress(0.94, text="再生用の動画に変換中...")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", temp_raw,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path
    ], check=True)
    progress.progress(1.0, text="完了")
    return {"count": count, "good": good, "bad": bad, "rows": rows, "history": history}


st.title("🏋️ 筋トレフォームAI")
st.write("左右の関節を自動検出し、回数カウントと筋肉の動作段階を色で表示します。")
st.info("色の意味：🔴 赤＝高負荷の目安　🟡 黄＝中程度　🟢 緑＝低負荷の目安。実際の筋力・筋負荷を測定した結果ではありません。")
exercise = st.selectbox("① 種目", list(CONFIG))
sensitivity = st.slider("② 回数カウント感度（大きいほど小さい動きも数えやすい）", 0, 12, 5)
uploaded = st.file_uploader("③ 動画を選択（90秒以内）", type=["mp4", "mov", "m4v"])
st.caption("関節が見えるように、ベンチプレスとアームカールは斜め前・横、スクワットは横から撮影すると認識しやすくなります。")
if uploaded is not None:
    st.video(uploaded)
    signature = (uploaded.name, uploaded.size, exercise, sensitivity)
    if st.session_state.get("signature") != signature:
        st.session_state.pop("evaluation", None)
    if st.button("フォーム評価を開始", type="primary", use_container_width=True):
        try:
            with tempfile.TemporaryDirectory() as folder:
                src = os.path.join(folder, "input" + Path(uploaded.name).suffix.lower())
                dst = os.path.join(folder, "evaluation.mp4")
                with open(src, "wb") as f:
                    f.write(uploaded.getbuffer())
                progress = st.progress(0, text="準備中...")
                data = evaluate_video(src, dst, exercise, sensitivity, progress)
                with open(dst, "rb") as f:
                    video = f.read()
            st.session_state["evaluation"] = (data, video)
            st.session_state["signature"] = signature
        except Exception as e:
            st.error(f"解析できませんでした：{e}")

    if st.session_state.get("signature") == signature and "evaluation" in st.session_state:
        data, video = st.session_state["evaluation"]
        st.success("解析完了")
        a, b, c = st.columns(3)
        a.metric("総回数", data["count"])
        b.metric("基準達成", data["good"])
        c.metric("基準未達", data["bad"])
        st.subheader("筋肉の色付け・フォーム評価動画")
        st.video(video)
        st.download_button("評価動画をダウンロード", video, "form_evaluation.mp4", "video/mp4")
        if data["history"]:
            st.subheader("関節角度の変化")
            chart = pd.DataFrame(data["history"])
            st.line_chart(chart.set_index("秒")["角度"])
        if data["rows"]:
            st.subheader("回数ごとの判定")
            df = pd.DataFrame(data["rows"])
            st.dataframe(df, hide_index=True)
            st.download_button("CSVをダウンロード", df.to_csv(index=False).encode("utf-8-sig"), "form_results.csv", "text/csv")
        else:
            st.warning("0回の場合は、関節が見える角度で撮影するか、カウント感度を上げて再解析してください。")

st.caption("※ 色は姿勢と関節角度から推定した動作段階の可視化です。筋活動・筋負荷の実測値ではありません。")
