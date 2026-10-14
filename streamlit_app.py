
import os
import math
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
    "ベンチプレス": {"triples": [(11, 13, 15), (12, 14, 16)], "low": 105, "high": 145, "good_low": 105, "good_high": 145, "joint": "肘"},
    "スクワット": {"triples": [(23, 25, 27), (24, 26, 28)], "low": 120, "high": 155, "good_low": 110, "good_high": 155, "joint": "膝"},
    "アームカール": {"triples": [(11, 13, 15), (12, 14, 16)], "low": 90, "high": 135, "good_low": 80, "good_high": 145, "joint": "肘"},
}

@st.cache_resource(show_spinner="姿勢推定モデルを準備中...")
def model_file():
    path = Path(tempfile.gettempdir()) / "form_pose_landmarker_lite.task"
    if not path.exists() or path.stat().st_size < 1000000:
        urllib.request.urlretrieve(MODEL_URL, path)
    return str(path)


def angle3(a, b, c):
    u = a - b
    v = c - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-7:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1, 1))))


def landmark_angle(lms, triple, w, h):
    if any((not np.isfinite(lms[i].x) or not np.isfinite(lms[i].y) or lms[i].visibility < 0.40) for i in triple):
        return None
    points = [np.array([lms[i].x * w, lms[i].y * h], dtype=float) for i in triple]
    return angle3(*points)


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
    # 角度の上下動をヒステリシスで検出する。初期姿勢は屈曲・伸展どちらでもよい。
    phase = None
    observed_low = None
    observed_high = None
    count = good = bad = 0
    rows = []
    history = []
    smooth = None
    last_angles = [None, None]
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
                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    angles = [landmark_angle(lms, t, w, h) for t in cfg["triples"]]
                last_angles = angles
                valid = [x for x in angles if x is not None]
                if valid:
                    invalid_streak = 0
                    # 左右両方が見える場合は平均、片側ならその角度を使用。
                    measured = float(np.mean(valid))
                    smooth = measured if smooth is None else 0.55 * measured + 0.45 * smooth
                    a = smooth
                    history.append({"秒": round(idx / fps, 2), "角度": round(a, 1)})
                    observed_low = a if observed_low is None else min(observed_low, a)
                    observed_high = a if observed_high is None else max(observed_high, a)
                    # 最初の十分な屈曲または伸展を起点にする。
                    if phase is None:
                        if a <= low_threshold:
                            phase = "bent"
                        elif a >= high_threshold:
                            phase = "extended"
                    elif phase == "extended" and a <= low_threshold:
                        phase = "bent"
                    elif phase == "bent" and a >= high_threshold:
                        count += 1
                        # 1回の動作中に観測した角度の最大・最小で評価。
                        low_seen = observed_low if observed_low is not None else a
                        high_seen = observed_high if observed_high is not None else a
                        passed = low_seen <= cfg["good_low"] and high_seen >= cfg["good_high"]
                        good += int(passed)
                        bad += int(not passed)
                        rows.append({"回数": count, "最小角度(°)": round(low_seen, 1), "最大角度(°)": round(high_seen, 1), "判定": "基準達成" if passed else "基準未達"})
                        phase = "extended"
                        observed_low = a
                        observed_high = a
                    # 直近サイクルの最大・最小のみ記録する。
                    if phase is not None and (not rows or observed_low is not None):
                        pass
                else:
                    invalid_streak += 1
                    if invalid_streak > 15:
                        smooth = None
                        phase = None
                        observed_low = None
                        observed_high = None

            # 解析結果を動画上に重ねる。日本語はStreamlit側に表示し動画は英字にする。
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (min(w, 355), 115), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            angle_label = f"{smooth:.1f}" if smooth is not None else "--"
            cv2.putText(frame, f"ANGLE: {angle_label} deg", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 255), 2)
            cv2.putText(frame, f"REPS: {count}  OK: {good}  NG: {bad}", (12, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 255, 140), 2)
            cv2.putText(frame, f"SIDE: {'BOTH' if all(x is not None for x in last_angles) else 'ONE' if any(x is not None for x in last_angles) else 'NONE'}", (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
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
st.write("左右の関節を自動検出し、運動の曲げ伸ばしから回数を数えます。")
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
        st.subheader("評価動画")
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

st.caption("※ 判定は関節角度に基づく簡易評価で、医療・運動指導の専門家による判定に代わるものではありません。")
