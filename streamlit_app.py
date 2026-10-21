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
            p1 = (int(lms[a].x*w), int(lms[a].y*h))
            p2 = (int(lms[b].x*w), int(lms[b].y*h))

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
                (int(lms[i].x*w), int(lms[i].y*h)),
                4, (255, 255, 255), -1
            )

    cv2.putText(
        frame, f'REPS: {reps}', (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX, .8,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    label = (
        f'ANGLE: {value:.0f} deg'
        if value is not None else 'ANGLE: --'
    )

    cv2.putText(
        frame, label, (12, 59),
        cv2.FONT_HERSHEY_SIMPLEX, .57,
        (255, 255, 255), 2, cv2.LINE_AA
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


def analyze(src, out, exercise, progress):
    cfg=CONFIG[exercise]
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
    reps=0; valid=0; invalid=0; phase='up'; side=None; prev=None; smoothed=None
    min_angle=None; max_angle=None; angles=[]; rows=[]; last_ms=-1  
    frame_id=0; visible_frames=0; completed_depth=[]
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
                if result.pose_landmarks:
                    lms=result.pose_landmarks[0]
                    side, raw=select_side(lms,cfg,w,h,side)
                    if raw is not None:
                        visible_frames+=1
                        smoothed=raw if smoothed is None else .65*smoothed+.35*raw
                        value=smoothed
                        angles.append(value)
                        min_angle=value if min_angle is None else min(min_angle,value)
                        max_angle=value if max_angle is None else max(max_angle,value)
                        prev=load_state(value,cfg,prev)
                        level=prev
                        if phase=='up' and value<=cfg['down']:
                            phase='down'; completed_depth.append(value)
                        elif phase=='down':
                            if completed_depth:
                                completed_depth[-1]=min(completed_depth[-1],value)
                            if value>=cfg['up']:
                                reps+=1
                                depth=completed_depth[-1] if completed_depth else value
                                if cfg['ideal_low']<=depth<=cfg['ideal_high']:
                                    valid+=1
                                else:
                                    invalid+=1
                                phase='up'
                    draw_pose(frame,lms,exercise,value,side,reps,level)
                else:
                    cv2.putText(frame,f'REPS: {reps} / NO POSE',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
                rows.append({'秒':round((frame_id-1)/fps,2),'角度':round(value,1) if value is not None else None,'側':side,'推定負荷':level,'回数':reps})
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
    return {'reps':reps,'valid':valid,'invalid':invalid,'score':score,'min':min_angle,'max':max_angle,'ratio':detection_ratio,'advice':advice,'csv':pd.DataFrame(rows)}


st.title('🏋️ 筋トレフォーム判定AI')
st.caption('動画から姿勢を推定し、回数・フォーム・使用筋肉を可視化します。スマホ対応。')
exercise=st.selectbox('種目を選択',list(CONFIG.keys()))
st.info('撮影のコツ：体の関節が隠れないように、明るい場所で横から撮影してください。90秒以内の動画を推奨します。')
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
    st.subheader('💡 改善アドバイス')
    for message in result['advice']:
        st.write('・'+message)
    st.subheader('💪 使用する筋肉')
    left,right=st.columns([1,1])
    with left:
        st.image(
            "muscle_reference.png.png",
            caption="使用する筋肉の参考画像",
            use_container_width=True
        )
    with right:
        st.write('**主な使用部位：** '+cfg['muscles'])
        st.write('**測定した関節：** '+cfg['joint'])
        st.write('**負荷表示：** HIGH / MID / LOW（関節角度による目安）')
        st.caption('筋肉の活動量や実際の負荷を直接測定しているわけではありません。')
    st.subheader('🎬 姿勢推定付き動画')
    st.video(data['video'])
    st.download_button('解析動画をダウンロード（MP4）',data['video'],file_name='form_analysis.mp4',mime='video/mp4',use_container_width=True)
    csv_bytes=result['csv'].to_csv(index=False).encode('utf-8-sig')
    st.download_button('角度・回数データをダウンロード（CSV）',csv_bytes,file_name='form_analysis.csv',mime='text/csv',use_container_width=True)
    with st.expander('関節角度の推移'):
        st.line_chart(result['csv'].set_index('秒')['角度'])
        st.dataframe(result['csv'].tail(30),use_container_width=True)
    st.caption('本アプリのフォーム採点は簡易的な関節角度ルールに基づく参考値です。安全性やトレーニング効果を保証するものではありません。')

