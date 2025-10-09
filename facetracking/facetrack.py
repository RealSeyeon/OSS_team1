# facetrack.py
# pip install facenet-pytorch opencv-python pillow pandas torch torchvision torchaudio

import cv2
import torch
from facenet_pytorch import MTCNN
import pandas as pd
from PIL import Image

def process_video(video_path, frame_interval=1, output_csv="face_boxes.csv", preview_out="preview.mp4"):
    """
    영상에서 N프레임마다 얼굴을 검출하여 (frame, x, y, w, h, score) CSV로 저장하고
    검출 결과가 그려진 미리보기 영상을 저장한다. (단일 얼굴 기준)
    """
    # 1) 모델 셋업
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=False, device=device)  # 단일 얼굴 기준

    # 2) 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"비디오를 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(preview_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 지정한 간격만 처리 (속도 절약)
        if frame_idx % frame_interval == 0:
            # OpenCV → PIL 변환
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            boxes, probs = mtcnn.detect(pil)
            if boxes is not None and len(boxes) > 0:
                # 단일 얼굴 선택 (첫 번째)
                box = boxes[0]
                prob = float(probs[0])
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1

                records.append({
                    "frame": frame_idx,
                    "x": x1, "y": y1,
                    "w": w, "h": h,
                    "score": prob
                })

                # 미리보기: 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{prob:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                # 검출 실패 프레임 기록
                records.append({
                    "frame": frame_idx,
                    "x": -1, "y": -1, "w": -1, "h": -1,
                    "score": 0.0
                })

        # 항상 미리보기 프레임 저장
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print("Saved coords to", output_csv)
    print("Preview video saved:", preview_out)

# 내보낼 심볼 명시 (선택)
__all__ = ["process_video"]
