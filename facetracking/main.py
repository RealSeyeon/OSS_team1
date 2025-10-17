# main.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import facetrack  # ← 모듈 자체를 임포트

def parse_args():
    p = argparse.ArgumentParser(description="Face Tracking 실행")
    p.add_argument("--video", "-v", type=str, required=True, help="입력 비디오 경로 (예: testvideo.mp4)")
    p.add_argument("--interval", "-i", type=int, default=1, help="얼굴 검출 프레임 간격 (기본=1)")
    return p.parse_args()

def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"입력 비디오 없음: {video_path}")

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "face_boxes.csv"
    preview_path = out_dir / "preview.mp4"

    # facetrack 모듈에 process_video가 정말 있는지 확인
    if not hasattr(facetrack, "process_video"):
        raise AttributeError(
            "facetrack 모듈에 process_video 함수가 없습니다. "
            "facetrack.py 파일 최상위에 def process_video(...): 가 있는지 확인하세요."
        )

    print(f"[INFO] 입력 비디오: {video_path}")
    print(f"[INFO] 프레임 간격: {args.interval}")
    print(f"[INFO] CSV 저장: {csv_path}")
    print(f"[INFO] 미리보기: {preview_path}")

    facetrack.process_video(
        video_path=str(video_path),
        frame_interval=args.interval,
        output_csv=str(csv_path),
        preview_out=str(preview_path),
    )

if __name__ == "__main__":
    main()
