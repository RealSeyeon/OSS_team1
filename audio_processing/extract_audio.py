from moviepy import VideoFileClip  # .editor 제거
import sys

if len(sys.argv) < 3:
    print("Usage: python3 extract_audio.py input_video.mp4 output.wav")
    sys.exit(1)

in_file = sys.argv[1]
out_file = sys.argv[2]

# 비디오 파일 열기
clip = VideoFileClip(in_file)

# 오디오 부분만 추출
audio = clip.audio

# 16kHz, 16bit PCM wav로 저장
audio.write_audiofile(out_file, fps=16000, nbytes=2)

clip.close()
