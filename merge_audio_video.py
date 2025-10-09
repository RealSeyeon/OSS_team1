from moviepy import VideoFileClip, AudioFileClip
import sys

# 기본 파일명 설정
video_file = "video1.mp4"
audio_file = "adversarial_audio.wav"
output_file = "video1_protected.mp4"

# 명령줄 인자가 있으면 사용
if len(sys.argv) >= 2:
    video_file = sys.argv[1]
if len(sys.argv) >= 3:
    audio_file = sys.argv[2]
if len(sys.argv) >= 4:
    output_file = sys.argv[3]

print(f"Loading video: {video_file}")
video = VideoFileClip(video_file)

print(f"Loading audio: {audio_file}")
audio = AudioFileClip(audio_file)

print("Merging audio with video...")

try:
    final_video = video.with_audio(audio)
except AttributeError:
    final_video = video.set_audio(audio)

print(f"Saving to: {output_file}")
final_video.write_videofile(output_file, codec='libx264', audio_codec='aac')

video.close()
audio.close()
final_video.close()

print("\n✓ Done! Protected video saved as:", output_file)