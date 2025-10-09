import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import sys
import os

class AudioToMel:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 멜 스펙트로그램 설정
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 80
        
        # 멜 스펙트로그램 변환기
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=0,
            f_max=8000
        ).to(self.device)
    
    def load_audio(self, audio_path):
        """오디오 파일 로드"""
        waveform, sr = torchaudio.load(audio_path)
        
        # 샘플레이트 변환 (필요시)
        if sr != self.sample_rate:
            print(f"Resampling from {sr}Hz to {self.sample_rate}Hz...")
            resampler = T.Resample(sr, self.sample_rate).to(self.device)
            waveform = resampler(waveform)
        
        # 모노로 변환
        if waveform.shape[0] > 1:
            print("Converting to mono...")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.to(self.device)
    
    def waveform_to_mel(self, waveform):
        """Waveform을 멜 스펙트로그램으로 변환"""
        mel_spec = self.mel_transform(waveform)
        # dB 스케일로 변환 (로그)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        return mel_spec_db
    
    def save_mel(self, mel_spec, output_path):
        """멜 스펙트로그램 저장 (.pt 파일)"""
        torch.save({
            'mel_spectrogram': mel_spec.cpu(),
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels
        }, output_path)
        print(f"Saved mel spectrogram: {output_path}")
    
    def save_mel_numpy(self, mel_spec, output_path):
        """멜 스펙트로그램을 numpy로 저장 (.npy 파일)"""
        np.save(output_path, mel_spec.cpu().numpy())
        
        # 메타데이터도 별도 저장
        meta_path = output_path.replace('.npy', '_meta.pt')
        torch.save({
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels
        }, meta_path)
        print(f"Saved mel spectrogram: {output_path}")
        print(f"Saved metadata: {meta_path}")


def main():
    # 기본 파일명 설정
    input_file = "original_audio.wav"
    output_file = "original_mel.pt"
    
    # 명령줄 인자가 있으면 그걸 사용
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Please make sure '{input_file}' exists in the current directory.")
        sys.exit(1)
    
    print("=== Audio to Mel Spectrogram Conversion ===")
    
    # 변환 실행
    converter = AudioToMel()
    
    print(f"\nLoading: {input_file}")
    waveform = converter.load_audio(input_file)
    print(f"Waveform shape: {waveform.shape}")
    print(f"Duration: {waveform.shape[-1] / converter.sample_rate:.2f} seconds")
    
    print("\nConverting to mel spectrogram...")
    mel_spec = converter.waveform_to_mel(waveform)
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print(f"  Channels: {mel_spec.shape[0]}")
    print(f"  Mel bins: {mel_spec.shape[1]}")
    print(f"  Time frames: {mel_spec.shape[2]}")
    
    # 저장
    print()
    if output_file.endswith('.npy'):
        converter.save_mel_numpy(mel_spec, output_file)
    else:
        converter.save_mel(mel_spec, output_file)
    
    print("\n✓ Conversion completed!")


if __name__ == "__main__":
    main()