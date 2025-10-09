import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import sys
import os

class MelToAudio:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Griffin-Lim 알고리즘으로 phase 복원
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=32,  # iteration 수 (높을수록 품질 좋지만 느림)
            power=1.0
        ).to(self.device)
    
    def load_mel(self, mel_path):
        """멜 스펙트로그램 로드"""
        if mel_path.endswith('.npy'):
            # numpy 파일
            mel_spec = torch.from_numpy(np.load(mel_path)).to(self.device)
            
            # 메타데이터 로드
            meta_path = mel_path.replace('.npy', '_meta.pt')
            if os.path.exists(meta_path):
                metadata = torch.load(meta_path)
                self.sample_rate = metadata.get('sample_rate', self.sample_rate)
                self.n_fft = metadata.get('n_fft', self.n_fft)
                self.hop_length = metadata.get('hop_length', self.hop_length)
                self.n_mels = metadata.get('n_mels', self.n_mels)
                print("Loaded metadata from file")
            else:
                print("Warning: No metadata file found, using default parameters")
        else:
            # pytorch 파일
            data = torch.load(mel_path)
            mel_spec = data['mel_spectrogram'].to(self.device)
            
            # 메타데이터 적용
            self.sample_rate = data.get('sample_rate', self.sample_rate)
            self.n_fft = data.get('n_fft', self.n_fft)
            self.hop_length = data.get('hop_length', self.hop_length)
            self.n_mels = data.get('n_mels', self.n_mels)
        
        return mel_spec
    
    def mel_to_waveform(self, mel_spec_db):
        """
        멜 스펙트로그램을 waveform으로 복원
        Griffin-Lim 알고리즘 사용
        """
        # dB에서 선형 스케일로 복원
        mel_spec = torch.pow(10, mel_spec_db / 20.0)
        
        # 차원 정리: [batch, n_mels, time] -> [n_mels, time]
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.squeeze(0)
        
        # 멜 스케일에서 선형 주파수로 역변환
        mel_basis = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=0,
            f_max=8000,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm='slaney'
        ).to(self.device)
        
        # mel_basis shape: [n_freqs, n_mels]
        # mel_spec shape: [n_mels, time]
        # 결과: [n_freqs, time]
        stft_approx = torch.matmul(mel_basis, mel_spec)
        
        # Griffin-Lim으로 phase 복원 및 waveform 생성
        print("Reconstructing waveform with Griffin-Lim algorithm...")
        waveform = self.griffin_lim(stft_approx)
        
        return waveform.unsqueeze(0)
    
    def save_audio(self, waveform, output_path):
        """오디오 파일 저장"""
        # 정규화 (클리핑 방지)
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val * 0.95
        
        # 저장
        torchaudio.save(output_path, waveform.cpu(), self.sample_rate)
        print(f"Saved audio: {output_path}")
        
        # 정보 출력
        duration = waveform.shape[-1] / self.sample_rate
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Samples: {waveform.shape[-1]}")


def main():
    # 기본 파일명 설정
    input_file = "adversarial_mel.pt"
    output_file = "adversarial_audio.wav"
    
    # 명령줄 인자가 있으면 사용
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Please make sure '{input_file}' exists in the current directory.")
        sys.exit(1)
    
    # 파라미터 파싱 (옵션)
    sample_rate = None
    n_fft = None
    hop_length = None
    n_mels = None
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--sample_rate' and i + 1 < len(sys.argv):
            sample_rate = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--n_fft' and i + 1 < len(sys.argv):
            n_fft = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--hop_length' and i + 1 < len(sys.argv):
            hop_length = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--n_mels' and i + 1 < len(sys.argv):
            n_mels = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    print("=== Mel Spectrogram to Audio Conversion ===")
    
    # 변환 실행
    converter = MelToAudio(
        sample_rate=sample_rate or 16000,
        n_fft=n_fft or 1024,
        hop_length=hop_length or 256,
        n_mels=n_mels or 80
    )
    
    print(f"\nLoading: {input_file}")
    mel_spec = converter.load_mel(input_file)
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print(f"Parameters:")
    print(f"  Sample rate: {converter.sample_rate} Hz")
    print(f"  N_FFT: {converter.n_fft}")
    print(f"  Hop length: {converter.hop_length}")
    print(f"  N_mels: {converter.n_mels}")
    
    # 멜 → 오디오 변환
    print()
    waveform = converter.mel_to_waveform(mel_spec)
    
    # 저장
    print()
    converter.save_audio(waveform, output_file)
    
    print("\n✓ Conversion completed!")


if __name__ == "__main__":
    main()