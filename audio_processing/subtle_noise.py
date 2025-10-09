import torch
import torchaudio
import numpy as np
import sys

class SubtleAudioAttack:
    def __init__(self, epsilon=0.002, alpha=0.0005, iterations=50):
        """
        귀에 거의 안 들리는 작은 perturbation
        epsilon: 매우 작은 값 (0.001-0.005 추천)
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_audio(self, audio_path):
        """오디오 로드"""
        waveform, sr = torchaudio.load(audio_path)
        
        # 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.to(self.device), sr
    
    def add_high_frequency_noise(self, waveform):
        """
        고주파 노이즈 추가 (사람 귀에 덜 민감)
        2kHz-8kHz 대역에 작은 노이즈
        """
        sample_rate = 16000
        
        # 고주파 필터 생성 (2kHz 이상)
        from scipy import signal
        nyquist = sample_rate / 2
        low_cutoff = 2000 / nyquist
        b, a = signal.butter(4, low_cutoff, btype='high')
        
        # 노이즈 생성
        noise = torch.randn_like(waveform) * self.epsilon
        
        # numpy로 변환하여 필터 적용
        noise_np = noise.cpu().numpy()
        filtered_noise = signal.filtfilt(b, a, noise_np)
        
        # torch로 다시 변환
        filtered_noise = torch.from_numpy(filtered_noise).to(self.device)
        
        perturbed = waveform + filtered_noise
        perturbed = torch.clamp(perturbed, -1.0, 1.0)
        
        return perturbed
    
    def add_psychoacoustic_noise(self, waveform):
        """
        심리음향학적 마스킹을 이용한 노이즈
        큰 소리 뒤에 작은 노이즈를 숨김
        """
        # 에너지가 큰 부분 찾기
        window_size = 400  # 25ms at 16kHz
        energy = waveform.abs().unfold(1, window_size, window_size//2).mean(2)
        
        # 에너지 임계값 이상인 부분에만 노이즈
        threshold = energy.mean() + energy.std()
        
        noise = torch.randn_like(waveform) * self.epsilon
        
        # 에너지가 큰 부분에만 노이즈 추가
        for i in range(0, waveform.shape[1] - window_size, window_size//2):
            window_energy = waveform[:, i:i+window_size].abs().mean()
            if window_energy > threshold:
                noise[:, i:i+window_size] *= 2.0  # 마스킹 효과 활용
            else:
                noise[:, i:i+window_size] *= 0.3  # 조용한 부분은 노이즈 줄임
        
        perturbed = waveform + noise
        perturbed = torch.clamp(perturbed, -1.0, 1.0)
        
        return perturbed
    
    def add_temporal_perturbation(self, waveform):
        """
        시간축 불연속성 추가 (딥러닝 모델 혼란)
        """
        perturbed = waveform.clone()
        
        # 랜덤하게 작은 시간 shift 추가
        for _ in range(self.iterations):
            # 랜덤 위치에 작은 perturbation
            idx = torch.randint(100, waveform.shape[1]-100, (10,))
            
            for i in idx:
                # 미세한 시간축 왜곡
                perturbation = torch.randn(1, 50).to(self.device) * self.epsilon
                perturbed[:, i:i+50] += perturbation
        
        perturbed = torch.clamp(perturbed, -1.0, 1.0)
        return perturbed
    
    def add_minimal_pgd_noise(self, waveform):
        """
        최소한의 PGD 노이즈 (매우 작은 epsilon)
        """
        original = waveform.clone()
        perturbed = waveform.clone()
        
        print(f"\nApplying minimal PGD attack...")
        
        for i in range(self.iterations):
            perturbed.requires_grad = True
            
            # 간단한 손실: 미분값 최대화
            diff = perturbed[:, 1:] - perturbed[:, :-1]
            loss = -torch.mean(torch.abs(diff))
            
            loss.backward()
            
            with torch.no_grad():
                grad = perturbed.grad
                perturbed = perturbed + self.alpha * grad.sign()
                
                # Epsilon ball 제약
                delta = perturbed - original
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                perturbed = original + delta
                perturbed = torch.clamp(perturbed, -1.0, 1.0)
            
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1}/{self.iterations}")
        
        return perturbed
    
    def analyze_difference(self, original, perturbed):
        """차이 분석"""
        diff = perturbed - original
        
        snr = 20 * torch.log10(torch.norm(original) / torch.norm(diff))
        
        print(f"\n=== Audio Quality Analysis ===")
        print(f"Max difference: {diff.abs().max().item():.6f}")
        print(f"Mean difference: {diff.abs().mean().item():.6f}")
        print(f"SNR: {snr.item():.2f} dB (높을수록 원본과 비슷)")
        print(f"Perturbation energy: {(diff**2).mean().sqrt().item():.6f}")
    
    def save_audio(self, waveform, sr, output_path):
        """오디오 저장"""
        # 정규화
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val * 0.95
        
        torchaudio.save(output_path, waveform.cpu(), sr)
        print(f"\nSaved: {output_path}")


def main():
    # 기본 설정
    input_file = "original_audio.wav"
    output_file = "adversarial_audio.wav"
    method = "minimal"  # minimal, high_freq, psycho, temporal
    
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    if len(sys.argv) >= 4:
        method = sys.argv[3]
    
    print("=== Subtle Audio Adversarial Attack ===")
    print(f"Method: {method}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # 공격 실행
    attacker = SubtleAudioAttack(
        epsilon=0.002,   # 매우 작은 값 (귀에 거의 안 들림)
        alpha=0.0005,
        iterations=50
    )
    
    print(f"\nLoading audio...")
    waveform, sr = attacker.load_audio(input_file)
    print(f"Duration: {waveform.shape[1] / sr:.2f}s")
    
    # 방법 선택
    print(f"\nApplying {method} attack...")
    
    if method == "minimal":
        perturbed = attacker.add_minimal_pgd_noise(waveform)
    elif method == "high_freq":
        perturbed = attacker.add_high_frequency_noise(waveform)
    elif method == "psycho":
        perturbed = attacker.add_psychoacoustic_noise(waveform)
    elif method == "temporal":
        perturbed = attacker.add_temporal_perturbation(waveform)
    else:
        print(f"Unknown method: {method}")
        print("Available: minimal, high_freq, psycho, temporal")
        sys.exit(1)
    
    # 분석
    attacker.analyze_difference(waveform, perturbed)
    
    # 저장
    attacker.save_audio(perturbed, sr, output_file)
    
    print("\n✓ Attack completed!")
    print("\nRecommendations:")
    print("  - 'minimal': 가장 작은 변화, SNR > 40dB (추천)")
    print("  - 'high_freq': 고주파 노이즈, 귀에 덜 민감")
    print("  - 'psycho': 심리음향학적 마스킹 활용")
    print("  - 'temporal': 시간축 교란")


if __name__ == "__main__":
    main()