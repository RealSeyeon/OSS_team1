import torch
import numpy as np
import sys
import os

class MelAdversarialNoise:
    def __init__(self, epsilon=0.05, alpha=0.01, iterations=100):
        """
        멜 스펙트로그램에 적대적 노이즈 추가
        epsilon: 최대 perturbation 크기
        alpha: PGD step size
        iterations: 공격 반복 횟수
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_mel(self, mel_path):
        """멜 스펙트로그램 로드"""
        if mel_path.endswith('.npy'):
            # numpy 파일
            mel_spec = torch.from_numpy(np.load(mel_path)).to(self.device)
            
            # 메타데이터 로드
            meta_path = mel_path.replace('.npy', '_meta.pt')
            if os.path.exists(meta_path):
                metadata = torch.load(meta_path)
            else:
                metadata = {}
        else:
            # pytorch 파일
            data = torch.load(mel_path)
            mel_spec = data['mel_spectrogram'].to(self.device)
            metadata = {k: v for k, v in data.items() if k != 'mel_spectrogram'}
        
        return mel_spec, metadata
    
    def pgd_attack(self, mel_spec):
        """
        PGD (Projected Gradient Descent) 공격
        멜 스펙트로그램에 반복적으로 perturbation 추가
        """
        original_mel = mel_spec.clone().detach()
        perturbed_mel = original_mel.clone()
        
        print(f"\nStarting PGD attack ({self.iterations} iterations)...")
        
        for i in range(self.iterations):
            perturbed_mel.requires_grad = True
            
            # 대리 손실 함수
            loss = self._surrogate_loss(perturbed_mel)
            loss.backward()
            
            with torch.no_grad():
                # Gradient 방향으로 perturbation
                grad_sign = perturbed_mel.grad.sign()
                perturbed_mel = perturbed_mel + self.alpha * grad_sign
                
                # Epsilon ball 제약 (원본 기준)
                perturbation = perturbed_mel - original_mel
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                perturbed_mel = original_mel + perturbation
            
            if (i + 1) % 20 == 0:
                current_loss = loss.item()
                print(f"  Iteration {i+1}/{self.iterations} - Loss: {current_loss:.4f}")
        
        return perturbed_mel.detach()
    
    def fgsm_attack(self, mel_spec):
        """
        FGSM (Fast Gradient Sign Method) 공격
        단일 step으로 빠르게 perturbation 추가
        """
        mel_tensor = mel_spec.clone().detach().requires_grad_(True)
        
        print("\nApplying FGSM attack...")
        loss = self._surrogate_loss(mel_tensor)
        loss.backward()
        
        # Gradient의 부호를 이용한 perturbation
        perturbation = self.epsilon * mel_tensor.grad.sign()
        perturbed_mel = mel_tensor + perturbation
        
        print(f"  Loss: {loss.item():.4f}")
        
        return perturbed_mel.detach()
    
    def targeted_frequency_attack(self, mel_spec):
        """
        특정 주파수 대역을 타겟으로 하는 공격
        중간 주파수(20-60 mel bins)를 집중 공격
        """
        original_mel = mel_spec.clone().detach()
        perturbed_mel = original_mel.clone()
        
        # 타겟 주파수 대역 (20-60번째 mel bin)
        target_start = 20
        target_end = 60
        
        print(f"\nTargeted frequency attack (mel bins {target_start}-{target_end})...")
        
        for i in range(self.iterations):
            perturbed_mel.requires_grad = True
            
            # 타겟 대역에만 손실 계산
            target_region = perturbed_mel[:, target_start:target_end, :]
            loss = -torch.mean(torch.abs(target_region))  # 해당 영역 강화
            
            loss.backward()
            
            with torch.no_grad():
                # 전체 스펙트로그램에 perturbation
                grad_sign = perturbed_mel.grad.sign()
                perturbed_mel = perturbed_mel + self.alpha * grad_sign
                
                # Epsilon ball 제약
                perturbation = perturbed_mel - original_mel
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                perturbed_mel = original_mel + perturbation
            
            if (i + 1) % 20 == 0:
                print(f"  Iteration {i+1}/{self.iterations} - Loss: {loss.item():.4f}")
        
        return perturbed_mel.detach()
    
    def _surrogate_loss(self, mel_spec):
        """
        대리 손실 함수
        실제 타겟 모델이 있다면 그 모델의 출력 사용
        """
        # 1. 시간축 불연속성 증가
        time_diff = mel_spec[:, :, 1:] - mel_spec[:, :, :-1]
        time_loss = -torch.mean(torch.abs(time_diff))
        
        # 2. 주파수축 불연속성 증가
        freq_diff = mel_spec[:, 1:, :] - mel_spec[:, :-1, :]
        freq_loss = -torch.mean(torch.abs(freq_diff))
        
        # 3. 고주파 대역(40-80 mel bins) 강화
        if mel_spec.shape[1] >= 80:
            high_freq = mel_spec[:, 40:80, :]
            high_freq_loss = -torch.mean(torch.abs(high_freq))
        else:
            high_freq_loss = 0
        
        # 4. 전체 에너지 증가
        energy_loss = -torch.mean(torch.abs(mel_spec))
        
        # 총 손실
        loss = time_loss + freq_loss + 0.5 * high_freq_loss + 0.3 * energy_loss
        
        return loss
    
    def analyze_perturbation(self, original, perturbed):
        """Perturbation 분석"""
        diff = perturbed - original
        
        print(f"\n=== Perturbation Analysis ===")
        print(f"Max perturbation: {diff.abs().max().item():.4f} dB")
        print(f"Mean perturbation: {diff.abs().mean().item():.4f} dB")
        print(f"Std perturbation: {diff.std().item():.4f} dB")
        
        # 주파수 대역별 분석
        n_mels = original.shape[1]
        low_freq = diff[:, :n_mels//3, :].abs().mean()
        mid_freq = diff[:, n_mels//3:2*n_mels//3, :].abs().mean()
        high_freq = diff[:, 2*n_mels//3:, :].abs().mean()
        
        print(f"\nFrequency band analysis:")
        print(f"  Low freq (0-{n_mels//3}): {low_freq.item():.4f} dB")
        print(f"  Mid freq ({n_mels//3}-{2*n_mels//3}): {mid_freq.item():.4f} dB")
        print(f"  High freq ({2*n_mels//3}-{n_mels}): {high_freq.item():.4f} dB")
    
    def save_mel(self, mel_spec, output_path, metadata=None):
        """멜 스펙트로그램 저장"""
        if output_path.endswith('.npy'):
            # numpy 파일
            np.save(output_path, mel_spec.cpu().numpy())
            
            # 메타데이터 저장
            if metadata:
                meta_path = output_path.replace('.npy', '_meta.pt')
                torch.save(metadata, meta_path)
                print(f"Saved: {output_path}")
                print(f"Saved metadata: {meta_path}")
            else:
                print(f"Saved: {output_path}")
        else:
            # pytorch 파일
            save_data = {'mel_spectrogram': mel_spec.cpu()}
            if metadata:
                save_data.update(metadata)
            torch.save(save_data, output_path)
            print(f"Saved: {output_path}")


def main():
    # 기본 파일명 및 설정
    input_file = "original_mel.pt"
    output_file = "adversarial_mel.pt"
    method = "pgd"
    
    # 명령줄 인자가 있으면 사용
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    if len(sys.argv) >= 4:
        method = sys.argv[3]
    
    # 파라미터 파싱
    epsilon = 0.05
    alpha = 0.01
    iterations = 100
    
    i = 4 if len(sys.argv) >= 4 else 1
    while i < len(sys.argv):
        if sys.argv[i] == '--epsilon' and i + 1 < len(sys.argv):
            epsilon = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--alpha' and i + 1 < len(sys.argv):
            alpha = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--iterations' and i + 1 < len(sys.argv):
            iterations = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)
    
    print("=== Add Adversarial Noise to Mel Spectrogram ===")
    print(f"Method: {method.upper()}")
    print(f"Parameters:")
    print(f"  Epsilon: {epsilon}")
    print(f"  Alpha: {alpha}")
    print(f"  Iterations: {iterations}")
    
    # 노이즈 추가 실행
    attacker = MelAdversarialNoise(
        epsilon=epsilon,
        alpha=alpha,
        iterations=iterations
    )
    
    print(f"\nLoading: {input_file}")
    mel_spec, metadata = attacker.load_mel(input_file)
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    
    # 선택한 방법으로 공격
    print("\n" + "="*50)
    if method == "pgd":
        perturbed_mel = attacker.pgd_attack(mel_spec)
    elif method == "fgsm":
        perturbed_mel = attacker.fgsm_attack(mel_spec)
    elif method == "targeted":
        perturbed_mel = attacker.targeted_frequency_attack(mel_spec)
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)
    print("="*50)
    
    # 분석
    attacker.analyze_perturbation(mel_spec, perturbed_mel)
    
    # 저장
    print()
    attacker.save_mel(perturbed_mel, output_file, metadata)
    
    print("\n✓ Attack completed successfully!")



if __name__ == "__main__":
    main()