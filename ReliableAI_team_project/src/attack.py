import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC
import os
import soundfile as sf  # 🎵 오디오 저장을 위해 soundfile 임포트
from datasets import load_from_disk
from transformers import AutoProcessor

class AudioPGDAttacker:
    def __init__(self, model_name="facebook/mms-1b-all", device="cuda"):
        self.device = device
        print(f"Loading target STT model: {model_name}...")
        # MMS 모델 로드 및 평가 모드 고정
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def attack(self, input_waveform, labels, eps=0.005, alpha=0.001, iters=5):
        """
        1차원 오디오 파형(Waveform)에 대해 PGD-5 적대적 공격을 수행.
        
        Args:
            input_waveform (list or np.ndarray or torch.Tensor): 원본 1차원 음성 배열
            labels (list or torch.Tensor): 정수형 토큰 정답 레이블
            eps (float): 최대 노이즈 허용치 (소리가 찢어지지 않는 마지노선)
            alpha (float): 1스텝당 노이즈 주입 크기
            iters (int): 반복 횟수 (제안서 스펙인 PGD-5 기본 적용)
        """
        # 1. 데이터를 GPU/CPU 텐서로 변환하고 배치 차원(Batch dim) 추가
        orig_input = torch.tensor(input_waveform, dtype=torch.float32).unsqueeze(0).to(self.device)
        target_labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # 2. 원본 복사본에 그라디언트 추적 활성화
        adv_input = orig_input.clone().detach().requires_grad_(True)
        
        # PGD 반복 루프 수행 (PGD-5)
        for i in range(iters):
            # 모델 순전파 (MMS는 입력값과 레이블을 주면 자동으로 내부 CTC Loss를 계산함)
            outputs = self.model(adv_input, labels=target_labels)
            loss = outputs.loss
            
            # 그라디언트 초기화 및 역전파
            self.model.zero_grad()
            if adv_input.grad is not None:
                adv_input.grad.zero_grad()
            loss.backward()
            
            # 3. 경사 하강법의 반대 방향(Gradient Ascent)으로 노이즈 주입 (Loss 극대화)
            grad_sign = adv_input.grad.sign()
            adv_input = adv_input + alpha * grad_sign
            
            # 4. 투영(Projection) 스텝: 노이즈가 원본에서 eps 범위를 벗어나지 않게 가둠
            perturbation = torch.clamp(adv_input - orig_input, min=-eps, max=eps)
            # 오디오 신호의 물리적 한계선 [-1.0, 1.0] 내로 클리핑 후 다음 루프 준비
            adv_input = torch.clamp(orig_input + perturbation, min=-1.0, max=1.0).detach().requires_grad_(True)
            
        # 배치 차원 제거 후 CPU 넘파이 배열 형태로 반환
        perturbed_audio = adv_input.squeeze(0).detach().cpu().numpy()
        actual_noise = perturbed_audio - input_waveform
        
        return perturbed_audio, actual_noise

    def predict(self, input_waveform, processor):
        """공격 전후의 텍스트 결과를 비교하기 위한 추론 함수"""
        inputs = torch.tensor(input_waveform, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription[0]


if __name__ == "__main__":
    # 간단한 작동 테스트
    from datasets import load_from_disk
    from transformers import AutoProcessor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 우리가 아까 구워둔 4,000개 데이터셋 불러오기
    try:
        dataset = load_from_disk("./data/waveform")
        processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    except FileNotFoundError:
        print("데이터가 없습니다")
        exit()
        
    # 2. 테스트용으로 한국어(ko) 샘플 하나 집어오기
    ko_samples = [sample for sample in dataset if sample['lang_tag'] == 'ko']
    test_sample = ko_samples[0]
    
    # 3. 공격자 객체 생성
    attacker = AudioPGDAttacker(device=device)
    
    # MMS 토크나이저에게 지금 한국어 타겟팅한다고 명시
    attacker.model.load_adapter("kor")
    processor.tokenizer.set_target_lang("kor")
    
    # 원본 데이터셋 구조에 맞게 가공
    input_waveform = test_sample['audio']['array']
    labels = processor(text=test_sample['raw_transcription']).input_ids
    
    print("\n💥 Running PGD-5 Attack on Korean Sample...")
    adv_audio, noise = attacker.attack(
        input_waveform=input_waveform, # 수정됨
        labels=labels,                 # 수정됨
        eps=0.005, 
        alpha=0.001, 
        iters=5
    )
    
    # 4. 결과 검증 (여기도 수정된 변수명을 넣어줍니다)
    orig_text = attacker.predict(input_waveform, processor)
    adv_text = attacker.predict(adv_audio, processor)
    
    print("\n============= ATTACK RESULT =============")
    print(f"🌍 Language Tag: {test_sample['lang_tag']}")
    print(f"🟢 Original Model Prediction : '{orig_text}'")
    print(f"🔴 Adversarial Prediction    : '{adv_text}'")
    print(f"🛡️ Max Noise Amplitude (L-inf): {abs(noise).max():.5f}")
    print("=========================================")
    output_dir = "./data/attack_results"
    os.makedirs(output_dir, exist_ok=True)
    
    lang = test_sample['lang_tag']
    sampling_rate = test_sample['audio']['sampling_rate'] # 16000Hz
    
    # 파일명 정의
    orig_file_path = os.path.join(output_dir, f"fleurs_orig_{lang}.wav")
    adv_file_path = os.path.join(output_dir, f"fleurs_adv_{lang}.wav")
    
    # 오디오 파일 쓰기
    sf.write(orig_file_path, input_waveform, sampling_rate)
    sf.write(adv_file_path, adv_audio, sampling_rate)
    
    print(f"\n💾 Audio files successfully saved to '{output_dir}'")
    print(f"   - Clean Audio       : {orig_file_path}")
    print(f"   - Adversarial Audio : {adv_file_path}")