import os
import json
import io
import torch
import torchaudio
from tqdm import tqdm
from datasets import Audio, load_from_disk
from transformers import AutoProcessor
from src.attack import AudioPGDAttacker

TARGET_SAMPLE_RATE = 16000


def load_waveform(audio):
    if audio.get("bytes") is not None:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio["bytes"]))
    elif audio.get("path"):
        waveform, sample_rate = torchaudio.load(audio["path"])
    else:
        raise ValueError("Audio sample has neither embedded bytes nor a path.")

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    return waveform.flatten().clamp(-1.0, 1.0).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("📦 Loading dataset and base models...")
    dataset = load_from_disk("./data/waveform")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    attacker = AudioPGDAttacker(device=device)

    output_dir = "./data/attack_results"
    os.makedirs(output_dir, exist_ok=True)

    # MMS 언어 코드 매핑
    lang_to_mms = {    
        "ko": "kor",
        "en": "eng",
        "zh": "cmn-script_simplified",
        "ru": "rus"
    }
    output_path = os.path.join(output_dir, "all_results.jsonl")

    with open(output_path, "a", encoding="utf-8") as f:
        for lang_tag, mms_code in lang_to_mms.items():
            print(f"\n🌍 ============ Processing Language: {lang_tag.upper()} ============")
            
            attacker.model.load_adapter(mms_code)
            processor.tokenizer.set_target_lang(mms_code)
            
            lang_indices = [
                index for index, sample_lang in enumerate(dataset["lang_tag"])
                if sample_lang == lang_tag
            ]
            
            for sample_index in tqdm(lang_indices, desc=f"Attacking {lang_tag.upper()}"):
                sample = dataset[sample_index]
                input_waveform = load_waveform(sample["audio"])
                ground_truth = sample['raw_transcription'] 
                labels = processor(text=ground_truth).input_ids
                
                # PGD-5 공격 진행
                adv_audio, _ = attacker.attack(input_waveform, labels, eps=0.005, alpha=0.001, iters=5)
                
                # 예측
                clean_pred = attacker.predict(input_waveform, processor)
                adv_pred = attacker.predict(adv_audio, processor)
                
                # 저장할 데이터 딕셔너리 생성
                result_row = {
                    "lang_tag": lang_tag,
                    "ground_truth": ground_truth,
                    "clean_pred": clean_pred,
                    "adv_pred": adv_pred
                }
                
                # 1. 디스크에 한 줄 즉시 쓰기
                f.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                f.flush()  # 🔥 중요: 버퍼를 비워 OS가 파일에 즉시 기록하도록 강제합니다.
                
                # 2. 터미널 창에 실시간 결과 출력 (당장 확인용)
                print(f"\n⚡ [{lang_tag.upper()}] Real-time Result")
                print(f" 📑 정답 문장 (GT) : {ground_truth}")
                print(f" 🟢 공격 전 예측 (Clean): {clean_pred}")
                print(f" 🔴 공격 후 예측 (Adv)  : {adv_pred}")
                print("-" * 60)

    print(f"\n✅ 모든 공격 완료! 결과가 실시간으로 저장되었습니다: {output_path}")

if __name__ == "__main__":
    main()
