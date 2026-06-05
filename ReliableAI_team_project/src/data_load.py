import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import AutoProcessor

# 1. FLEURS용 코드 (Hugging Face 데이터셋 다운로드용)
FLEURS_LANG_CODES = {
    "ko": "ko_kr",
    "en": "en_us",
    "zh": "cmn_hans_cn",
    "ru": "ru_ru"
}

# 2. MMS 모델 내부용 ISO 639-3 코드 (토크나이저 타겟팅용)
MMS_LANG_CODES = {
    "ko": "kor",
    "en": "eng",
    "zh": "cmn-script_simplified",
    "ru": "rus"
}

def load_equalized_dataset(
    dataset_name="google/fleurs", 
    split="train", 
    languages=["ko", "en", "zh", "ru"], 
    samples_per_lang=10
):
    dataset_list = []
    for lang in languages:
        hf_lang_code = FLEURS_LANG_CODES.get(lang)
        print(f"Loading {lang.upper()} dataset ({hf_lang_code})...")
        
        ds = load_dataset(dataset_name, hf_lang_code, split=split)
        
        if len(ds) > samples_per_lang:
            ds = ds.select(range(samples_per_lang))
            
        ds = ds.add_column("lang_tag", [lang] * len(ds))
        dataset_list.append(ds)
        
    merged_dataset = concatenate_datasets(dataset_list)
    merged_dataset = merged_dataset.cast_column("audio", Audio(sampling_rate=16000))
    merged_dataset = merged_dataset.shuffle(seed=42)
    
    return merged_dataset


def get_prep_function(processor_name, model_type="waveform"):
    # 이제 mms-1b-all을 사용하므로 정상적으로 로드됩니다!
    processor = AutoProcessor.from_pretrained(processor_name)
    
    def prepare_dataset(batch):
        audio = batch["audio"]
        lang = batch["lang_tag"]
        
        # MMS 모델은 전처리 및 토크나이징 전에 목표 언어를 명시해 주어야 합니다.
        mms_code = MMS_LANG_CODES.get(lang)
        processor.tokenizer.set_target_lang(mms_code)
        
        if model_type == "waveform":
            inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"])
            batch["inputs"] = inputs.input_values[0]
        elif model_type == "mel":
            inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"])
            batch["inputs"] = inputs.input_features[0]
            
        # 정답 텍스트 인코딩
        batch["labels"] = processor(text=batch["raw_transcription"]).input_ids
            
        return batch

    return prepare_dataset, processor


if __name__ == "__main__":
    import os
    
    # 1. 처음이자 마지막으로 딱 한 번 인터넷 연결해서 40개 빌드
    raw_dataset = load_equalized_dataset(samples_per_lang=1000)
    
    # 2. 내 프로젝트 폴더 안으로 영구 저장
    output_dir = "./data/waveform"
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset.save_to_disk(output_dir)
    print(f"🎉 데이터셋 저장 완료! 경로: {output_dir}")