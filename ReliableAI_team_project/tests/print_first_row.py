#src/data_load.py에서 데이터 로드 및 전처리 함수들을 가져와서 테스트하는 코드입니다.
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import AutoProcessor
from src.data_load import load_equalized_dataset, get_prep_function

if __name__ == "__main__":
    # 데이터 로드 (테스트용 언어당 10개)
    raw_dataset = load_equalized_dataset(samples_per_lang=10)
    
    print("Raw dataset loaded:")
    print(raw_dataset[0])