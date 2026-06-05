import soundfile as sf
import os
from src.data_load import load_equalized_dataset, get_prep_function

# 1. 0번 샘플 데이터 가져오기
raw_dataset = load_equalized_dataset(samples_per_lang=100)

for row in raw_dataset:
    if row['lang_tag'] == 'ko':  # 예시로 한국어 샘플을 선택
        sample = row
        break
print("샘플 데이터:")
print(sample)
audio_array = sample['audio']['array']
sampling_rate = sample['audio']['sampling_rate']
lang = sample['lang_tag']

# 2. 저장할 폴더 생성 (없다면)
os.makedirs("data/audio_test", exist_ok=True)

# 3. .wav 파일로 저장 (파일명에 언어 태그를 붙여서 구분하기 쉽게 함)
output_path = f"data/audio_test/sample_{lang}.wav"
sf.write(output_path, audio_array, sampling_rate)

print(f"✅ 오디오 파일이 성공적으로 저장되었습니다: {output_path}")
print(f"📝 해당 샘플의 정답 텍스트: {sample['raw_transcription']}")