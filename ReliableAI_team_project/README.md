# 🌍 거대 다국어 ASR 모델의 적대적 공격 취약성 및 강건성 격차(Robustness Gap) 분석

## 가상환경

이 레포는 `CompareNoiseWaveMel`과 같은 작업공간 가상환경을 사용하도록
의존성을 맞췄습니다. 작업공간 루트(`/home/wjs/Univ/ReliableAI/Project`)에서
아래 명령을 실행하세요.

```bash
bash scripts/create_venv.sh cpu
source .venv/bin/activate
```

CUDA 환경은 `cpu` 대신 `cu121` 또는 `cu128`을 사용할 수 있습니다.

본 연구 프로젝트는 Meta의 **MMS(Massive Multilingual Speech) 1B** 모델을 대상으로, 적대적 섭동(Adversarial Perturbation)을 주입했을 때 발생하는 **언어별 음성 인식(ASR) 성능 저하 추이 및 강건성 격차(Robustness Gap)**를 정량적으로 분석하는 것을 목적으로 합니다. 

---

## 1. 연구 배경 및 목적 (Introduction)
최근 인공지능 기반 음성 인식(STT) 기술은 단일 언어를 넘어 수백 개의 다국어를 동시에 처리하는 거대 파운데이션 모델 형태로 진화했습니다. 그러나 이러한 모델들이 구조적으로 모든 언어에 대해 동일한 수준의 보안 강건성(Adversarial Robustness)을 보장하는지에 대한 연구는 미진합니다.

본 프로젝트는 고전적인 적대적 공격 기법인 **PGD(Projected Gradient Descent)**를 활용하여, 주류 언어(예: 영어)와 비주류 혹은 고유한 문자 체계를 가진 언어(예: 한국어, 중국어, 러시아어) 간의 **적대적 취약성 비대칭성**을 규명하고자 합니다.

> **💡 핵심 연구 가설 (Main Hypothesis)**
> * "MMS 모델은 사전 학습(Pre-training) 데이터의 양과 언어적 특성(음절 구조, 형태소 복잡도 등)에 따라 적대적 공격에 저항하는 내구도가 다를 것이다."
> * "단어 기반 지표(WER)와 글자 기반 지표(CER) 간의 비교를 통해, 특정 문자 체계가 공격에 더 취약하게 무너지는 현상을 증명할 수 있을 것이다."

---

## 2. 이론적 배경 및 공격 메커니즘 (Theoretical Background)

### 2.1 CTC Loss 기반 Gradient Ascent
MMS 모델은 시퀀스 종종 인코딩을 위해 **CTC(Connectionist Temporal Classification) Loss**를 사용합니다. 일반적인 딥러닝 학습이 Loss를 최소화하도록 모델 가중치 $\theta$를 업데이트하는 반면, 본 공격은 가중치를 고정한 채 **Loss를 극대화(Maximization)**하도록 입력 음성 파형(Waveform)을 변조합니다.

### 2.2 PGD-5 (Projected Gradient Descent) 수식 모델
본 실험에서 채택한 PGD-5 알고리즘은 미세 노이즈를 누적 주입하는 반복적(Iterative) 기법입니다. $t$번째 스텝에서의 공격 파형 $x^t$는 다음과 같이 정의됩니다.

$$x^{t+1} = \Pi_{x + \mathcal{S}} \left( x^t + \alpha \cdot \text{sgn}(\nabla_{x^t} \mathcal{L}(\theta, x^t, y)) \right)$$

* **$\mathcal{L}(\theta, x^t, y)$**: 모델 가중치 $\theta$와 정답 레이블 $y$에 대한 CTC 손실 함수
* **$\text{sgn}(\cdot)$**: 그라디언트의 방향만을 추출하는 부호 함수 (Sign Function)
* **$\alpha$**: 1회 반복당 주입되는 섭동의 보폭 크기 (Step size, 본 실험에서는 `0.001` 적용)
* **$\Pi_{x + \mathcal{S}}$**: 생성된 노이즈가 원본 신호의 허용 반경 $\epsilon$ (L-infinity bound, 본 실험에서는 `0.005`)을 초과하지 않도록 제한하는 투영(Projection) 연산자

---

## 3. 실험 환경 및 데이터셋 구조 (Experimental Setup)

### 3.1 대상 모델 (Target Model)
* **Model Name**: `facebook/mms-1b-all` (Meta Massive Multilingual Speech)
* **Architecture**: Wav2Vec 2.0 기반 + 언어별 전용 어댑터(Language Adapter) 탑재 구조

### 3.2 실험 데이터셋 (Dataset Specifications)
Google FLEURS 다국어 음성 데이터셋에서 아래 4개 대조군 언어를 선정하여 **언어별 1,000개(총 4,000개) 샘플**로 환경을 균등화(Equalization)하였습니다.

| 언어 태그 (Tag) | 대상 언어 (Language) | MMS 어댑터 코드 | 오디오 주파수 (Sampling Rate) | 샘플 수 (Rows) |
| :---: | :--- | :---: | :---: | :---: |
| **KO** | 한국어 (Korean) | `kor` | 16,000 Hz | 1,000 |
| **EN** | 영어 (English) | `eng` | 16,000 Hz | 1,000 |
| **ZH** | 중국어 (Chinese) | `cmn` | 16,000 Hz | 1,000 |
| **RU** | 러시아어 (Russian) | `rus` | 16,000 Hz | 1,000 |

---

## 4. 시스템 아키텍처 및 소스코드 구성 (System Architecture)

본 파이프라인은 데이터 변동을 최소화하고 실험 진행 상황을 유연하게 추적할 수 있도록 모듈화되어 설계되었습니다.

```text
team_project/
├── data/
│   ├── waveform/               # 4,000개의 밸런싱된 FLEURS 원본 데이터셋 (Arrow 포맷)
│   └── attack_results/
│       └── all_results.jsonl   # [실시간 기록] 공격 전후 텍스트 쌍 스트리밍 데이터
├── src/
│   ├── data_load.py            # 데이터셋 다국어 샘플링 및 16kHz 다운샘플링 뼈대
│   ├── attack.py               # PyTorch Tensor 기반 1차원 Waveform PGD 공격 엔진
│   └── analysis.py             # 수집된 JSONL 데이터를 기반으로 한 실시간 종합 통계기
└── exe.py                      # 전체 배치 실험 제어 및 실시간 파일 쓰기 버퍼(Flush) 관리자
