# SRT 자막 번역기

OpenAI GPT-4o-mini 또는 NLLB 모델을 선택하여 영어 SRT 자막을 자연스러운 한글로 번역하는 프로그램입니다.

## 기능

- 📁 SRT 파일 불러오기 및 저장
- 🤖 **듀얼 모델 지원**
    - **OpenAI GPT-4o-mini** (추천): 고품질 번역, 초저비용, GPU 불필요
    - **NLLB-200**: 오프라인 사용 가능, GPU 가속 지원
- 🎨 문체 선택 (존댓말/반말/자동)
- 📊 실시간 진행 상황 표시 (진행률, 남은 시간, 번역 속도)
- 💰 초저비용 (OpenAI 사용 시 자막 1000줄당 약 15~30원)
- 🖥️ 직관적인 GUI 인터페이스

## 설치 방법

### 0. API 키 설정

1. `.env.example` 파일을 복사하여 `.env` 파일 생성:

    ```bash
    cp .env.example .env
    ```

2. `.env` 파일을 열고 API 키 입력:

    ```bash
    # OpenAI API Key
    OPENAI_API_KEY=your_openai_api_key_here

    # Hugging Face Token
    HF_TOKEN=your_huggingface_token_here
    ```

### 1-1. OpenAI 모델 사용 (추천)

1. [OpenAI 플랫폼](https://platform.openai.com/)에 가입
2. [API Keys](https://platform.openai.com/api-keys)에서 새 API 키 생성
3. 위에서 생성한 키를 `.env` 파일의 `OPENAI_API_KEY`에 입력
4. 최소 $5 크레딧 충전 (수천 개 자막 번역 가능)

### 1-2. NLLB 모델 사용 (오프라인)

1. [Hugging Face](https://huggingface.co/)에 가입
2. [Settings > Tokens](https://huggingface.co/settings/tokens)에서 토큰 생성
3. 생성한 토큰을 `.env` 파일의 `HF_TOKEN`에 입력

PyTorch 설치 필요:

**GPU 가속 사용:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU 전용:**

```bash
pip install torch torchvision torchaudio
```

### 2. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

1. 프로그램 실행:

    ```bash
    python main.py
    ```

2. GUI에서 다음 단계를 따르세요:
    - **"번역 모델 선택"**에서 원하는 모델 선택
        - OpenAI GPT-4o-mini (추천): 고품질, 초저비용
        - NLLB: 오프라인, GPU 가속
    - "파일 선택" 버튼을 클릭하여 번역할 SRT 파일 선택
    - 원하는 문체 선택 (자동/존댓말/반말)
    - "번역 시작" 버튼 클릭

3. 번역이 완료되면 원본 파일과 같은 위치에 `_kor` 접미사가 붙은 파일이 생성됩니다.

## 비용 안내 (OpenAI 사용 시)

**OpenAI GPT-4o-mini 요금:**

- Input: $0.150 / 1M 토큰
- Output: $0.600 / 1M 토큰

**예상 비용:**

- 자막 100줄: 약 2~5원
- 자막 1000줄: 약 15~30원
- 영화 전체 (2000~3000줄): 약 50~100원

## 시스템 요구사항

**공통:**

- Python 3.8 이상

**OpenAI 모델:**

- 인터넷 연결 (API 호출)
- OpenAI API 키

**NLLB 모델:**

- 최소 8GB RAM
- GPU 사용 시: CUDA 지원 GPU (RTX 4060 이상 권장)
- 오프라인 사용 가능

## 사용 모델

### OpenAI GPT-4o-mini (추천)

- **제공자**: OpenAI
- **특징**: 빠르고 저렴하며 자연스러운 번역 품질
- **장점**: 고품질, 초저비용, GPU 불필요
- **단점**: 인터넷 연결 필요, API 비용 발생

### NLLB-200-distilled-600M

- **제공자**: Meta AI
- **특징**: 200개 언어 간 다국어 번역
- **장점**: 오프라인 사용 가능, 무료
- **단점**: GPU 권장, 번역 품질이 OpenAI보다 낮음

## 파일 구조

```
translate_srt/
├── main.py              # 메인 애플리케이션
├── translator_openai.py # OpenAI 번역 엔진
├── translator.py        # NLLB 번역 엔진
├── srt_parser.py        # SRT 파일 파서
├── gui.py               # GUI 인터페이스
├── requirements.txt     # 의존성 패키지
├── .env                 # API 키 설정 (git에 포함되지 않음)
├── .env.example         # API 키 설정 템플릿
└── README.md            # 문서
```

## 주의사항

**공통:**

- `.env` 파일에 API 키를 반드시 설정해야 합니다
- `.env` 파일은 git에 업로드하지 마세요 (민감한 정보 포함)

**OpenAI 모델:**

- API 키가 필요합니다
- API 사용량에 따라 과금됩니다 (매우 저렴)
- 인터넷 연결이 필요합니다

**NLLB 모델:**

- 처음 실행 시 모델 다운로드 (약 1.2GB)
- GPU 사용 시 번역 속도가 크게 향상됩니다
- PyTorch 설치 필요

## 라이선스

이 프로젝트는 개인 및 교육 목적으로 자유롭게 사용 가능합니다.
