# SRT 자막 번역기

DeepL, OpenAI GPT-4o-mini 또는 NLLB 모델을 선택하여 영어 SRT 자막을 자연스러운 한글로 번역하는 프로그램입니다.

## 기능

- 📁 SRT 파일 불러오기 및 저장
- 🤖 **3가지 모델 지원**
    - **DeepL** (추천): 최고 품질, 빠른 속도, 자연스러운 번역
    - **OpenAI GPT-4o-mini**: 고품질 번역, 초저비용, GPU 불필요
    - **NLLB-200**: 오프라인 사용 가능, GPU 가속 지원
- 🎨 문체 선택 (존댓말/반말/자동)
- 📊 실시간 진행 상황 표시 (진행률, 남은 시간, 번역 속도)
- 💰 합리적인 비용 (DeepL/OpenAI 사용 시)
- 🖥️ 직관적인 GUI 인터페이스

## 설치 방법

### 0. API 키 설정

1. `.env.example` 파일을 복사하여 `.env` 파일 생성:

    ```bash
    cp .env.example .env
    ```

2. `.env` 파일을 열고 API 키 입력:

    ```bash
    # DeepL API Key (추천)
    DEEPL_API_KEY=your_deepl_api_key_here

    # OpenAI API Key
    OPENAI_API_KEY=your_openai_api_key_here

    # Hugging Face Token
    HF_TOKEN=your_huggingface_token_here
    ```

### 1-1. DeepL 모델 사용 (추천)

1. [DeepL Pro](https://www.deepl.com/en/pro-api)에 가입
2. API 플랜 선택:
    - **DeepL API Free**: 월 50만 자까지 무료
    - **DeepL API Pro**: 종량제 ($5.50/100만 자)
3. [계정 설정](https://www.deepl.com/account/summary)에서 API 키 생성
4. 생성한 키를 `.env` 파일의 `DEEPL_API_KEY`에 입력

### 1-2. OpenAI 모델 사용

1. [OpenAI 플랫폼](https://platform.openai.com/)에 가입
2. [API Keys](https://platform.openai.com/api-keys)에서 새 API 키 생성
3. 위에서 생성한 키를 `.env` 파일의 `OPENAI_API_KEY`에 입력
4. 최소 $5 크레딧 충전 (수천 개 자막 번역 가능)

### 1-3. NLLB 모델 사용 (오프라인)

1. [Hugging Face](https://huggingface.co/)에 가입
2. [Settings > Tokens](https://huggingface.co/settings/tokens)에서 토큰 생성
3. 생성한 토큰을 `.env` 파일의 `HF_TOKEN`에 입력

**GPU 하드웨어 가속 사용 시 (권장):**

NVIDIA GPU가 있는 경우 CUDA Toolkit과 cuDNN을 먼저 설치해야 합니다:

1. **CUDA Toolkit 설치**
    - [NVIDIA CUDA Toolkit 다운로드](https://developer.nvidia.com/cuda-downloads)
    - 권장 버전: CUDA 11.8 또는 12.x
    - 설치 후 시스템 재시작

2. **cuDNN 설치**
    - [NVIDIA cuDNN 다운로드](https://developer.nvidia.com/cudnn) (NVIDIA 계정 필요)
    - CUDA 버전에 맞는 cuDNN 선택
    - 다운로드한 파일을 CUDA 설치 폴더에 복사

3. **설치 확인**

    ```bash
    # CUDA 버전 확인
    nvcc --version

    # GPU 확인
    nvidia-smi
    ```

**PyTorch 설치:**

GPU 가속 사용 (CUDA 11.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

GPU 가속 사용 (CUDA 12.x):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

GPU 가속 사용 (CUDA 13.x):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

CPU 전용:

```bash
pip install torch torchvision
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
        - DeepL (추천): 최고 품질, 빠른 속도
        - OpenAI GPT-4o-mini: 고품질, 초저비용
        - NLLB: 오프라인, GPU 가속
    - "파일 선택" 버튼을 클릭하여 번역할 SRT 파일 선택
    - 원하는 문체 선택 (자동/존댓말/반말)
    - "번역 시작" 버튼 클릭

3. 번역이 완료되면 원본 파일과 같은 위치에 `_kor` 접미사가 붙은 파일이 생성됩니다.

## 비용 안내

### DeepL API 요금 (추천)

**DeepL API Free:**
- 무료: 월 50만 자까지
- 대부분의 영화/드라마 자막 번역 가능

**DeepL API Pro:**
- 종량제: $5.50 / 100만 자 (약 7,700원)

**예상 비용 (Pro 기준):**
- 자막 100줄: 약 4~8원
- 자막 1000줄: 약 40~80원
- 영화 전체 (2000~3000줄): 약 80~240원

### OpenAI GPT-4o-mini 요금

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

**DeepL 모델:**

- 인터넷 연결 (API 호출)
- DeepL API 키

**OpenAI 모델:**

- 인터넷 연결 (API 호출)
- OpenAI API 키

**NLLB 모델:**

- 최소 8GB RAM
- GPU 사용 시:
    - NVIDIA GPU (RTX 4060 이상 권장)
    - CUDA Toolkit 11.8 또는 12.x
    - cuDNN (CUDA 버전에 맞는 버전)
- 오프라인 사용 가능

## 사용 모델

### DeepL (추천)

- **제공자**: DeepL SE
- **특징**: 업계 최고 수준의 자연스러운 번역 품질
- **장점**: 최고 품질, 빠른 속도, 월 50만 자 무료, GPU 불필요
- **단점**: 인터넷 연결 필요, 무료 한도 초과 시 비용 발생

### OpenAI GPT-4o-mini

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
├── translator_deepl.py  # DeepL 번역 엔진
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

**DeepL 모델:**

- API 키가 필요합니다
- 월 50만 자까지 무료, 초과 시 종량제
- 인터넷 연결이 필요합니다

**OpenAI 모델:**

- API 키가 필요합니다
- API 사용량에 따라 과금됩니다 (매우 저렴)
- 인터넷 연결이 필요합니다

**NLLB 모델:**

- 처음 실행 시 모델 다운로드 (약 1.2GB)
- GPU 사용 시:
    - CUDA Toolkit과 cuDNN을 먼저 설치해야 합니다
    - 번역 속도가 크게 향상됩니다 (CPU 대비 5~10배 이상)
- PyTorch 설치 필요
- GPU 없이도 CPU 모드로 작동 가능 (속도 느림)

## 라이선스

이 프로젝트는 개인 및 교육 목적으로 자유롭게 사용 가능합니다.
