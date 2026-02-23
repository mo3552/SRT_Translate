"""NLLB 모델을 사용한 번역 엔진"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, Callable
import re


class NLLBTranslator:
    """NLLB 모델 기반 번역기"""
    
    # 언어 코드 매핑
    LANG_CODES = {
        'eng': 'eng_Latn',  # 영어
        'kor': 'kor_Hang',  # 한국어
    }
    
    # 문체 변환 패턴 (순서가 중요함 - 구체적인 패턴부터 매칭)
    TONE_PATTERNS = {
        'formal': [  # 존댓말 (자막 스타일)
            # 반말 -> 존댓말 변환
            (r'거야', '거예요'),
            (r'할게', '할게요'),
            (r'을게', '을게요'),
            (r'([가-힣]+)어$', r'\1어요'),
            (r'([가-힣]+)아$', r'\1아요'),
            (r'([가-힣]+)야$', r'\1예요'),
            (r'([가-힣]+)해$', r'\1해요'),
            (r'([가-힣]+)지$', r'\1지요'),
            (r'([가-힣]+)네$', r'\1네요'),
            (r'([가-힣]+)데$', r'\1데요'),
            (r'([가-힣]+)군$', r'\1군요'),
        ],
        'casual': [  # 반말 (자막 스타일)
            # 존댓말 -> 반말 변환 (구체적인 표현부터)
            (r'감사합니다', '고마워'),
            (r'고맙습니다', '고마워'),
            (r'환영합니다', '환영해'),
            (r'죄송합니다', '미안해'),
            (r'([가-힣]+)하겠습니다', r'\1할게'),
            (r'([가-힣]+)겠습니다', r'\1을게'),
            (r'([가-힣]+)ㄹ 것입니다', r'\1을 거야'),
            (r'것입니다', '거야'),
            (r'([가-힣]+)였습니다', r'\1었어'),
            (r'([가-힣]+)았습니다', r'\1았어'),
            (r'([가-힣]+)었습니다', r'\1었어'),
            (r'([가-힣]+)했습니다', r'\1했어'),
            (r'([가-힣]+)합니다', r'\1해'),
            (r'([가-힣]+)습니다', r'\1어'),
            (r'([가-힣]+)ㅂ니다', r'\1어'),
            (r'([가-힣]+)십시오', r'\1어'),
            (r'([가-힣]+)세요', r'\1어'),
            (r'([가-힣]+)해요', r'\1해'),
            (r'([가-힣]+)어요', r'\1어'),
            (r'([가-힣]+)아요', r'\1아'),
            (r'([가-힣]+)예요', r'\1야'),
            (r'([가-힣]+)이에요', r'\1이야'),
            (r'([가-힣]+)지요', r'\1지'),
            (r'([가-힣]+)네요', r'\1네'),
            (r'([가-힣]+)데요', r'\1데'),
            (r'([가-힣]+)군요', r'\1군'),
            (r'거예요', '거야'),
        ]
    }
    
    # 자연스러운 표현으로 변환 (직역투 개선 + 자막 스타일)
    NATURAL_EXPRESSIONS = [
        # === 자막 스타일 개선 ===
        
        # 불필요한 표현 제거
        (r'당신은\s+', ''),
        (r'당신의\s+', ''),
        (r'그것은\s+', ''),
        (r'저것은\s+', ''),
        (r'이것은\s+', ''),
        
        # 간결한 표현
        (r'~하셨나요\?', '~하셨죠?'),
        (r'~했나요\?', '~했어?'),
        (r'~입니까\?', '~인가요?'),
        (r'~할 수 있습니까', '~할 수 있어요'),
        (r'~할 수 있나요', '~할 수 있어'),
        
        # 구어체 변환
        (r'그리고\s+', ''),
        (r'하지만\s+', ''),
        (r'그러나\s+', ''),
        (r'또한\s+', ''),
        (r'게다가\s+', ''),
        
        # === 군사/액션 영화 관용 표현 ===
        
        # "give us the room" 패턴
        (r'방을 비워\s*주[세요|시오|십시오|시죠]', '자리 좀 비켜 주시죠'),
        (r'방을 내어\s*주[세요|시오]', '자리 좀 비켜 주시죠'),
        (r'우리에게 방을', '자리를'),
        
        # "without a yip" 패턴
        (r'아무\s*말\s*없이', '군말 없이'),
        (r'소리\s*없이', '군말 없이'),
        (r'한\s*마디도\s*없이', '군말 없이'),
        
        # "high-level spook" 패턴
        (r'고위\s*급?\s*스파이', '높은 자리의 정보원'),
        (r'고위\s*급?\s*첩보원', '높은 자리의 정보원'),
        (r'높은\s*수준의\s*스파이', '높은 자리의 정보원'),
        
        # "retired" 패턴
        (r'퇴직한\s*지', '은퇴한 지'),
        (r'퇴역한\s*지', '은퇴한 지'),
        
        # "salute" 패턴  
        (r'경례를\s*받', '경례'),
        
        # === 일반 관용 표현 ===
        
        # 잘못된 번역 수정
        (r'귀여워요\?', '여보'),
        (r'귀여운', '여보'),
        (r'준비됐어이다', '준비됐어'),
        (r'준비된이다', '준비됐어'),
        (r'([가-힣]+)어이다', r'\1어'),
        (r'([가-힣]+)았어이다', r'\1았어'),
        (r'([가-힣]+)었어이다', r'\1었어'),
        
        # 문법 오류 수정
        (r'잘 맞았어 것 같았다', '잘 맞는 것 같았다'),
        (r'([가-힣]+)어 것 같[았|다]', r'\1는 것 같았어'),
        (r'소울메이트과', '소울메이트'),
        (r'정착할 생각 노력하지', '정착하려고 하지'),
        (r'정착할 생각이 노력', '정착할 생각'),
        
        # 영어 관용구의 자연스러운 번역
        (r'클릭하[는다|었|습니다|어요|았어]', '잘 맞았어'),
        (r'클릭하는 것 같[았|습니다|았어|았다]', '잘 맞는 것 같았어'),
        (r'단지 클릭', '딱 맞'),
        (r'혼동동생', '소울메이트'),
        (r'영혼의 짝', '소울메이트'),
        (r'영혼 동료', '소울메이트'),
        
        # 부자연스러운 표현 개선
        (r'충분히 무죄', '아주 순수'),
        (r'충분히 순수하게', '순수하게'),
        (r'무죄로', '순수하게'),
        (r'정착하기 위해 노력', '정착하려'),
        (r'정착하기 위해', '정착할 생각으로'),
        (r'노력하지 않았음에도 불구하고', '생각이 없었지만'),
        (r'나이가 많[았|습니다|았어]', '나이가 더 많았어'),
        (r'보다 나이가 많', '보다 나이가 더 많'),
        (r'동일한 목표', '같은 목표'),
        (r'동일한', '같은'),
        (r'파트너의 기대가 동일', '파트너에 대한 기대도 같'),
        (r'기대가 같', '기대도 같'),
        
        # 자연스러운 연결
        (r'한 가지는 다른 것을 이끌', '자연스럽게 하나씩 진행되'),
        (r'그것은 방금 일어났', '그냥 일어났'),
        (r'그것은 단지 일어났', '그냥 일어났'),
        (r'정말로 발생했', '정말 일어났'),
        
        # 과거형 통일
        (r'가지고 있었다', '가지고 있었어'),
        (r'같았다$', '같았어'),
        
        # === 자막 길이 최적화 ===
        
        # 장황한 표현 축약
        (r'~하는 것이 가능합니다', '~할 수 있어요'),
        (r'~하는 것이 가능해요', '~할 수 있어요'),
        (r'~하지 않을까요\?', '~하지 않을까?'),
        (r'~라고 생각합니다', '~라고 봐요'),
        (r'~라고 생각해요', '~라고 봐요'),
        
        # 연속 공백 제거
        (r'\s{2,}', ' '),
    ]
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        hf_token: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        번역기 초기화
        
        Args:
            model_name: 허깅페이스 모델 이름
            hf_token: 허깅페이스 토큰
            device: 사용할 디바이스 ('cuda', 'cpu' 또는 None=자동 선택)
        """
        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"디바이스: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 버전: {torch.version.cuda}")
        
        # 모델 및 토크나이저 로드
        print("모델 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            src_lang=self.LANG_CODES['eng']
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            token=hf_token
        ).to(self.device)
        
        # 평가 모드로 설정 (추론 최적화)
        self.model.eval()
        
        print("모델 로딩 완료!")
    
    def _get_lang_token_id(self, lang_code: str) -> int:
        """
        언어 코드를 token ID로 변환
        
        Args:
            lang_code: 언어 코드 (예: 'kor_Hang')
            
        Returns:
            token ID
        """
        return self.tokenizer.convert_tokens_to_ids(lang_code)
    
    def check_cuda(self) -> dict:
        """
        CUDA 환경 체크
        
        Returns:
            CUDA 정보를 담은 딕셔너리
        """
        cuda_info = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': None,
            'device_name': None,
            'cuda_version': torch.version.cuda,
        }
        
        if cuda_info['available']:
            cuda_info['current_device'] = torch.cuda.current_device()
            cuda_info['device_name'] = torch.cuda.get_device_name(0)
        
        return cuda_info
    
    def apply_tone(self, text: str, tone: str) -> str:
        """
        번역된 텍스트에 문체 적용 및 자연스러운 표현으로 변환
        
        Args:
            text: 번역된 텍스트
            tone: 문체 ('formal', 'casual', 'auto')
            
        Returns:
            문체가 적용되고 자연스럽게 변환된 텍스트
        """
        # 문장 단위로 분리 (개행 포함)
        lines = text.split('\n')
        result_lines = []
        
        for line in lines:
            modified_line = line
            
            # 1단계: 자연스러운 표현으로 먼저 변환
            for pattern, replacement in self.NATURAL_EXPRESSIONS:
                modified_line = re.sub(pattern, replacement, modified_line)
            
            # 2단계: 문체 적용 (formal/casual)
            if tone != 'auto' and tone in self.TONE_PATTERNS:
                patterns = self.TONE_PATTERNS[tone]
                for pattern, replacement in patterns:
                    modified_line = re.sub(pattern, replacement, modified_line)
            
            result_lines.append(modified_line)
        
        return '\n'.join(result_lines)
    
    def translate(
        self,
        text: str,
        src_lang: str = 'eng',
        tgt_lang: str = 'kor',
        tone: str = 'auto',
        max_length: int = 512
    ) -> str:
        """
        텍스트 번역
        
        Args:
            text: 번역할 텍스트
            src_lang: 원본 언어 코드
            tgt_lang: 대상 언어 코드
            tone: 문체 ('formal', 'casual', 'auto')
            max_length: 최대 토큰 길이
            
        Returns:
            번역된 텍스트
        """
        if not text.strip():
            return text
        
        # 언어 코드 변환
        src_code = self.LANG_CODES.get(src_lang, src_lang)
        tgt_code = self.LANG_CODES.get(tgt_lang, tgt_lang)
        
        # 토크나이저 설정
        self.tokenizer.src_lang = src_code
        
        # 입력 준비
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # 번역 수행 (no_grad로 메모리 최적화)
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self._get_lang_token_id(tgt_code),
                max_length=max_length,
                num_beams=8,  # 빔 서치 증가로 품질 향상
                length_penalty=1.0,  # 적절한 길이 유지
                repetition_penalty=1.2,  # 반복 방지
                no_repeat_ngram_size=3,  # n-gram 반복 방지
                early_stopping=True
            )
        
        # 디코딩
        translated = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        # 문체 적용
        translated = self.apply_tone(translated, tone)
        
        return translated
    
    def translate_batch(
        self,
        texts: list,
        src_lang: str = 'eng',
        tgt_lang: str = 'kor',
        tone: str = 'auto',
        max_length: int = 512,
        batch_size: int = 8,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list:
        """
        배치 번역 (최적화된 처리)
        
        Args:
            texts: 번역할 텍스트 리스트
            src_lang: 원본 언어 코드
            tgt_lang: 대상 언어 코드
            tone: 문체 ('formal', 'casual', 'auto')
            max_length: 최대 토큰 길이
            batch_size: 배치 크기
            progress_callback: 진행 상황 콜백 함수 (current, total)
            
        Returns:
            번역된 텍스트 리스트
        """
        results = []
        total = len(texts)
        
        # 언어 코드 변환
        src_code = self.LANG_CODES.get(src_lang, src_lang)
        tgt_code = self.LANG_CODES.get(tgt_lang, tgt_lang)
        self.tokenizer.src_lang = src_code
        
        # 배치 단위로 처리
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            
            # 빈 텍스트 처리
            non_empty_indices = [j for j, t in enumerate(batch) if t.strip()]
            non_empty_texts = [batch[j] for j in non_empty_indices]
            
            if non_empty_texts:
                # 토크나이징
                inputs = self.tokenizer(
                    non_empty_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # 번역
                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self._get_lang_token_id(tgt_code),
                        max_length=max_length,
                        num_beams=8,  # 빔 서치 증가
                        length_penalty=1.0,  # 적절한 길이 유지
                        repetition_penalty=1.2,  # 반복 방지
                        no_repeat_ngram_size=3,  # n-gram 반복 방지
                        early_stopping=True
                    )
                
                # 디코딩
                translations = self.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True
                )
                
                # 문체 적용
                translations = [self.apply_tone(t, tone) for t in translations]
                
                # 결과 재구성
                batch_results = [''] * len(batch)
                for idx, trans in zip(non_empty_indices, translations):
                    batch_results[idx] = trans
            else:
                batch_results = [''] * len(batch)
            
            results.extend(batch_results)
            
            # 진행 상황 콜백
            if progress_callback:
                progress_callback(min(i + batch_size, total), total)
        
        return results
