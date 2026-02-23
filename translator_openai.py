"""OpenAI API를 사용한 번역 엔진"""
from openai import OpenAI
from typing import Optional, Callable, List
import time
import re


class OpenAITranslator:
    """OpenAI GPT 모델 기반 번역기"""
    
    # 정식 자막 스타일 학습용 Few-shot 예시
    FEW_SHOT_EXAMPLES = [
        {
            "eng": "Colonel, give us the room, please.",
            "kor": "대령, 자리 좀 비켜 주시죠"
        },
        {
            "eng": "Four years retired yet two salutes?",
            "kor": "은퇴한 지 4년인데\n경례를 두 번이나?"
        },
        {
            "eng": "- Shows respect.\n- Have a seat.",
            "kor": "- 존경심의 표현입니다\n- 앉아요"
        },
        {
            "eng": "Do you know who I am?",
            "kor": "내가 누군지 아나요?"
        },
        {
            "eng": "Well, if you were to guess.",
            "kor": "한번 맞혀 보시죠"
        },
        {
            "eng": "Judging by your age, appearance,\nand how a Marine colonel\njust followed your order without a yip,",
            "kor": "나이와 외모\n해병대 대령이 군말 없이\n명령을 따른 걸 미루어 보면"
        },
        {
            "eng": "I'd guess you're some kind\nof high-level spook.",
            "kor": "꽤 높은 자리의\n정보원일 것 같군요"
        },
        {
            "eng": "Your father and grandfather served?",
            "kor": "아버지와 할아버지도\n군인이었죠?"
        },
        {
            "eng": "Yes.",
            "kor": "그렇습니다"
        },
        {
            "eng": "As an elite sniper,\nyou have 113 confirmed kills\nand an additional 81 probable kills.",
            "kor": "엘리트 저격수로서\n확인된 사살이 113건\n거기에 추정 사살 81건"
        }
    ]
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini"
    ):
        """
        번역기 초기화
        
        Args:
            api_key: OpenAI API 키
            model: 사용할 모델 (기본: gpt-4o-mini)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        print(f"OpenAI 번역기 초기화 완료!")
        print(f"모델: {model}")
    
    def check_api(self) -> dict:
        """
        API 연결 체크
        
        Returns:
            API 정보를 담은 딕셔너리
        """
        try:
            # 간단한 테스트 요청
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5
            )
            
            return {
                'available': True,
                'model': self.model,
                'status': 'Connected'
            }
        except Exception as e:
            return {
                'available': False,
                'model': self.model,
                'status': f'Error: {str(e)}'
            }
    
    def _create_translation_prompt(self, text: str, tone: str) -> str:
        """
        번역 프롬프트 생성 (정식 자막 수준의 고품질 번역)
        
        Args:
            text: 번역할 텍스트
            tone: 문체 설정
            
        Returns:
            프롬프트 문자열
        """
        tone_instruction = {
            'formal': '모든 대사를 존댓말(~습니다, ~해요)로 번역하되, 자막답게 간결하게.',
            'casual': '모든 대사를 반말(~어, ~야, ~해)로 번역하되, 자막답게 간결하게.',
            'auto': '대화 상황을 파악하여:\n   - 부하가 상급자/권위자에게: 존댓말 필수 (예: "그렇습니다", "~군요", "~죠?")\n   - 상급자가 부하에게: 해요체 또는 반말\n   - 단순 대답 "Yes" → 존댓말이면 "그렇습니다", 반말이면 "응"/"네"'
        }
        
        tone_guide = tone_instruction.get(tone, tone_instruction['auto'])
        
        # Few-shot 예시 생성 (모든 예시 사용)
        examples = "\n\n".join([
            f"예시 {i+1}:\n영문: {ex['eng']}\n한글: {ex['kor']}"
            for i, ex in enumerate(self.FEW_SHOT_EXAMPLES)
        ])
        
        prompt = f"""당신은 전문 영상 자막 번역가입니다. 아래 영어 자막을 정식 배포 자막 수준의 고품질 한국어로 번역하세요.

## 번역 원칙 (엄수 필수)

1. **자막 스타일**: 
   - 짧고 간결하게 (1~2줄 권장)
   - 불필요한 조사, 부사 과감히 생략
   - 구어체, 자연스러운 말투
   - 마침표(.) 사용 최소화

2. **줄바꿈 처리 (매우 중요!)**:
   - 원문에 줄바꿈(\\n)이 있으면 한국어도 반드시 줄바꿈 유지
   - 의미 단위로 줄을 나누되, 자연스럽게 조정
   - 예시에서 줄바꿈 위치를 정확히 학습할 것

3. **의역 우선**:
   - 직역 절대 금지, 한국어 관용 표현으로 의역
   - "give us the room" → "자리 좀 비켜 주시죠"
   - "without a yip" → "군말 없이"
   - "high-level spook" → "높은 자리의 정보원"
   - "served" (군 복무) → "군인이었다"

4. **문체 일관성 (중요!)**: 
   {tone_guide}

5. **생략과 압축**:
   - "당신의", "그것은" 등 불필요한 대명사 생략
   - 문맥상 명확한 주어/목적어는 과감히 생략
   - 자막 특성상 최대한 짧게

## 정식 자막 스타일 학습 예시

{examples}

## 번역 수행
- 위 예시의 스타일, 줄바꿈, 문체를 정확히 모방하세요
- 번역문만 출력 (설명, 주석, 따옴표 금지)
- 마침표는 가급적 생략

## 번역할 영어 자막

{text}

## 한국어 번역"""
        
        return prompt
    
    def _optimize_subtitle(self, text: str) -> str:
        """
        자막 최적화 후처리
        
        Args:
            text: 번역된 텍스트
            
        Returns:
            최적화된 텍스트
        """
        # 불필요한 표현 정리
        optimizations = [
            # 번역 투 제거
            (r'당신은\s+', ''),
            (r'당신의\s+', ''),
            (r'그것은\s+', ''),
            (r'저것은\s+', ''),
            (r'이것은\s+', ''),
            
            # 간결화
            (r'하셨나요\?', '하셨죠?'),
            (r'했나요\?', '했어?'),
            (r'입니까\?', '인가요?'),
            
            # 자연스러운 구어체
            (r'그리고\s+', ''),
            (r'하지만\s+', ''),
            (r'그러나\s+', ''),
            
            # 불필요한 마침표 제거 (자막 스타일)
            (r'\.$', ''),  # 문장 끝 마침표 제거
            (r'\.\s*\n', '\n'),  # 줄바꿈 앞 마침표 제거
            
            # 연속 공백 제거
            (r'\s+', ' '),
            (r'\n\s+', '\n'),
            (r'\s+\n', '\n'),
        ]
        
        result = text
        for pattern, replacement in optimizations:
            result = re.sub(pattern, replacement, result)
        
        return result.strip()
    
    def translate(
        self,
        text: str,
        tone: str = 'auto'
    ) -> str:
        """
        텍스트 번역
        
        Args:
            text: 번역할 텍스트
            tone: 문체 ('formal', 'casual', 'auto')
            
        Returns:
            번역된 텍스트
        """
        if not text.strip():
            return text
        
        try:
            prompt = self._create_translation_prompt(text, tone)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 넷플릭스, 디즈니+ 등에서 근무하는 전문 영상 자막 번역가입니다. 정식 배포 자막 수준의 고품질 번역을 제공합니다."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # 일관성 있는 번역
                max_tokens=1000
            )
            
            translated = response.choices[0].message.content
            if translated is None:
                return text
            
            # 후처리 최적화 적용
            optimized = self._optimize_subtitle(translated)
            return optimized
            
        except Exception as e:
            print(f"번역 오류: {e}")
            return text
    
    def translate_batch(
        self,
        texts: List[str],
        tone: str = 'auto',
        batch_size: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        배치 번역 (문맥 유지)
        
        Args:
            texts: 번역할 텍스트 리스트
            tone: 문체 ('formal', 'casual', 'auto')
            batch_size: 한 번에 번역할 자막 수 (5~10 권장)
            progress_callback: 진행 상황 콜백 함수 (current, total)
            
        Returns:
            번역된 텍스트 리스트
        """
        results = []
        total = len(texts)
        
        # 배치 단위로 처리 (여러 자막을 한 번에 번역하면 문맥 파악이 더 좋음)
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            
            # 빈 텍스트 처리
            non_empty_indices = [j for j, t in enumerate(batch) if t.strip()]
            non_empty_texts = [batch[j] for j in non_empty_indices]
            
            if non_empty_texts:
                # 여러 자막을 번호와 함께 묶어서 번역 (문맥 유지)
                combined_text = "\n\n".join([f"[{idx+1}]\n{text}" for idx, text in enumerate(non_empty_texts)])
                
                try:
                    # 배치 번역용 프롬프트
                    tone_instruction = {
                        'formal': '존댓말(~습니다, ~해요)을 사용하되, 자막답게 간결하게 번역하세요.',
                        'casual': '반말(~어, ~야, ~해)을 사용하되, 자막답게 간결하게 번역하세요.',
                        'auto': '대화 맥락에 맞는 자연스러운 문체를 사용하세요.'
                    }
                    
                    batch_prompt = f"""아래는 연속된 영화 자막입니다. 각 자막을 번호와 함께 정식 배포 자막 수준으로 번역하세요.

번역 원칙:
1. 짧고 간결하게 (자막 스타일)
2. {tone_instruction.get(tone, tone_instruction['auto'])}  
3. 직역 금지, 자연스러운 의역
4. 전후 맥락을 고려한 일관된 번역
5. 각 번역은 "[번호]" 형식으로 구분

영어 자막들:
{combined_text}

한국어 번역:"""
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "당신은 전문 영상 자막 번역가입니다. 연속된 자막의 문맥을 파악하여 일관성 있게 번역합니다."
                            },
                            {
                                "role": "user",
                                "content": batch_prompt
                            }
                        ],
                        temperature=0.3,
                        max_tokens=2000
                    )
                    
                    translated_combined = response.choices[0].message.content
                    if translated_combined is None:
                        raise Exception("번역 결과 없음")
                    
                    # 번역 결과 분리
                    translations = []
                    lines = translated_combined.strip().split('\n')
                    
                    current_translation = []
                    for line in lines:
                        # [숫자] 패턴으로 시작하면 새 자막
                        if line.strip().startswith('[') and ']' in line:
                            if current_translation:
                                trans_text = '\n'.join(current_translation).strip()
                                # 후처리 최적화
                                trans_text = self._optimize_subtitle(trans_text)
                                translations.append(trans_text)
                                current_translation = []
                            # [1] 제거하고 내용만 추출
                            content = line.split(']', 1)[1].strip() if ']' in line else ''
                            if content:
                                current_translation.append(content)
                        elif line.strip():
                            current_translation.append(line.strip())
                    
                    # 마지막 번역 추가
                    if current_translation:
                        trans_text = '\n'.join(current_translation).strip()
                        trans_text = self._optimize_subtitle(trans_text)
                        translations.append(trans_text)
                    
                    # 번역 개수가 맞지 않으면 개별 번역으로 폴백
                    if len(translations) != len(non_empty_texts):
                        print(f"배치 번역 개수 불일치 ({len(translations)} vs {len(non_empty_texts)}), 개별 번역으로 전환")
                        translations = [self.translate(t, tone) for t in non_empty_texts]
                    
                except Exception as e:
                    print(f"배치 번역 실패, 개별 번역으로 전환: {e}")
                    translations = [self.translate(t, tone) for t in non_empty_texts]
                    # API 제한 방지를 위한 짧은 대기
                    time.sleep(0.5)
                
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
            
            # API 레이트 리밋 방지 (짧은 대기)
            if i + batch_size < total:
                time.sleep(0.2)
        
        return results
