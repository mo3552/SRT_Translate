"""OpenAI API를 사용한 번역 엔진"""
from openai import OpenAI
from typing import Optional, Callable, List
import time


class OpenAITranslator:
    """OpenAI GPT 모델 기반 번역기"""
    
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
        번역 프롬프트 생성
        
        Args:
            text: 번역할 텍스트
            tone: 문체 설정
            
        Returns:
            프롬프트 문자열
        """
        tone_instruction = {
            'formal': '존댓말(~습니다, ~해요)로 번역해주세요.',
            'casual': '반말(~어, ~야, ~해)로 번역해주세요.',
            'auto': '자연스러운 문체로 번역해주세요.'
        }
        
        tone_guide = tone_instruction.get(tone, tone_instruction['auto'])
        
        prompt = f"""다음 영어 자막을 자연스러운 한국어로 번역해주세요.

지침:
1. 영화/드라마 자막 스타일로 자연스럽게 번역
2. {tone_guide}
3. 관용구와 구어체 표현을 한국어 맥락에 맞게 의역
4. 번역문만 출력 (설명이나 주석 없이)
5. 줄바꿈은 원문과 동일하게 유지

영어 원문:
{text}

한국어 번역:"""
        
        return prompt
    
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
                        "content": "당신은 전문 영상 번역가입니다. 영어를 자연스러운 한국어로 번역합니다."
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
            
            return translated.strip()
            
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
        배치 번역
        
        Args:
            texts: 번역할 텍스트 리스트
            tone: 문체 ('formal', 'casual', 'auto')
            batch_size: 한 번에 번역할 자막 수 (OpenAI는 여러 개 묶어서 가능)
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
                combined_text = "\n".join([f"[{idx+1}] {text}" for idx, text in enumerate(non_empty_texts)])
                
                try:
                    # 번역
                    translated_combined = self.translate(combined_text, tone)
                    
                    # 번역 결과 분리
                    translations = []
                    lines = translated_combined.strip().split('\n')
                    
                    current_translation = []
                    for line in lines:
                        # [숫자] 패턴으로 시작하면 새 자막
                        if line.strip().startswith('[') and ']' in line:
                            if current_translation:
                                translations.append('\n'.join(current_translation))
                                current_translation = []
                            # [1] 제거하고 내용만 추출
                            content = line.split(']', 1)[1].strip() if ']' in line else line
                            current_translation.append(content)
                        else:
                            current_translation.append(line)
                    
                    # 마지막 번역 추가
                    if current_translation:
                        translations.append('\n'.join(current_translation))
                    
                    # 번역 개수가 맞지 않으면 개별 번역으로 폴백
                    if len(translations) != len(non_empty_texts):
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
