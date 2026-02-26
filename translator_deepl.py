"""DeepL API를 사용한 번역 엔진"""
import deepl
from typing import Optional, Callable, List
import time
import re


class DeepLTranslator:
    """DeepL API 기반 번역기"""
    
    def __init__(
        self,
        api_key: str
    ):
        """
        번역기 초기화
        
        Args:
            api_key: DeepL API 키
        """
        self.translator = deepl.Translator(api_key)
        self.api_key = api_key
        
        print(f"DeepL 번역기 초기화 완료!")
    
    def check_api(self) -> dict:
        """
        API 연결 체크
        
        Returns:
            API 정보를 담은 딕셔너리
        """
        try:
            # API 사용량 확인으로 연결 테스트
            usage = self.translator.get_usage()
            
            # 사용 가능한 문자 수 계산
            count = usage.character.count if usage.character.count is not None else 0
            limit = usage.character.limit
            
            if limit is not None:
                remaining = limit - count
                limit_info = f"{count:,} / {limit:,} 문자 사용"
            else:
                remaining = float('inf')
                limit_info = f"{count:,} 문자 사용 (무제한)"
            
            return {
                'available': True,
                'status': 'Connected',
                'usage': limit_info,
                'remaining': remaining
            }
        except Exception as e:
            return {
                'available': False,
                'status': f'Error: {str(e)}',
                'usage': 'N/A',
                'remaining': 0
            }
    
    def _apply_tone(self, text: str, tone: str) -> str:
        """
        문체 변환 적용
        
        Args:
            text: 변환할 텍스트
            tone: 문체 ('formal', 'casual', 'auto')
            
        Returns:
            변환된 텍스트
        """
        if tone == 'auto':
            return text
        
        # 간단한 문체 변환 패턴
        if tone == 'formal':
            # 반말 -> 존댓말
            patterns = [
                (r'이야$', '이에요'),
                (r'거야$', '거예요'),
                (r'할게$', '할게요'),
                (r'을게$', '을게요'),
                (r'([가-힣]+)어$', r'\1어요'),
                (r'([가-힣]+)아$', r'\1아요'),
                (r'([가-힣]+)해$', r'\1해요'),
            ]
        else:  # casual
            # 존댓말 -> 반말
            patterns = [
                (r'합니다$', '해'),
                (r'습니다$', '어'),
                (r'해요$', '해'),
                (r'어요$', '어'),
                (r'아요$', '아'),
                (r'이에요$', '이야'),
                (r'거예요$', '거야'),
            ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def translate(
        self,
        text: str,
        src_lang: str = 'EN',
        tgt_lang: str = 'KO',
        tone: str = 'auto'
    ) -> str:
        """
        단일 텍스트 번역
        
        Args:
            text: 번역할 텍스트
            src_lang: 소스 언어 코드 (EN, etc.)
            tgt_lang: 타겟 언어 코드 (KO, etc.)
            tone: 문체 ('auto', 'formal', 'casual')
            
        Returns:
            번역된 텍스트
        """
        try:
            # DeepL API로 번역
            result = self.translator.translate_text(
                text,
                source_lang=src_lang,
                target_lang=tgt_lang
            )
            
            # 결과가 리스트인 경우와 단일 객체인 경우 처리
            if isinstance(result, list):
                translated = result[0].text
            else:
                translated = result.text
            
            # 문체 변환 적용
            if tone != 'auto':
                translated = self._apply_tone(translated, tone)
            
            return translated
            
        except Exception as e:
            print(f"번역 오류: {e}")
            return text  # 오류 시 원문 반환
    
    def translate_batch(
        self,
        texts: List[str],
        src_lang: str = 'EN',
        tgt_lang: str = 'KO',
        tone: str = 'auto',
        batch_size: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        배치 번역
        
        Args:
            texts: 번역할 텍스트 리스트
            src_lang: 소스 언어 코드
            tgt_lang: 타겟 언어 코드
            tone: 문체 설정
            batch_size: 배치 크기
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            번역된 텍스트 리스트
        """
        total = len(texts)
        translated = []
        
        # 배치 단위로 처리
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # DeepL API는 여러 텍스트를 한 번에 번역 가능
                results = self.translator.translate_text(
                    batch,
                    source_lang=src_lang,
                    target_lang=tgt_lang
                )
                
                # 결과가 리스트가 아닐 수도 있음 (단일 항목인 경우)
                if not isinstance(results, list):
                    results = [results]
                
                # 번역 결과 추출 및 문체 변환
                for result in results:
                    text = result.text
                    if tone != 'auto':
                        text = self._apply_tone(text, tone)
                    translated.append(text)
                
            except Exception as e:
                print(f"배치 번역 오류: {e}")
                # 오류 시 원문 그대로 추가
                translated.extend(batch)
            
            # 진행 상황 업데이트
            if progress_callback:
                progress_callback(len(translated), total)
            
            # API 레이트 리밋 고려하여 짧은 대기
            if i + batch_size < total:
                time.sleep(0.1)
        
        return translated
    
    def get_supported_languages(self) -> dict:
        """
        지원하는 언어 목록 조회
        
        Returns:
            소스 언어와 타겟 언어 딕셔너리
        """
        try:
            source_langs = self.translator.get_source_languages()
            target_langs = self.translator.get_target_languages()
            
            return {
                'source': {lang.code: lang.name for lang in source_langs},
                'target': {lang.code: lang.name for lang in target_langs}
            }
        except Exception as e:
            print(f"언어 목록 조회 오류: {e}")
            return {'source': {}, 'target': {}}
