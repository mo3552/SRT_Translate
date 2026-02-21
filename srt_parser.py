"""SRT 파일 파싱 및 저장 모듈"""
import re
from dataclasses import dataclass
from typing import List


@dataclass
class SubtitleEntry:
    """자막 엔트리 데이터 클래스"""
    index: int
    start_time: str
    end_time: str
    text: str
    
    def __str__(self):
        return f"{self.index}\n{self.start_time} --> {self.end_time}\n{self.text}\n"


class SRTParser:
    """SRT 파일 파서"""
    
    @staticmethod
    def parse(file_path: str) -> List[SubtitleEntry]:
        """
        SRT 파일을 파싱하여 SubtitleEntry 리스트로 반환
        
        Args:
            file_path: SRT 파일 경로
            
        Returns:
            SubtitleEntry 리스트
        """
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        # SRT 블록 분리 (빈 줄로 구분)
        blocks = re.split(r'\n\s*\n', content.strip())
        entries = []
        
        for block in blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            try:
                # 인덱스
                index = int(lines[0].strip())
                
                # 시간 정보
                time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                if not time_match:
                    continue
                    
                start_time = time_match.group(1)
                end_time = time_match.group(2)
                
                # 자막 텍스트 (여러 줄일 수 있음)
                text = '\n'.join(lines[2:])
                
                entries.append(SubtitleEntry(index, start_time, end_time, text))
                
            except (ValueError, IndexError):
                continue
        
        return entries
    
    @staticmethod
    def save(entries: List[SubtitleEntry], file_path: str):
        """
        SubtitleEntry 리스트를 SRT 파일로 저장
        
        Args:
            entries: SubtitleEntry 리스트
            file_path: 저장할 파일 경로
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(entries):
                f.write(str(entry))
                # 마지막 엔트리가 아니면 빈 줄 추가
                if i < len(entries) - 1:
                    f.write('\n')
