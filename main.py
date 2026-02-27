"""SRT 자막 번역기 메인 애플리케이션"""
import tkinter as tk
from pathlib import Path
import time
import sys
import os
from dotenv import load_dotenv

from translator_openai import OpenAITranslator
from translator_deepl import DeepLTranslator
from translator import NLLBTranslator
from srt_parser import SRTParser
from gui import TranslatorGUI

# .env 파일 로드
load_dotenv()


class SRTTranslatorApp:
    """자막 번역기 메인 애플리케이션"""
    
    def __init__(self):
        """애플리케이션 초기화"""
        # 환경 변수에서 API 키 로드
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.deepl_api_key = os.getenv('DEEPL_API_KEY')
        self.hf_token = os.getenv('HF_TOKEN')
        
        # API 키 검증
        if not self.openai_api_key:
            print("경고: OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.", file=sys.stderr)
        if not self.deepl_api_key:
            print("경고: DEEPL_API_KEY가 .env 파일에 설정되지 않았습니다.", file=sys.stderr)
        if not self.hf_token:
            print("경고: HF_TOKEN이 .env 파일에 설정되지 않았습니다.", file=sys.stderr)
        
        # GUI 생성
        self.root = tk.Tk()
        self.gui = TranslatorGUI(self.root)
        
        # 번역기 초기화 (None으로 시작)
        self.openai_translator = None
        self.deepl_translator = None
        self.nllb_translator = None
        
        # GUI 콜백 설정
        self.gui.set_translate_callback(self.translate_file)
        self.gui.set_model_change_callback(self._on_model_change)
        
        # 시작 시 모델 로드
        self._initialize()
    
    def _initialize(self):
        """초기화 작업"""
        # 기본 상태 체크
        self._check_environment()
        
        # 모델 로딩은 별도 스레드에서
        import threading
        thread = threading.Thread(target=self._load_models, daemon=True)
        thread.start()
    
    def _check_environment(self):
        """환경 체크"""
        try:
            status = "시스템 확인 중..."
            self.gui.update_cuda_status(status, True)
        except Exception as e:
            status = f"✗ 오류: {str(e)}"
            self.gui.update_cuda_status(status, False)
    
    def _on_model_change(self, model_type: str):
        """
        모델 변경 시 호출되는 콜백
        
        Args:
            model_type: 선택된 모델 타입 ('deepl', 'openai' 또는 'nllb')
        """
        if model_type == "deepl":
            # DeepL 모델이 이미 로드되어 있으면 상태 업데이트
            if self.deepl_translator:
                self.gui.update_cuda_status("✓ DeepL API 연결됨", True)
                self.gui.status_label.config(
                    text="DeepL 모델 선택됨 - 전문 번역 서비스, 최고 품질",
                    foreground="green"
                )
            else:
                self.gui.update_cuda_status("DeepL 초기화 대기 중...", True)
                self.gui.status_label.config(
                    text="DeepL 모델 초기화 중...",
                    foreground="blue"
                )
        elif model_type == "openai":
            # OpenAI 모델이 이미 로드되어 있으면 상태 업데이트
            if self.openai_translator:
                self.gui.update_cuda_status("✓ OpenAI API 연결됨", True)
                self.gui.status_label.config(
                    text="OpenAI 모델 선택됨 - 고품질 번역, GPU 불필요",
                    foreground="green"
                )
            else:
                self.gui.update_cuda_status("OpenAI 초기화 대기 중...", True)
                self.gui.status_label.config(
                    text="OpenAI 모델 초기화 중...",
                    foreground="blue"
                )
        else:  # nllb
            # NLLB 모델이 이미 로드되어 있으면 상태 업데이트
            if self.nllb_translator:
                cuda_info = self.nllb_translator.check_cuda()
                if cuda_info['available']:
                    status = f"✓ NLLB + {cuda_info['device_name']}"
                else:
                    status = "✓ NLLB (CPU 모드)"
                
                self.gui.update_cuda_status(status, cuda_info['available'])
                self.gui.status_label.config(
                    text="NLLB 모델 선택됨 - GPU 가속 권장, 오프라인 사용 가능",
                    foreground="green"
                )
            else:
                self.gui.update_cuda_status("NLLB 모델 미로드", False)
                self.gui.status_label.config(
                    text="NLLB 모델 선택됨 - 첫 번역 시 자동 로드됩니다",
                    foreground="blue"
                )
    
    def _load_models(self):
        """모델 로딩"""
        try:
            # DeepL API 키 확인
            if not self.deepl_api_key:
                raise ValueError("DEEPL_API_KEY가 .env 파일에 설정되지 않았습니다.")
            
            # 기본적으로 DeepL 모델 로드
            self.root.after(0, lambda: self.gui.status_label.config(
                text="DeepL 모델 초기화 중...",
                foreground="blue"
            ))
            
            # DeepL 번역기 초기화
            self.deepl_translator = DeepLTranslator(
                api_key=self.deepl_api_key
            )
            
            # API 연결 테스트
            api_info = self.deepl_translator.check_api()
            
            if api_info['available']:
                self.root.after(0, lambda: self.gui.update_cuda_status(
                    f"✓ DeepL 준비 완료",
                    True
                ))
                self.root.after(0, lambda: self.gui.status_label.config(
                    text=f"준비 완료! ({api_info['usage']}) SRT 파일을 선택해주세요.",
                    foreground="green"
                ))
                
                # 현재 선택된 모델의 상태 업데이트
                current_model = self.gui.model_var.get()
                self.root.after(100, lambda: self._on_model_change(current_model))
            else:
                raise Exception(api_info['status'])
            
        except Exception as e:
            error_msg = f"모델 초기화 실패: {str(e)}"
            self.root.after(0, lambda: self.gui.status_label.config(
                text=error_msg,
                foreground="red"
            ))
            self.root.after(0, lambda: self.gui.update_cuda_status(
                f"✗ 초기화 오류",
                False
            ))
            print(error_msg, file=sys.stderr)
    
    def _ensure_translator_loaded(self, model_type: str):
        """선택한 모델의 번역기가 로드되어 있는지 확인하고 로드"""
        if model_type == "deepl" and self.deepl_translator is None:
            # DeepL API 키 확인
            if not self.deepl_api_key:
                raise RuntimeError("DEEPL_API_KEY가 .env 파일에 설정되지 않았습니다.")
            
            # DeepL 번역기 로딩
            try:
                self.root.after(0, lambda: self.gui.status_label.config(
                    text="DeepL 모델 로딩 중...",
                    foreground="blue"
                ))
                
                self.deepl_translator = DeepLTranslator(
                    api_key=self.deepl_api_key
                )
                
                api_info = self.deepl_translator.check_api()
                if not api_info['available']:
                    raise Exception(api_info['status'])
                
                self.root.after(0, lambda: self.gui.update_cuda_status("✓ DeepL API 연결됨", True))
                self.root.after(0, lambda: self.gui.status_label.config(
                    text=f"DeepL 모델 준비 완료! ({api_info['usage']})",
                    foreground="green"
                ))
            except Exception as e:
                raise RuntimeError(f"DeepL 초기화 실패: {str(e)}")
        
        elif model_type == "openai" and self.openai_translator is None:
            # OpenAI API 키 확인
            if not self.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
            
            # OpenAI 번역기 로딩
            try:
                self.root.after(0, lambda: self.gui.status_label.config(
                    text="OpenAI 모델 로딩 중...",
                    foreground="blue"
                ))
                
                self.openai_translator = OpenAITranslator(
                    api_key=self.openai_api_key,
                    model="gpt-4o-mini"
                )
                
                api_info = self.openai_translator.check_api()
                if not api_info['available']:
                    raise Exception(api_info['status'])
                
                self.root.after(0, lambda: self.gui.update_cuda_status("✓ OpenAI API 연결됨", True))
                self.root.after(0, lambda: self.gui.status_label.config(
                    text="OpenAI 모델 준비 완료!",
                    foreground="green"
                ))
            except Exception as e:
                raise RuntimeError(f"OpenAI 초기화 실패: {str(e)}")
        
        elif model_type == "nllb" and self.nllb_translator is None:
            # NLLB 모델 로딩 (별도 스레드에서)
            import threading
            
            loading_complete = threading.Event()
            loading_error: list[Exception | None] = [None]  # 에러를 저장할 리스트
            
            def load_nllb():
                try:
                    self.root.after(0, lambda: self.gui.status_label.config(
                        text="NLLB 모델 로딩 중... (처음 실행 시 다운로드가 진행될 수 있습니다)",
                        foreground="blue"
                    ))
                    
                    self.nllb_translator = NLLBTranslator(
                        model_name="facebook/nllb-200-distilled-600M",
                        hf_token=self.hf_token
                    )
                    
                    # CUDA 상태 업데이트
                    cuda_info = self.nllb_translator.check_cuda()
                    if cuda_info['available']:
                        status = f"✓ NLLB + {cuda_info['device_name']}"
                    else:
                        status = "✓ NLLB (CPU 모드)"
                    
                    self.root.after(0, lambda: self.gui.update_cuda_status(status, cuda_info['available']))
                    self.root.after(0, lambda: self.gui.status_label.config(
                        text="NLLB 모델 로딩 완료!",
                        foreground="green"
                    ))
                    
                except Exception as e:
                    loading_error[0] = e
                    self.root.after(0, lambda: self.gui.status_label.config(
                        text=f"NLLB 모델 로딩 실패: {str(e)}",
                        foreground="red"
                    ))
                finally:
                    loading_complete.set()
            
            # 스레드 시작
            thread = threading.Thread(target=load_nllb, daemon=True)
            thread.start()
            
            # 로딩 완료 대기
            loading_complete.wait()
            
            # 에러가 있으면 예외 발생
            if loading_error[0]:
                raise loading_error[0]
    
    def translate_file(self, file_path: str, tone: str):
        """
        SRT 파일 번역
        
        Args:
            file_path: SRT 파일 경로
            tone: 문체 설정
        """
        # 선택한 모델 타입 가져오기
        model_type = self.gui.model_var.get()
        
        # 선택한 모델이 로드되어 있는지 확인
        try:
            self._ensure_translator_loaded(model_type)
        except Exception as e:
            raise RuntimeError(f"모델 로딩 실패: {str(e)}")
        
        # 진행 상황 초기화
        self.root.after(0, self.gui.reset_progress)
        
        # SRT 파일 파싱
        self.root.after(0, lambda: self.gui.status_label.config(
            text="SRT 파일 파싱 중...",
            foreground="blue"
        ))
        
        entries = SRTParser.parse(file_path)
        total = len(entries)
        
        if total == 0:
            raise ValueError("유효한 자막 엔트리를 찾을 수 없습니다.")
        
        # 번역할 텍스트 추출
        texts = [entry.text for entry in entries]
        
        # 번역 시작
        start_time = time.time()
        
        def progress_callback(current, total):
            """진행 상황 업데이트 콜백"""
            self.root.after(0, lambda: self.gui.update_progress(current, total, start_time))
        
        # 배치 번역 수행 (모델에 따라 다른 파라미터)
        if model_type == "deepl":
            if not self.deepl_translator:
                raise RuntimeError("DeepL 번역기가 초기화되지 않았습니다.")
            translated_texts = self.deepl_translator.translate_batch(
                texts,
                src_lang='EN',
                tgt_lang='KO',
                tone=tone,
                batch_size=50,  # DeepL은 대량 처리 가능
                progress_callback=progress_callback
            )
        elif model_type == "openai":
            if not self.openai_translator:
                raise RuntimeError("OpenAI 번역기가 초기화되지 않았습니다.")
            translated_texts = self.openai_translator.translate_batch(
                texts,
                tone=tone,
                batch_size=5,  # OpenAI는 문맥 파악을 위해 5개씩 묶음
                progress_callback=progress_callback
            )
        else:  # NLLB
            if not self.nllb_translator:
                raise RuntimeError("NLLB 번역기가 초기화되지 않았습니다.")
            translated_texts = self.nllb_translator.translate_batch(
                texts,
                src_lang='eng',
                tgt_lang='kor',
                tone=tone,
                batch_size=8,
                progress_callback=progress_callback
            )
        
        # 번역된 텍스트 적용
        for entry, translated in zip(entries, translated_texts):
            entry.text = translated
        
        # 출력 파일 경로 생성
        input_path = Path(file_path)
        output_path = input_path.parent / f"{input_path.stem}_kor{input_path.suffix}"
        
        # SRT 파일 저장
        SRTParser.save(entries, str(output_path))
        
        # 완료 메시지
        self.root.after(0, lambda: self.gui.show_completion(str(output_path)))
    
    def run(self):
        """애플리케이션 실행"""
        self.root.mainloop()


def main():
    """메인 함수"""
    try:
        app = SRTTranslatorApp()
        app.run()
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
