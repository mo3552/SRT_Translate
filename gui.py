"""GUI 인터페이스"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import time
from typing import Optional, Callable


class TranslatorGUI:
    """자막 번역기 GUI"""
    
    def __init__(self, root: tk.Tk):
        """
        GUI 초기화
        
        Args:
            root: Tkinter 루트 윈도우
        """
        self.root = root
        self.root.title("SRT 자막 번역기")
        self.root.geometry("700x700")
        self.root.resizable(False, False)
        
        # 번역 콜백 함수
        self.translate_callback: Optional[Callable] = None
        self.model_change_callback: Optional[Callable] = None  # 모델 변경 콜백
        
        # 변수
        self.file_path = tk.StringVar()
        self.title_var = tk.StringVar()  # 작품명 (선택사항)
        self.model_var = tk.StringVar(value="deepl")  # 모델 선택 (DeepL 기본값)
        self.tone_var = tk.StringVar(value="auto")
        self.cuda_status = tk.StringVar(value="확인 중...")
        
        # GUI 구성
        self._create_widgets()
        
    def _create_widgets(self):
        """위젯 생성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # 제목
        title_label = ttk.Label(
            main_frame,
            text="SRT 자막 번역기",
            font=("맑은 고딕", 18, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # CUDA 상태
        cuda_frame = ttk.LabelFrame(main_frame, text="시스템 정보", padding="10")
        cuda_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 15))
        
        ttk.Label(cuda_frame, text="GPU 상태:").grid(row=0, column=0, sticky=tk.W)
        cuda_label = ttk.Label(cuda_frame, textvariable=self.cuda_status, foreground="blue")
        cuda_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 모델 선택
        model_frame = ttk.LabelFrame(main_frame, text="번역 모델 선택", padding="10")
        model_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 15))
        
        ttk.Radiobutton(
            model_frame,
            text="DeepL (추천: 최고 품질, 빠른 속도)",
            variable=self.model_var,
            value="deepl",
            command=self._on_model_change
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        ttk.Radiobutton(
            model_frame,
            text="OpenAI GPT-4o-mini (고품질, 초저비용)",
            variable=self.model_var,
            value="openai",
            command=self._on_model_change
        ).grid(row=1, column=0, sticky=tk.W, padx=(0, 20))
        
        ttk.Radiobutton(
            model_frame,
            text="NLLB (GPU 필요, 오프라인)",
            variable=self.model_var,
            value="nllb",
            command=self._on_model_change
        ).grid(row=2, column=0, sticky=tk.W, padx=(0, 20))
        
        # 작품명 입력 (선택 사항)
        title_frame = ttk.LabelFrame(main_frame, text="작품명 (선택 사항)", padding="10")
        title_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 15))
        
        title_entry = ttk.Entry(
            title_frame,
            textvariable=self.title_var,
            width=50
        )
        title_entry.grid(row=0, column=0, sticky="ew")
        
        ttk.Label(
            title_frame,
            text="작품명을 입력하면 번역 품질이 향상됩니다 (장르별 문체, 용어 통일 등)",
            font=("맑은 고딕", 8),
            foreground="gray"
        ).grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # 파일 선택
        file_frame = ttk.LabelFrame(main_frame, text="파일 선택", padding="10")
        file_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 15))
        
        ttk.Entry(
            file_frame,
            textvariable=self.file_path,
            width=50,
            state="readonly"
        ).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(
            file_frame,
            text="파일 선택",
            command=self._select_file
        ).grid(row=0, column=1)
        
        # 문체 설정
        tone_frame = ttk.LabelFrame(main_frame, text="문체 설정", padding="10")
        tone_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 15))
        
        ttk.Radiobutton(
            tone_frame,
            text="자동",
            variable=self.tone_var,
            value="auto"
        ).grid(row=0, column=0, padx=(0, 20))
        
        ttk.Radiobutton(
            tone_frame,
            text="존댓말",
            variable=self.tone_var,
            value="formal"
        ).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Radiobutton(
            tone_frame,
            text="반말",
            variable=self.tone_var,
            value="casual"
        ).grid(row=0, column=2)
        
        # 진행 상황
        progress_frame = ttk.LabelFrame(main_frame, text="진행 상황", padding="10")
        progress_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 15))
        
        # 진행률 바
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=600
        )
        self.progress_bar.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # 진행 정보
        self.progress_label = ttk.Label(progress_frame, text="대기 중...")
        self.progress_label.grid(row=1, column=0, sticky=tk.W)
        
        self.speed_label = ttk.Label(progress_frame, text="")
        self.speed_label.grid(row=2, column=0, sticky=tk.W)
        
        self.time_label = ttk.Label(progress_frame, text="")
        self.time_label.grid(row=3, column=0, sticky=tk.W)
        
        # 번역 버튼
        self.translate_btn = ttk.Button(
            main_frame,
            text="번역 시작",
            command=self._start_translation
        )
        self.translate_btn.grid(row=7, column=0, columnspan=3, pady=(0, 10))
        
        # 상태 표시
        self.status_label = ttk.Label(
            main_frame,
            text="",
            foreground="green",
            font=("맑은 고딕", 9)
        )
        self.status_label.grid(row=8, column=0, columnspan=3)
        
    def _select_file(self):
        """파일 선택 다이얼로그"""
        filename = filedialog.askopenfilename(
            title="SRT 파일 선택",
            filetypes=[("SRT 파일", "*.srt"), ("모든 파일", "*.*")]
        )
        
        if filename:
            self.file_path.set(filename)
            self.status_label.config(text="파일이 선택되었습니다.", foreground="green")
    
    def _on_model_change(self):
        """모델 변경 시 호출"""
        model = self.model_var.get()
        
        # 모델 변경 콜백 호출
        if self.model_change_callback:
            self.model_change_callback(model)
        else:
            # 콜백이 없으면 기본 메시지만 표시
            if model == "deepl":
                self.status_label.config(
                    text="DeepL 모델 선택됨 - 전문 번역 서비스, 최고 품질, 빠른 속도",
                    foreground="green"
                )
            elif model == "openai":
                self.status_label.config(
                    text="OpenAI 모델 선택됨 - 고품질 번역, GPU 불필요",
                    foreground="green"
                )
            else:
                self.status_label.config(
                    text="NLLB 모델 선택됨 - GPU 가속 권장, 오프라인 사용 가능",
                    foreground="blue"
                )
    
    def _start_translation(self):
        """번역 시작"""
        if not self.file_path.get():
            messagebox.showwarning("경고", "SRT 파일을 선택해주세요.")
            return
        
        if not self.translate_callback:
            messagebox.showerror("오류", "번역 엔진이 초기화되지 않았습니다.")
            return
        
        # 버튼 비활성화
        self.translate_btn.config(state="disabled")
        
        # 별도 스레드에서 번역 실행
        thread = threading.Thread(target=self._run_translation, daemon=True)
        thread.start()
    
    def _run_translation(self):
        """번역 실행 (별도 스레드)"""
        try:
            file_path = self.file_path.get()
            tone = self.tone_var.get()
            title = self.title_var.get().strip()  # 작품명 (선택사항)
            
            self.status_label.config(text="번역 진행 중...", foreground="blue")
            
            # 콜백 함수 호출
            if self.translate_callback:
                self.translate_callback(file_path, tone, title)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("오류", f"번역 중 오류 발생:\n{str(e)}"))
            self.root.after(0, lambda: self.status_label.config(text="오류 발생", foreground="red"))
        finally:
            # 버튼 활성화
            self.root.after(0, lambda: self.translate_btn.config(state="normal"))
    
    def set_translate_callback(self, callback: Callable):
        """번역 콜백 함수 설정"""
        self.translate_callback = callback
    
    def set_model_change_callback(self, callback: Callable):
        """모델 변경 콜백 함수 설정"""
        self.model_change_callback = callback
    
    def update_cuda_status(self, status: str, is_available: bool = True):
        """CUDA 상태 업데이트"""
        color = "green" if is_available else "red"
        self.cuda_status.set(status)
        # cuda_label 찾아서 색상 변경
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.LabelFrame):
                        for subchild in child.winfo_children():
                            if isinstance(subchild, ttk.Label) and subchild.cget("textvariable") == str(self.cuda_status):
                                subchild.config(foreground=color)
    
    def update_progress(self, current: int, total: int, start_time: float):
        """
        진행 상황 업데이트
        
        Args:
            current: 현재 진행 수
            total: 전체 수
            start_time: 시작 시간
        """
        # 진행률 계산
        progress = (current / total) * 100 if total > 0 else 0
        self.progress_bar['value'] = progress
        
        # 경과 시간
        elapsed = time.time() - start_time
        
        # 속도 계산 (항목/초)
        speed = current / elapsed if elapsed > 0 else 0
        
        # 남은 시간 계산
        remaining_items = total - current
        eta = remaining_items / speed if speed > 0 else 0
        
        # 라벨 업데이트
        self.progress_label.config(text=f"진행률: {current}/{total} ({progress:.1f}%)")
        self.speed_label.config(text=f"번역 속도: {speed:.2f} 항목/초")
        
        if eta > 0:
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            self.time_label.config(text=f"남은 시간: 약 {eta_min}분 {eta_sec}초")
        else:
            self.time_label.config(text="")
        
        # UI 업데이트
        self.root.update_idletasks()
    
    def show_completion(self, output_path: str):
        """번역 완료 메시지"""
        self.progress_label.config(text="번역 완료!")
        self.status_label.config(text=f"저장 위치: {output_path}", foreground="green")
        messagebox.showinfo("완료", f"번역이 완료되었습니다!\n\n저장 위치:\n{output_path}")
    
    def reset_progress(self):
        """진행 상황 초기화"""
        self.progress_bar['value'] = 0
        self.progress_label.config(text="대기 중...")
        self.speed_label.config(text="")
        self.time_label.config(text="")
