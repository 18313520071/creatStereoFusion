import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
import threading
import queue
import time
import json

# --- 配置文件路径 (修改为代码同一目录) ---
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_stereo_fusion_config.json")

# --- 设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 核心算法函数 (保持不变) ---

def rational_approx(x, max_denom):
    if x == 0: return 0, 1
    start_x = x
    a = int(x)
    num0, den0 = a, 1
    num1, den1 = 1, 0
    x -= a
    while x != 0 and den0 <= max_denom:
        a = int(1 / x)
        aux = num0
        num0 = a * num0 + num1
        num1 = aux
        aux = den0
        den0 = a * den0 + den1
        den1 = aux
        x = 1 / start_x - a
        start_x = 1 / x if x != 0 else 0
    if den0 > max_denom or den0 <= 0:
        den0 = max_denom
        num0 = round(x * den0)
        g = np.gcd(int(num0), int(den0))
        num0, den0 = int(num0/g), int(den0/g)
    return num0, den0

def generate_balanced_lr_sequence(P, total_length):
    I = int(P)
    f = P - I
    if f == 0:
        L_num = I // 2
        R_num = I - L_num
        base_seq = ['L'] * L_num + ['R'] * R_num
        seq = (base_seq * (total_length // len(base_seq) + 1))[:total_length]
        return np.array(seq)
    M, N = rational_approx(f, 100)
    T = N * I + M
    if T % 2 == 1:
        repeat_factor = 2
        T_total = 2 * T
    else:
        repeat_factor = 1
        T_total = T
    base_unit = np.zeros(N, dtype=int)
    error = 0
    for i in range(N):
        error += M
        if error >= N:
            base_unit[i] = I + 1
            error -= N
        else:
            base_unit[i] = I
    full_units = np.tile(base_unit, repeat_factor)
    L_counts = np.zeros_like(full_units)
    num_I = np.sum(full_units == I)
    num_Ip1 = np.sum(full_units == I + 1)
    target_L_total = T_total // 2
    L_from_Ip1 = num_Ip1 * ((I + 1) // 2)
    L_needed_from_I = target_L_total - L_from_Ip1
    low = I // 2
    high = (I + 1) // 2
    if high == low:
        L_counts[full_units == I] = low
    else:
        x = L_needed_from_I - num_I * low
        x = max(0, min(x, num_I))
        idx_I = np.where(full_units == I)[0]
        assign_high = np.zeros(len(idx_I), dtype=bool)
        if x > 0:
            step = max(1, len(idx_I) // x)
            assign_high[::step] = True
            if np.sum(assign_high) != x:
                assign_high = np.zeros(len(idx_I), dtype=bool)
                assign_high[:x] = True
        L_counts[idx_I[assign_high]] = high
        L_counts[idx_I[~assign_high]] = low
    L_counts[full_units == I + 1] = (I + 1) // 2
    char_seq = []
    for k in range(len(full_units)):
        L_num = L_counts[k]
        R_num = full_units[k] - L_num
        char_seq.extend(['L'] * L_num + ['R'] * R_num)
    while len(char_seq) < total_length:
        char_seq.extend(char_seq)
    seq = np.array(char_seq[:total_length])
    return seq

# --- GUI 主程序 ---

class VideoStereoFusionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("立体视频融合生成器 (视频专用版)")
        self.root.geometry("1000x800")
        
        self.video_captures = []
        self.video_writer = None
        self.processing = False
        self.cancel_flag = False
        self.dialog_queue = queue.Queue()
        self.preview_queue = queue.Queue()
        self.test_video_mode = None  # 'black_white', 'white_black', 'red_blue'
        
        self.load_config()
        self.create_widgets()
        self.bind_config_save_events()
        self.check_preview_queue()

    def load_config(self):
        default_config = {
            "width": "1920",
            "height": "1080",
            "period": "8",
            "views": "2",
            "angle": "18.435",
            "fps": "30",
            "codec": "mp4v",
            "test_duration": "2.0"
        }
        
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    default_config.update(config)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
        
        self.config = default_config

    def save_config(self):
        try:
            config = {
                "width": self.width_var.get(),
                "height": self.height_var.get(),
                "period": self.period_var.get(),
                "views": self.views_var.get(),
                "angle": self.angle_var.get(),
                "fps": self.fps_var.get(),
                "codec": self.codec_var.get(),
                "test_duration": self.test_duration_var.get()
            }
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"✓ 配置已保存至: {CONFIG_FILE}")
        except Exception as e:
            print(f"✗ 保存配置文件失败: {e}")

    def bind_config_save_events(self):
        def create_trace(var_name, var):
            def callback(*args):
                if hasattr(self, '_save_timer'):
                    self.root.after_cancel(self._save_timer)
                self._save_timer = self.root.after(1000, self.save_config)
            var.trace_add('write', callback)
        
        create_trace('width', self.width_var)
        create_trace('height', self.height_var)
        create_trace('period', self.period_var)
        create_trace('views', self.views_var)
        create_trace('angle', self.angle_var)
        create_trace('fps', self.fps_var)
        create_trace('codec', self.codec_var)
        create_trace('test_duration', self.test_duration_var)

    def check_preview_queue(self):
        try:
            while True:
                preview_data = self.preview_queue.get_nowait()
                self.show_preview(preview_data)
        except queue.Empty:
            pass
        self.root.after(100, self.check_preview_queue)

    def create_widgets(self):
        # 顶部参数区域
        param_frame = ttk.LabelFrame(self.root, text="参数设置", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5)

        # 第一行：分辨率
        ttk.Label(param_frame, text="分辨率:").grid(row=0, column=0, sticky="e")
        self.width_var = tk.StringVar(value=self.config["width"])
        self.height_var = tk.StringVar(value=self.config["height"])
        ttk.Entry(param_frame, textvariable=self.width_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(param_frame, text="x").grid(row=0, column=2)
        ttk.Entry(param_frame, textvariable=self.height_var, width=8).grid(row=0, column=3, padx=2)

        # 第二行：周期和视点数
        ttk.Label(param_frame, text="周期 (Period):").grid(row=0, column=4, sticky="e", padx=(10,0))
        self.period_var = tk.StringVar(value=self.config["period"])
        ttk.Entry(param_frame, textvariable=self.period_var, width=8).grid(row=0, column=5, padx=2)
        
        ttk.Label(param_frame, text="视点数:").grid(row=0, column=6, sticky="e", padx=(10,0))
        self.views_var = tk.StringVar(value=self.config["views"])
        ttk.Entry(param_frame, textvariable=self.views_var, width=5).grid(row=0, column=7, padx=2)

        # 第三行：角度和帧率
        ttk.Label(param_frame, text="斜向偏移角度:").grid(row=1, column=0, sticky="e", pady=(5,0))
        self.angle_var = tk.StringVar(value=self.config["angle"])
        ttk.Entry(param_frame, textvariable=self.angle_var, width=8).grid(row=1, column=1, padx=2, pady=(5,0))
        ttk.Label(param_frame, text="度").grid(row=1, column=2, pady=(5,0))
        
        ttk.Label(param_frame, text="输出帧率:").grid(row=1, column=4, sticky="e", pady=(5,0))
        self.fps_var = tk.StringVar(value=self.config["fps"])
        ttk.Entry(param_frame, textvariable=self.fps_var, width=8).grid(row=1, column=5, padx=2, pady=(5,0))
        
        ttk.Label(param_frame, text="编解码器:").grid(row=1, column=6, sticky="e", pady=(5,0))
        self.codec_var = tk.StringVar(value=self.config["codec"])
        codec_combo = ttk.Combobox(param_frame, textvariable=self.codec_var, width=8, 
                                   values=["mp4v", "XVID", "H264", "MJPG"])
        codec_combo.grid(row=1, column=7, padx=2, pady=(5,0))

        # 第四行：测试视频时长
        ttk.Label(param_frame, text="测试视频时长:").grid(row=2, column=0, sticky="e", pady=(5,0))
        self.test_duration_var = tk.StringVar(value=self.config["test_duration"])
        ttk.Entry(param_frame, textvariable=self.test_duration_var, width=8).grid(row=2, column=1, padx=2, pady=(5,0))
        ttk.Label(param_frame, text="秒").grid(row=2, column=2, pady=(5,0))

        # 视频选择区域
        video_frame = ttk.LabelFrame(self.root, text="视频输入", padding=10)
        video_frame.pack(fill="x", padx=10, pady=5)

        self.video_label = ttk.Label(video_frame, text="未选择视频", foreground="gray")
        self.video_label.pack(side="left", fill="x", expand=True)

        ttk.Button(video_frame, text="选择输入视频", command=self.select_videos).pack(side="right", padx=5)
        
        # 测试视频生成区域
        test_frame = ttk.LabelFrame(self.root, text="测试视频生成 (2秒)", padding=10)
        test_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(test_frame, text="快速生成测试视频用于验证参数", foreground="blue").pack(side="left", padx=5)
        
        btn_frame_test = ttk.Frame(test_frame)
        btn_frame_test.pack(side="right")
        
        # 黑白测试视频
        bw_frame = ttk.Frame(btn_frame_test)
        bw_frame.pack(side="top", pady=2)
        ttk.Button(bw_frame, text="生成黑白测试视频", command=self.show_black_white_dialog).pack(side="left", padx=2)
        
        # 红蓝测试视频
        rb_frame = ttk.Frame(btn_frame_test)
        rb_frame.pack(side="top", pady=2)
        ttk.Button(rb_frame, text="生成红蓝测试视频", command=self.generate_red_blue_video).pack(side="left", padx=2)

        # 保存路径选择区域
        save_frame = ttk.LabelFrame(self.root, text="输出设置", padding=10)
        save_frame.pack(fill="x", padx=10, pady=5)

        self.save_label = ttk.Label(save_frame, text="保存路径: 未选择", foreground="gray")
        self.save_label.pack(side="left", fill="x", expand=True)

        ttk.Button(save_frame, text="选择保存位置", command=self.select_save_path).pack(side="right", padx=5)

        # 连续视点融合视频按钮区域
        continuous_frame = ttk.LabelFrame(self.root, text="连续视点融合视频", padding=10)
        continuous_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(continuous_frame, text="输出一个周期内所有视点顺序的融合视频", foreground="blue").pack(side="left", padx=5)
        self.continuous_btn = ttk.Button(continuous_frame, text="生成连续视点融合视频", 
                                        command=self.start_continuous_processing)
        self.continuous_btn.pack(side="right", padx=5)

        # 进度条
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side="right")

        # 状态显示区域
        status_frame = ttk.LabelFrame(self.root, text="运行状态", padding=10)
        status_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=12, state="disabled", bg="#f0f0f0")
        self.status_text.pack(fill="both", expand=True)
        
        # 底部按钮
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        self.cancel_btn = ttk.Button(btn_frame, text="取消", command=self.cancel_processing, state="disabled")
        self.cancel_btn.pack(side="left", padx=5)
        
        self.run_btn = ttk.Button(btn_frame, text="开始生成融合视频", command=self.start_processing)
        self.run_btn.pack(side="right")

    def log(self, message):
        self.status_text.config(state="normal")
        self.status_text.insert("end", message + "\n")
        self.status_text.see("end")
        self.status_text.config(state="disabled")
        self.root.update_idletasks()

    def update_progress(self, value, total, current_frame=None):
        """更新进度条"""
        if total > 0:
            progress = (value / total) * 100
            self.progress_var.set(progress)
            if current_frame is not None:
                self.progress_label.config(text=f"{progress:.1f}% (帧 {current_frame})")
            else:
                self.progress_label.config(text=f"{progress:.1f}%")
            self.root.update_idletasks()

    def select_videos(self):
        paths = filedialog.askopenfilenames(
            title="选择视点视频",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv")]
        )
        if paths:
            self.close_video_captures()
            self.video_captures = []
            
            for path in paths:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    messagebox.showerror("错误", f"无法打开视频文件: {path}")
                    self.close_video_captures()
                    return
                self.video_captures.append(cap)
            
            num_frames = int(self.video_captures[0].get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_captures[0].get(cv2.CAP_PROP_FPS)
            width = int(self.video_captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.video_label.config(text=f"已选择 {len(paths)} 个视频: {os.path.basename(paths[0])}... ({num_frames}帧, {fps:.2f}fps)", foreground="blue")
            self.log(f"已加载视频: {paths[0]}")
            self.log(f"视频信息: {num_frames}帧, {fps:.2f}fps, {width}x{height}")
            
            # 自动更新参数
            self.width_var.set(str(width))
            self.height_var.set(str(height))
            self.fps_var.set(str(fps))
            
            self.test_video_mode = None

    def close_video_captures(self):
        """关闭所有视频捕获对象"""
        for cap in self.video_captures:
            if cap is not None:
                cap.release()
        self.video_captures = []

    def select_save_path(self):
        path = filedialog.asksaveasfilename(
            title="选择保存位置及文件名",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")],
            initialfile=f"fusion_video_P{self.period_var.get()}_V{self.views_var.get()}"
        )
        
        if path:
            self.save_path = path
            self.save_label.config(text=f"保存路径: {path}", foreground="black")

    def show_black_white_dialog(self):
        """显示黑白测试视频选择对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("选择黑白测试视频类型")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="选择黑白测试视频的视点顺序:", font=("Arial", 12, "bold")).pack(pady=20)
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="左黑右白 (L=黑, R=白)", 
                  command=lambda: self.generate_black_white_video('black_white', dialog)).pack(pady=5)
        
        ttk.Button(btn_frame, text="左白右黑 (L=白, R=黑)", 
                  command=lambda: self.generate_black_white_video('white_black', dialog)).pack(pady=5)
        
        ttk.Button(dialog, text="取消", command=dialog.destroy).pack(pady=10)

    def generate_black_white_video(self, mode='black_white', dialog=None):
        """生成黑白测试视频"""
        if dialog:
            dialog.destroy()
        
        self.log(f"\n=== 开始生成黑白测试视频 ({mode}) ===")
        
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            fps = float(self.fps_var.get())
            duration = float(self.test_duration_var.get())
            num_frames = int(fps * duration)
            
            self.log(f"视频参数: {width}x{height}, {fps}fps, 时长 {duration}秒, 总帧数 {num_frames}")
            
            # 创建测试视频
            if mode == 'black_white':
                self.log("生成模式: 左黑右白")
                display_text = "左黑右白"
            else:
                self.log("生成模式: 左白右黑")
                display_text = "左白右黑"
            
            self.test_video_mode = mode
            
            # 请求保存路径
            self.root.after(0, lambda: self.dialog_queue.put(
                filedialog.asksaveasfilename(
                    title=f"保存黑白测试视频 ({display_text})",
                    defaultextension=".mp4",
                    filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi")],
                    initialfile=f"test_black_white_{mode}_{width}x{height}"
                )
            ))
            
            try:
                save_path = self.dialog_queue.get(timeout=60)
            except queue.Empty:
                self.log("超时：用户未在规定时间内选择路径")
                return
            
            if not save_path:
                self.log("用户取消了保存操作")
                return
            
            # 确保文件扩展名
            if not save_path.lower().endswith(('.mp4', '.avi')):
                save_path = os.path.splitext(save_path)[0] + '.mp4'
            
            # 创建视频写入器
            codec_str = self.codec_var.get()
            file_ext = os.path.splitext(save_path)[1].lower()
            if file_ext == '.avi':
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                raise ValueError(f"无法创建视频写入器")
            
            self.log(f"开始生成 {num_frames} 帧测试视频...")
            
            for frame_idx in range(num_frames):
                if frame_idx % 30 == 0:
                    self.log(f"生成帧 {frame_idx}/{num_frames}")
                
                if mode == 'black_white':
                    # 左黑右白
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    frame[:, width//2:, :] = 255  # 右半部分白色
                else:
                    # 左白右黑
                    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
                    frame[:, width//2:, :] = 0  # 右半部分黑色
                
                video_writer.write(frame)
            
            video_writer.release()
            
            self.log(f"\n✓ 黑白测试视频生成完成！")
            self.log(f"保存位置: {save_path}")
            self.log(f"视频信息: {width}x{height}, {fps}fps, {duration}秒, {num_frames}帧")
            
            self.root.after(0, lambda: messagebox.showinfo(
                "完成", 
                f"黑白测试视频生成完成！\n模式: {display_text}\n保存位置: {save_path}"
            ))
            
        except Exception as e:
            self.log(f"✗ 生成黑白测试视频失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))

    def generate_red_blue_video(self):
        """生成红蓝测试视频"""
        self.log(f"\n=== 开始生成红蓝测试视频 (左红右蓝) ===")
        
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            fps = float(self.fps_var.get())
            duration = float(self.test_duration_var.get())
            num_frames = int(fps * duration)
            
            self.log(f"视频参数: {width}x{height}, {fps}fps, 时长 {duration}秒, 总帧数 {num_frames}")
            self.log("生成模式: 左红右蓝")
            
            self.test_video_mode = 'red_blue'
            
            # 请求保存路径
            self.root.after(0, lambda: self.dialog_queue.put(
                filedialog.asksaveasfilename(
                    title="保存红蓝测试视频",
                    defaultextension=".mp4",
                    filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi")],
                    initialfile=f"test_red_blue_{width}x{height}"
                )
            ))
            
            try:
                save_path = self.dialog_queue.get(timeout=60)
            except queue.Empty:
                self.log("超时：用户未在规定时间内选择路径")
                return
            
            if not save_path:
                self.log("用户取消了保存操作")
                return
            
            # 确保文件扩展名
            if not save_path.lower().endswith(('.mp4', '.avi')):
                save_path = os.path.splitext(save_path)[0] + '.mp4'
            
            # 创建视频写入器
            codec_str = self.codec_var.get()
            file_ext = os.path.splitext(save_path)[1].lower()
            if file_ext == '.avi':
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                raise ValueError(f"无法创建视频写入器")
            
            self.log(f"开始生成 {num_frames} 帧测试视频...")
            
            for frame_idx in range(num_frames):
                if frame_idx % 30 == 0:
                    self.log(f"生成帧 {frame_idx}/{num_frames}")
                
                # 左红右蓝
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :width//2, 2] = 255  # 左半部分红色 (BGR格式，第2通道是红色)
                frame[:, width//2:, 0] = 255  # 右半部分蓝色 (BGR格式，第0通道是蓝色)
                
                video_writer.write(frame)
            
            video_writer.release()
            
            self.log(f"\n✓ 红蓝测试视频生成完成！")
            self.log(f"保存位置: {save_path}")
            self.log(f"视频信息: {width}x{height}, {fps}fps, {duration}秒, {num_frames}帧")
            
            self.root.after(0, lambda: messagebox.showinfo(
                "完成", 
                f"红蓝测试视频生成完成！\n模式: 左红右蓝\n保存位置: {save_path}"
            ))
            
        except Exception as e:
            self.log(f"✗ 生成红蓝测试视频失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))

    def cancel_processing(self):
        """取消处理"""
        self.cancel_flag = True
        self.log("正在取消处理...")
        self.cancel_btn.config(state="disabled")

    def start_continuous_processing(self):
        if not self.video_captures and self.test_video_mode is None:
            messagebox.showwarning("警告", "请先选择视频或生成测试视频！")
            return
        
        self.continuous_btn.config(state="disabled", text="处理中...")
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.cancel_flag = False
        self.processing = True
        self.log("\n=== 开始连续视点融合视频处理 ===")
        
        threading.Thread(target=self.process_continuous_logic, daemon=True).start()

    def start_processing(self):
        if not self.video_captures and self.test_video_mode is None:
            messagebox.showwarning("警告", "请先选择视频或生成测试视频！")
            return
        
        self.run_btn.config(state="disabled", text="处理中...")
        self.continuous_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.cancel_flag = False
        self.processing = True
        
        self.log("\n" + "="*60)
        self.log("=== 开始生成融合视频 ===")
        self.log("="*60 + "\n")
        
        threading.Thread(target=self.process_logic, daemon=True).start()

    def process_logic(self):
        try:
            self.process_video_logic()
        except Exception as e:
            self.log(f"发生错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.root.after(0, lambda: self.run_btn.config(state="normal", text="开始生成融合视频"))
            self.root.after(0, lambda: self.continuous_btn.config(state="normal", text="生成连续视点融合视频"))
            self.root.after(0, lambda: self.cancel_btn.config(state="disabled"))
            self.processing = False
            self.progress_var.set(0)
            self.progress_label.config(text="0%")
            self.close_video_captures()

    def process_video_logic(self):
        self.log("=== 视频融合处理模式 ===\n")
        
        width = int(self.width_var.get())
        height = int(self.height_var.get())
        period_input = float(self.period_var.get())
        num_views = int(self.views_var.get())
        angle_deg = float(self.angle_var.get())
        output_fps = float(self.fps_var.get())
        codec_str = self.codec_var.get()
        
        # 检查是否使用测试视频模式
        if self.test_video_mode is not None:
            self.log(f"使用测试视频模式: {self.test_video_mode}")
            self.process_test_video_logic(width, height, period_input, num_views, angle_deg, output_fps, codec_str)
            return
        
        # 使用实际视频
        if len(self.video_captures) < num_views:
            raise ValueError(f"需要 {num_views} 个视频，但只选择了 {len(self.video_captures)} 个")
        
        self.log("正在读取视频信息...")
        video_frames = []
        video_fps_list = []
        video_width_list = []
        video_height_list = []
        
        for i, cap in enumerate(self.video_captures[:num_views]):
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_frames.append(num_frames)
            video_fps_list.append(fps)
            video_width_list.append(w)
            video_height_list.append(h)
            
            self.log(f"视频 {i+1}: {num_frames}帧, {fps:.2f}fps, {w}x{h}")
        
        total_frames = min(video_frames)  # 使用最短视频的帧数
        self.log(f"\n总帧数 (按最短视频): {total_frames}")
        
        # 验证周期参数
        if num_views == 2 and period_input != int(period_input):
            is_balanced_mode = True
            P = period_input
            self.log(">>> 启用非整数周期平衡模式")
        else:
            is_balanced_mode = False
            P = round(period_input)
            if P % num_views != 0:
                raise ValueError(f'周期 {P} 必须能被视点数 {num_views} 整除')

        num_channels = 3
        original_width = width
        
        tan_angle = np.tan(np.radians(angle_deg))
        offset_per_row_float = tan_angle * num_channels
        
        max_offset = abs(offset_per_row_float) * (height - 1)
        subpixel_width_original = original_width * num_channels
        num_tiles = max(3, int(np.ceil((subpixel_width_original + 2 * max_offset) / subpixel_width_original)))
        
        self.log(f"扩展图像 {num_tiles} 倍以防止接缝...")
        
        width = original_width * num_tiles
        subpixel_width = width * num_channels

        # 预计算视点分配表
        view_assignment_table = self.precompute_view_assignment(
            height, subpixel_width, P, num_views, angle_deg, is_balanced_mode
        )

        # 获取保存路径
        final_save_path = self.save_path
        
        if not final_save_path:
            self.log("正在请求保存路径...")
            
            while not self.dialog_queue.empty():
                try:
                    self.dialog_queue.get_nowait()
                except:
                    pass
            
            self.root.after(0, lambda: self.dialog_queue.put(
                filedialog.asksaveasfilename(
                    title="选择保存位置",
                    defaultextension=".mp4",
                    filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")],
                    initialfile=f"fusion_video_P{P}_V{num_views}_A{angle_deg}"
                )
            ))
            
            try:
                final_save_path = self.dialog_queue.get(timeout=60)
            except queue.Empty:
                self.log("超时：用户未在规定时间内选择路径")
                return

        if not final_save_path:
            self.log("用户取消了保存操作。")
            return

        save_dir = os.path.dirname(final_save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 创建视频写入器
        file_ext = os.path.splitext(final_save_path)[1].lower()
        if file_ext == '.avi':
            fourcc_code = cv2.VideoWriter_fourcc(*codec_str)
        else:
            fourcc_code = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.video_writer = cv2.VideoWriter(
            final_save_path,
            fourcc_code,
            output_fps,
            (original_width, height)
        )
        
        if not self.video_writer.isOpened():
            raise ValueError(f"无法创建视频写入器，请检查编解码器设置")

        self.log(f"\n视频写入器已创建: {final_save_path}")
        self.log(f"输出参数: {original_width}x{height}, {output_fps}fps, 编解码器: {codec_str}\n")

        # 逐帧处理
        self.log("开始逐帧处理...")
        start_time = time.time()
        
        for frame_idx in range(total_frames):
            if self.cancel_flag:
                self.log("用户取消了处理")
                break
            
            # 读取所有视点的当前帧
            frame_list = []
            for cap in self.video_captures[:num_views]:
                ret, frame = cap.read()
                if not ret:
                    self.log(f"警告: 视频读取结束于帧 {frame_idx}")
                    break
                
                # 转换为RGB并调整大小
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if frame_rgb.shape[:2] != (height, original_width):
                    frame_rgb = cv2.resize(frame_rgb, (original_width, height))
                
                # 扩展图像
                frame_rgb = np.tile(frame_rgb, (1, num_tiles, 1))
                
                frame_list.append(frame_rgb)
            
            if len(frame_list) < num_views:
                self.log("视频帧读取不完整，停止处理")
                break
            
            # 融合处理
            fusion_rgb = self.fuse_frames(frame_list, view_assignment_table, height, width, num_views, num_tiles, original_width)
            
            # 写入视频
            fusion_bgr = cv2.cvtColor(fusion_rgb, cv2.COLOR_RGB2BGR)
            self.video_writer.write(fusion_bgr)
            
            # 更新进度
            if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                self.update_progress(frame_idx + 1, total_frames, frame_idx + 1)
                elapsed_time = time.time() - start_time
                fps_current = (frame_idx + 1) / elapsed_time if elapsed_time > 0 else 0
                self.log(f"处理帧 {frame_idx + 1}/{total_frames} ({fps_current:.1f} fps)")
        
        self.video_writer.release()
        elapsed_time = time.time() - start_time
        avg_fps = total_frames / elapsed_time if elapsed_time > 0 else 0
        
        self.log("\n" + "="*60)
        self.log(f"✓ 视频处理完成！")
        self.log(f"总帧数: {total_frames}")
        self.log(f"处理时间: {elapsed_time:.2f}秒")
        self.log(f"平均速度: {avg_fps:.2f} fps")
        self.log(f"已保存至: {final_save_path}")
        self.log("="*60)
        
        self.root.after(0, lambda: messagebox.showinfo(
            "完成", 
            f"视频融合处理完成！\n总帧数: {total_frames}\n处理时间: {elapsed_time:.2f}秒\n保存位置: {final_save_path}"
        ))

    def process_test_video_logic(self, width, height, period_input, num_views, angle_deg, output_fps, codec_str):
        """处理测试视频的融合逻辑"""
        self.log("=== 测试视频融合模式 ===\n")
        
        # 验证周期参数
        if num_views == 2 and period_input != int(period_input):
            is_balanced_mode = True
            P = period_input
            self.log(">>> 启用非整数周期平衡模式")
        else:
            is_balanced_mode = False
            P = round(period_input)
            if P % num_views != 0:
                raise ValueError(f'周期 {P} 必须能被视点数 {num_views} 整除')

        num_channels = 3
        original_width = width
        
        tan_angle = np.tan(np.radians(angle_deg))
        offset_per_row_float = tan_angle * num_channels
        
        max_offset = abs(offset_per_row_float) * (height - 1)
        subpixel_width_original = original_width * num_channels
        num_tiles = max(3, int(np.ceil((subpixel_width_original + 2 * max_offset) / subpixel_width_original)))
        
        self.log(f"扩展图像 {num_tiles} 倍以防止接缝...")
        
        width = original_width * num_tiles
        subpixel_width = width * num_channels

        # 预计算视点分配表
        view_assignment_table = self.precompute_view_assignment(
            height, subpixel_width, P, num_views, angle_deg, is_balanced_mode
        )

        # 获取保存路径
        final_save_path = self.save_path
        
        if not final_save_path:
            self.log("正在请求保存路径...")
            
            while not self.dialog_queue.empty():
                try:
                    self.dialog_queue.get_nowait()
                except:
                    pass
            
            self.root.after(0, lambda: self.dialog_queue.put(
                filedialog.asksaveasfilename(
                    title="选择保存位置",
                    defaultextension=".mp4",
                    filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")],
                    initialfile=f"test_fusion_{self.test_video_mode}_P{P}_V{num_views}"
                )
            ))
            
            try:
                final_save_path = self.dialog_queue.get(timeout=60)
            except queue.Empty:
                self.log("超时：用户未在规定时间内选择路径")
                return

        if not final_save_path:
            self.log("用户取消了保存操作。")
            return

        save_dir = os.path.dirname(final_save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 创建视频写入器
        file_ext = os.path.splitext(final_save_path)[1].lower()
        if file_ext == '.avi':
            fourcc_code = cv2.VideoWriter_fourcc(*codec_str)
        else:
            fourcc_code = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.video_writer = cv2.VideoWriter(
            final_save_path,
            fourcc_code,
            output_fps,
            (original_width, height)
        )
        
        if not self.video_writer.isOpened():
            raise ValueError(f"无法创建视频写入器，请检查编解码器设置")

        self.log(f"\n视频写入器已创建: {final_save_path}")
        self.log(f"输出参数: {original_width}x{height}, {output_fps}fps, 编解码器: {codec_str}\n")

        # 生成测试视频
        duration = float(self.test_duration_var.get())
        total_frames = int(output_fps * duration)
        
        self.log(f"生成测试融合视频: {duration}秒, {total_frames}帧")
        
        start_time = time.time()
        
        for frame_idx in range(total_frames):
            if self.cancel_flag:
                self.log("用户取消了处理")
                break
            
            # 生成测试帧
            frame_list = self.generate_test_frames(width, height, num_views, num_tiles, original_width)
            
            # 融合处理
            fusion_rgb = self.fuse_frames(frame_list, view_assignment_table, height, width, num_views, num_tiles, original_width)
            
            # 写入视频
            fusion_bgr = cv2.cvtColor(fusion_rgb, cv2.COLOR_RGB2BGR)
            self.video_writer.write(fusion_bgr)
            
            # 更新进度
            if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                self.update_progress(frame_idx + 1, total_frames, frame_idx + 1)
                elapsed_time = time.time() - start_time
                fps_current = (frame_idx + 1) / elapsed_time if elapsed_time > 0 else 0
                self.log(f"生成帧 {frame_idx + 1}/{total_frames} ({fps_current:.1f} fps)")
        
        self.video_writer.release()
        elapsed_time = time.time() - start_time
        avg_fps = total_frames / elapsed_time if elapsed_time > 0 else 0
        
        self.log("\n" + "="*60)
        self.log(f"✓ 测试视频融合完成！")
        self.log(f"总帧数: {total_frames}")
        self.log(f"处理时间: {elapsed_time:.2f}秒")
        self.log(f"平均速度: {avg_fps:.2f} fps")
        self.log(f"已保存至: {final_save_path}")
        self.log("="*60)
        
        self.root.after(0, lambda: messagebox.showinfo(
            "完成", 
            f"测试视频融合处理完成！\n总帧数: {total_frames}\n处理时间: {elapsed_time:.2f}秒\n保存位置: {final_save_path}"
        ))

    def precompute_view_assignment(self, height, subpixel_width, P, num_views, angle_deg, is_balanced_mode):
        """预计算视点分配表"""
        view_assignment_table = np.zeros((height, subpixel_width), dtype=int)
        
        tan_angle = np.tan(np.radians(angle_deg))
        offset_per_row_float = tan_angle * 3  # 3 channels
        
        if is_balanced_mode:
            lr_sequence = generate_balanced_lr_sequence(P, subpixel_width)
            for r in range(height):
                row_offset = r * offset_per_row_float
                positions = np.arange(subpixel_width)
                seq_positions = np.floor(positions - row_offset).astype(int) % subpixel_width
                view_assignment_table[r, :] = (lr_sequence[seq_positions] == 'L').astype(int)
        else:
            subpixels_per_view = P / num_views
            base_view_assignment = np.zeros(P, dtype=int)
            for i in range(P):
                view_index = int(np.floor((i % P) / subpixels_per_view + 1e-10))
                base_view_assignment[i] = view_index
            
            self.log(f"基础视点分配序列 (周期={P}, 视点数={num_views}):")
            self.log(f"{base_view_assignment}")
            
            full_assignment = np.tile(base_view_assignment, int(np.ceil(subpixel_width / P)))[:subpixel_width]
            
            for r in range(height):
                row_offset = r * offset_per_row_float
                positions = np.arange(subpixel_width)
                seq_positions = np.floor(positions - row_offset + 1e-10).astype(int) % subpixel_width
                view_assignment_table[r, :] = full_assignment[seq_positions]

        used_views = np.unique(view_assignment_table)
        self.log(f"使用的视点索引: {used_views}")
        if len(used_views) < num_views:
            self.log(f"警告: 只使用了 {len(used_views)} 个视点，预期 {num_views} 个")
        else:
            self.log(f"✓ 所有 {num_views} 个视点都被正确使用")
        
        return view_assignment_table

    def generate_test_frames(self, width, height, num_views, num_tiles, original_width):
        """生成测试帧（黑白或红蓝）"""
        frame_list = []
        
        for v in range(num_views):
            if self.test_video_mode == 'black_white':
                if v == 0:  # 左视点 - 黑色
                    frame = np.zeros((height, original_width, 3), dtype=np.uint8)
                else:  # 右视点 - 白色
                    frame = np.ones((height, original_width, 3), dtype=np.uint8) * 255
            elif self.test_video_mode == 'white_black':
                if v == 0:  # 左视点 - 白色
                    frame = np.ones((height, original_width, 3), dtype=np.uint8) * 255
                else:  # 右视点 - 黑色
                    frame = np.zeros((height, original_width, 3), dtype=np.uint8)
            elif self.test_video_mode == 'red_blue':
                if v == 0:  # 左视点 - 红色
                    frame = np.zeros((height, original_width, 3), dtype=np.uint8)
                    frame[:, :, 0] = 255  # BGR格式，红色在第2通道
                else:  # 右视点 - 蓝色
                    frame = np.zeros((height, original_width, 3), dtype=np.uint8)
                    frame[:, :, 2] = 255  # BGR格式，蓝色在第0通道
            else:
                # 默认使用黑白
                frame = np.zeros((height, original_width, 3), dtype=np.uint8) if v == 0 else np.ones((height, original_width, 3), dtype=np.uint8) * 255
            
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 扩展图像
            frame_rgb = np.tile(frame_rgb, (1, num_tiles, 1))
            
            frame_list.append(frame_rgb)
        
        return frame_list

    def fuse_frames(self, frame_list, view_assignment_table, height, width, num_views, num_tiles, original_width):
        """融合帧"""
        num_channels = 3
        subpixel_width = width * num_channels
        
        all_subpixels = []
        for v in range(num_views):
            subpixels = frame_list[v].reshape(height, -1)
            all_subpixels.append(subpixels.astype(np.uint8))
        
        fusion_subpixels = np.zeros((height, subpixel_width), dtype=np.uint8)
        
        for r in range(height):
            for v in range(num_views):
                mask = (view_assignment_table[r, :] == v)
                if np.any(mask):
                    fusion_subpixels[r, mask] = all_subpixels[v][r, mask]
        
        fusion_rgb = fusion_subpixels.reshape(height, width, num_channels)
        start_col = (num_tiles // 2) * original_width
        end_col = start_col + original_width
        fusion_rgb = fusion_rgb[:, start_col:end_col, :]
        
        return fusion_rgb

    def show_preview(self, preview_data):
        fusion_rgb = preview_data['image']
        period = preview_data['period']
        views = preview_data['views']
        angle = preview_data['angle']
        
        try:
            plt.figure(figsize=(10, 6))
            plt.imshow(fusion_rgb)
            plt.title(f"融合结果 - 周期: {period}, 视点数: {views}, 角度: {angle}°")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.log(f"显示预览失败: {str(e)}")

    def process_continuous_logic(self):
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            period_input = float(self.period_var.get())
            num_views = int(self.views_var.get())
            angle_deg = float(self.angle_var.get())
            output_fps = float(self.fps_var.get())
            codec_str = self.codec_var.get()
            
            self.log(f"\n开始生成 {num_views} 个连续视点融合视频...")
            
            # 请求保存目录
            self.log("正在请求保存目录...")
            
            while not self.dialog_queue.empty():
                try:
                    self.dialog_queue.get_nowait()
                except:
                    pass
            
            self.root.after(0, lambda: self.dialog_queue.put(
                filedialog.askdirectory(title="选择保存目录（用于保存多个融合视频）")
            ))
            
            try:
                save_dir = self.dialog_queue.get(timeout=60)
            except queue.Empty:
                self.log("超时：用户未在规定时间内选择目录")
                return
            
            if not save_dir or not os.path.exists(save_dir):
                self.log("用户取消了保存操作或目录不存在。")
                return
            
            self.log(f"保存目录: {save_dir}\n")
            
            # 为每个视点偏移生成视频
            for shift in range(num_views):
                if self.cancel_flag:
                    self.log("用户取消了处理")
                    break
                
                self.log(f"\n--- 处理第 {shift + 1}/{num_views} 个融合视频 (视点顺序: {[(i + shift) % num_views + 1 for i in range(num_views)]}) ---")
                
                # 保存当前状态
                original_captures = self.video_captures.copy() if self.video_captures else None
                original_test_mode = self.test_video_mode
                
                try:
                    # 如果是测试视频模式，不需要改变
                    if self.test_video_mode is None and self.video_captures:
                        # 重新打开视频文件并跳转到正确位置
                        self.close_video_captures()
                        self.video_captures = []
                        
                        for path in self.file_paths[:num_views]:
                            cap = cv2.VideoCapture(path)
                            if not cap.isOpened():
                                raise ValueError(f"无法打开视频文件: {path}")
                            self.video_captures.append(cap)
                    
                    # 生成输出文件名
                    filename = f"fusion_shift{shift+1}_P{period_input}_V{num_views}_A{angle_deg}.mp4"
                    save_path = os.path.join(save_dir, filename)
                    
                    # 临时设置保存路径
                    original_save_path = self.save_path
                    self.save_path = save_path
                    
                    # 处理视频
                    if self.test_video_mode is not None:
                        self.process_test_video_logic(width, height, period_input, num_views, angle_deg, output_fps, codec_str)
                    else:
                        self.process_single_video_logic(width, height, period_input, num_views, angle_deg, output_fps, codec_str, shift)
                    
                    self.log(f"✓ 第 {shift + 1} 个融合视频已保存: {filename}")
                    
                except Exception as e:
                    self.log(f"✗ 第 {shift + 1} 个融合视频处理失败: {str(e)}")
                    import traceback
                    self.log(traceback.format_exc())
                    continue
                finally:
                    # 恢复状态
                    self.save_path = original_save_path
                    if original_captures:
                        self.close_video_captures()
                        self.video_captures = original_captures
                
                self.update_progress(shift + 1, num_views)
            
            self.log(f"\n=== 连续视点融合视频处理完成！共生成 {num_views} 个融合视频 ===")
            self.log(f"所有文件已保存至: {save_dir}")
            
            self.root.after(0, lambda: messagebox.showinfo(
                "完成", 
                f"连续视点融合视频处理完成！\n共生成 {num_views} 个融合视频\n保存位置: {save_dir}"
            ))
            
        except Exception as e:
            self.log(f"发生错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.root.after(0, lambda: self.run_btn.config(state="normal", text="开始生成融合视频"))
            self.root.after(0, lambda: self.continuous_btn.config(state="normal", text="生成连续视点融合视频"))
            self.root.after(0, lambda: self.cancel_btn.config(state="disabled"))
            self.processing = False
            self.progress_var.set(0)
            self.progress_label.config(text="0%")
            self.close_video_captures()

    def process_single_video_logic(self, width, height, period_input, num_views, angle_deg, output_fps, codec_str, shift=0):
        """处理单个视频（支持视点偏移）"""
        if len(self.video_captures) < num_views:
            raise ValueError(f"需要 {num_views} 个视频，但只选择了 {len(self.video_captures)} 个")
        
        # 获取视频信息
        total_frames = int(self.video_captures[0].get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 验证周期参数
        if num_views == 2 and period_input != int(period_input):
            is_balanced_mode = True
            P = period_input
        else:
            is_balanced_mode = False
            P = round(period_input)
            if P % num_views != 0:
                raise ValueError(f'周期 {P} 必须能被视点数 {num_views} 整除')

        num_channels = 3
        original_width = width
        
        tan_angle = np.tan(np.radians(angle_deg))
        offset_per_row_float = tan_angle * num_channels
        
        max_offset = abs(offset_per_row_float) * (height - 1)
        subpixel_width_original = original_width * num_channels
        num_tiles = max(3, int(np.ceil((subpixel_width_original + 2 * max_offset) / subpixel_width_original)))
        
        width = original_width * num_tiles
        subpixel_width = width * num_channels

        # 预计算视点分配表
        view_assignment_table = self.precompute_view_assignment(
            height, subpixel_width, P, num_views, angle_deg, is_balanced_mode
        )

        # 创建视频写入器
        file_ext = os.path.splitext(self.save_path)[1].lower()
        if file_ext == '.avi':
            fourcc_code = cv2.VideoWriter_fourcc(*codec_str)
        else:
            fourcc_code = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.video_writer = cv2.VideoWriter(
            self.save_path,
            fourcc_code,
            output_fps,
            (original_width, height)
        )
        
        if not self.video_writer.isOpened():
            raise ValueError(f"无法创建视频写入器")

        # 逐帧处理
        start_time = time.time()
        
        for frame_idx in range(total_frames):
            if self.cancel_flag:
                break
            
            # 读取所有视点的当前帧（应用偏移）
            frame_list = []
            for i in range(num_views):
                cap = self.video_captures[(i + shift) % num_views]
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if frame_rgb.shape[:2] != (height, original_width):
                    frame_rgb = cv2.resize(frame_rgb, (original_width, height))
                
                frame_rgb = np.tile(frame_rgb, (1, num_tiles, 1))
                
                frame_list.append(frame_rgb)
            
            if len(frame_list) < num_views:
                break
            
            # 融合处理
            fusion_rgb = self.fuse_frames(frame_list, view_assignment_table, height, width, num_views, num_tiles, original_width)
            
            # 写入视频
            fusion_bgr = cv2.cvtColor(fusion_rgb, cv2.COLOR_RGB2BGR)
            self.video_writer.write(fusion_bgr)
        
        self.video_writer.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoStereoFusionApp(root)
    root.mainloop()