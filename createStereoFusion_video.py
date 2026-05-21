import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
import threading
import time

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

class StereoFusionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多视点融合图生成器 (GUI版) - 支持视频处理")
        self.root.geometry("950x750")
        
        self.img_cell = []
        self.file_paths = []
        self.save_path = ""
        self.is_video_mode = False  # 新增：标记是否为视频模式
        self.video_captures = []  # 新增：视频捕获对象列表
        self.video_writer = None  # 新增：视频写入对象
        self.processing = False  # 新增：处理状态标记
        self.cancel_flag = False  # 新增：取消标记
        
        self.create_widgets()

    def create_widgets(self):
        # 顶部参数区域
        param_frame = ttk.LabelFrame(self.root, text="参数设置", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5)

        # 第一行：分辨率
        ttk.Label(param_frame, text="分辨率:").grid(row=0, column=0, sticky="e")
        self.width_var = tk.StringVar(value="1920")
        self.height_var = tk.StringVar(value="1080")
        ttk.Entry(param_frame, textvariable=self.width_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(param_frame, text="x").grid(row=0, column=2)
        ttk.Entry(param_frame, textvariable=self.height_var, width=8).grid(row=0, column=3, padx=2)

        # 第二行：周期和视点数
        ttk.Label(param_frame, text="周期 (Period):").grid(row=0, column=4, sticky="e", padx=(10,0))
        self.period_var = tk.StringVar(value="8")
        ttk.Entry(param_frame, textvariable=self.period_var, width=8).grid(row=0, column=5, padx=2)
        
        ttk.Label(param_frame, text="视点数:").grid(row=0, column=6, sticky="e", padx=(10,0))
        self.views_var = tk.StringVar(value="2")
        ttk.Entry(param_frame, textvariable=self.views_var, width=5).grid(row=0, column=7, padx=2)

        # 第三行：角度
        ttk.Label(param_frame, text="斜向偏移角度:").grid(row=1, column=0, sticky="e", pady=(5,0))
        self.angle_var = tk.StringVar(value="18.435")
        ttk.Entry(param_frame, textvariable=self.angle_var, width=8).grid(row=1, column=1, padx=2, pady=(5,0))
        ttk.Label(param_frame, text="度").grid(row=1, column=2, pady=(5,0))

        # 新增：视频处理参数
        ttk.Label(param_frame, text="输出帧率:").grid(row=1, column=4, sticky="e", pady=(5,0))
        self.fps_var = tk.StringVar(value="30")
        ttk.Entry(param_frame, textvariable=self.fps_var, width=8).grid(row=1, column=5, padx=2, pady=(5,0))
        
        ttk.Label(param_frame, text="编解码器:").grid(row=1, column=6, sticky="e", pady=(5,0))
        self.codec_var = tk.StringVar(value="mp4v")
        codec_combo = ttk.Combobox(param_frame, textvariable=self.codec_var, width=8, 
                                   values=["mp4v", "XVID", "H264", "MJPG"])
        codec_combo.grid(row=1, column=7, padx=2, pady=(5,0))

        # 图片/视频选择区域
        file_frame = ttk.LabelFrame(self.root, text="输入选择", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)

        self.file_label = ttk.Label(file_frame, text="未选择文件", foreground="gray")
        self.file_label.pack(side="left", fill="x", expand=True)

        btn_frame_input = ttk.Frame(file_frame)
        btn_frame_input.pack(side="right")
        
        ttk.Button(btn_frame_input, text="选择图片", command=self.select_files).pack(side="top", pady=2)
        ttk.Button(btn_frame_input, text="选择视频", command=self.select_videos).pack(side="top", pady=2)
        
        # 新增：黑白图和红蓝图按钮
        btn_frame_colors = ttk.Frame(file_frame)
        btn_frame_colors.pack(side="right", padx=5)
        
        # 黑白图按钮组
        bw_frame = ttk.LabelFrame(btn_frame_colors, text="黑白融合图", padding=5)
        bw_frame.pack(side="top", pady=2)
        ttk.Button(bw_frame, text="左黑右白", command=lambda: self.generate_black_white_images('black_white')).pack(side="left", padx=2)
        ttk.Button(bw_frame, text="左白右黑", command=lambda: self.generate_black_white_images('white_black')).pack(side="left", padx=2)
        
        # 红蓝图按钮
        rb_frame = ttk.LabelFrame(btn_frame_colors, text="红蓝融合图", padding=5)
        rb_frame.pack(side="top", pady=2)
        ttk.Button(rb_frame, text="左红右蓝", command=self.generate_red_blue_images).pack(side="left", padx=2)

        # 保存路径选择区域
        save_frame = ttk.LabelFrame(self.root, text="输出设置", padding=10)
        save_frame.pack(fill="x", padx=10, pady=5)

        self.save_label = ttk.Label(save_frame, text="保存路径: 未选择", foreground="gray")
        self.save_label.pack(side="left", fill="x", expand=True)

        ttk.Button(save_frame, text="选择保存位置", command=self.select_save_path).pack(side="right", padx=5)

        # 新增：进度条
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
        
        self.status_text = tk.Text(status_frame, height=10, state="disabled", bg="#f0f0f0")
        self.status_text.pack(fill="both", expand=True)
        
        # 底部按钮
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        self.cancel_btn = ttk.Button(btn_frame, text="取消", command=self.cancel_processing, state="disabled")
        self.cancel_btn.pack(side="left", padx=5)
        
        self.run_btn = ttk.Button(btn_frame, text="开始生成融合图/视频", command=self.start_processing)
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

    def select_files(self):
        """选择图片文件"""
        paths = filedialog.askopenfilenames(
            title="选择视点图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if paths:
            self.file_paths = list(paths)
            self.is_video_mode = False
            self.file_label.config(text=f"已选择 {len(paths)} 张图片: {os.path.basename(paths[0])}...", foreground="black")
            self.log(f"已加载图片: {paths[0]}")
            self.img_cell = []
            # 关闭视频捕获（如果有）
            self.close_video_captures()

    def select_videos(self):
        """选择视频文件"""
        paths = filedialog.askopenfilenames(
            title="选择视点视频",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv")]
        )
        if paths:
            self.file_paths = list(paths)
            self.is_video_mode = True
            
            # 打开视频文件进行验证
            self.video_captures = []
            for path in paths:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    messagebox.showerror("错误", f"无法打开视频文件: {path}")
                    self.close_video_captures()
                    return
                self.video_captures.append(cap)
            
            # 获取视频信息
            num_frames = int(self.video_captures[0].get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_captures[0].get(cv2.CAP_PROP_FPS)
            width = int(self.video_captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.file_label.config(text=f"已选择 {len(paths)} 个视频: {os.path.basename(paths[0])}... ({num_frames}帧, {fps:.2f}fps)", foreground="blue")
            self.log(f"已加载视频: {paths[0]}")
            self.log(f"视频信息: {num_frames}帧, {fps:.2f}fps, {width}x{height}")
            
            # 设置默认参数
            self.width_var.set(str(width))
            self.height_var.set(str(height))
            self.fps_var.set(str(fps))
            
            # 清空图片缓存
            self.img_cell = []

    def close_video_captures(self):
        """关闭所有视频捕获对象"""
        for cap in self.video_captures:
            if cap is not None:
                cap.release()
        self.video_captures = []

    def select_save_path(self):
        """选择保存路径"""
        if self.is_video_mode:
            # 视频模式
            path = filedialog.asksaveasfilename(
                title="选择保存位置及文件名",
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")],
                initialfile=f"fusion_video_P{self.period_var.get()}_V{self.views_var.get()}.mp4"
            )
        else:
            # 图片模式
            path = filedialog.asksaveasfilename(
                title="选择保存位置及文件名",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
                initialfile=f"fusion_P{self.period_var.get()}_V{self.views_var.get()}.png"
            )
        
        if path:
            self.save_path = path
            self.save_label.config(text=f"保存路径: {path}", foreground="black")

    def generate_black_white_images(self, mode='black_white'):
        """生成黑白融合图测试图片"""
        self.log(f"正在生成黑白测试图片 ({mode})...")
        w, h = int(self.width_var.get()), int(self.height_var.get())
        self.img_cell = []
        num_views = int(self.views_var.get())
        
        if num_views != 2:
            self.log(f"警告: 黑白图模式自动将视点数设置为2")
            num_views = 2
            self.views_var.set("2")
        
        if mode == 'black_white':
            black_img = np.zeros((h, w, 3), dtype=np.uint8)
            white_img = np.ones((h, w, 3), dtype=np.uint8) * 255
            self.img_cell = [black_img, white_img]
            display_text = "左黑右白"
        else:
            white_img = np.ones((h, w, 3), dtype=np.uint8) * 255
            black_img = np.zeros((h, w, 3), dtype=np.uint8)
            self.img_cell = [white_img, black_img]
            display_text = "左白右黑"
        
        self.file_paths = []
        self.is_video_mode = False
        self.file_label.config(text=f"使用生成的黑白测试图片 ({display_text})", foreground="blue")
        self.log(f"黑白测试图片生成完毕: {display_text}")

    def generate_red_blue_images(self):
        """生成红蓝融合图测试图片"""
        self.log("正在生成红蓝测试图片 (左红右蓝)...")
        w, h = int(self.width_var.get()), int(self.height_var.get())
        self.img_cell = []
        num_views = int(self.views_var.get())
        
        if num_views != 2:
            self.log(f"警告: 红蓝图模式自动将视点数设置为2")
            num_views = 2
            self.views_var.set("2")
        
        red_img = np.zeros((h, w, 3), dtype=np.uint8)
        red_img[:, :, 0] = 255
        red_img[:, :, 1] = 0
        red_img[:, :, 2] = 0
        
        blue_img = np.zeros((h, w, 3), dtype=np.uint8)
        blue_img[:, :, 0] = 0
        blue_img[:, :, 1] = 0
        blue_img[:, :, 2] = 255
        
        red_img_rgb = cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB)
        blue_img_rgb = cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB)
        
        self.img_cell = [red_img_rgb, blue_img_rgb]
        
        self.file_paths = []
        self.is_video_mode = False
        self.file_label.config(text="使用生成的红蓝测试图片 (左红右蓝)", foreground="blue")
        self.log("红蓝测试图片生成完毕: 左红右蓝")

    def cancel_processing(self):
        """取消处理"""
        self.cancel_flag = True
        self.log("正在取消处理...")
        self.cancel_btn.config(state="disabled")

    def start_processing(self):
        """开始处理"""
        if not self.img_cell and not self.file_paths:
            messagebox.showwarning("警告", "请先选择图片/视频或生成测试图！")
            return
        
        if self.is_video_mode and len(self.video_captures) == 0:
            messagebox.showwarning("警告", "视频捕获对象未初始化，请重新选择视频！")
            return
        
        # 禁用按钮防止重复点击
        self.run_btn.config(state="disabled", text="处理中...")
        self.cancel_btn.config(state="normal")
        self.cancel_flag = False
        self.processing = True
        
        self.log("\n" + "="*50)
        self.log(f"=== {'开始视频处理' if self.is_video_mode else '开始图片处理'} ===")
        self.log("="*50 + "\n")
        
        # 在新线程中运行
        threading.Thread(target=self.process_logic, daemon=True).start()

    def process_logic(self):
        """处理逻辑"""
        try:
            if self.is_video_mode:
                self.process_video_logic()
            else:
                self.process_image_logic()
        except Exception as e:
            self.log(f"发生错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("错误", str(e))
        finally:
            self.run_btn.config(state="normal", text="开始生成融合图/视频")
            self.cancel_btn.config(state="disabled")
            self.processing = False
            self.progress_var.set(0)
            self.progress_label.config(text="0%")
            self.close_video_captures()

    def process_image_logic(self):
        """图片处理逻辑（原有功能）"""
        # 1. 获取参数
        width = int(self.width_var.get())
        height = int(self.height_var.get())
        period_input = float(self.period_var.get())
        num_views = int(self.views_var.get())
        angle_deg = float(self.angle_var.get())

        # 2. 加载/预处理图片
        if not self.img_cell:
            self.log("正在加载图片...")
            self.img_cell = []
            
            for i, path in enumerate(self.file_paths[:num_views]):
                img_array = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    raise ValueError(f"无法读取图片，请检查文件是否损坏: {path}")
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.img_cell.append(img)

            if len(self.img_cell) == 0:
                raise ValueError("未成功加载任何图片，请检查文件路径。")

            # 统一尺寸
            h_ref, w_ref = self.img_cell[0].shape[:2]
            for i in range(1, len(self.img_cell)):
                if self.img_cell[i].shape[:2] != (h_ref, w_ref):
                    self.img_cell[i] = cv2.resize(self.img_cell[i], (w_ref, h_ref))
            
            width, height = w_ref, h_ref
            self.log(f"成功加载 {len(self.img_cell)} 张图片，分辨率: {width}x{height}")

        # 3. 算法逻辑
        if num_views == 2 and period_input != int(period_input):
            is_balanced_mode = True
            P = period_input
            self.log(">>> 启用非整数周期平衡模式")
        else:
            is_balanced_mode = False
            P = round(period_input)
            if P % num_views != 0:
                raise ValueError(f'周期 {P} 必须能被视点数 {num_views} 整除')

        # 扩展图像
        num_channels = 3
        original_width = width
        
        tan_angle = np.tan(np.radians(angle_deg))
        offset_per_row_float = tan_angle * num_channels
        
        max_offset = abs(offset_per_row_float) * (height - 1)
        subpixel_width_original = original_width * num_channels
        num_tiles = max(3, int(np.ceil((subpixel_width_original + 2 * max_offset) / subpixel_width_original)))
        
        self.log(f"扩展图像 {num_tiles} 倍以防止接缝...")
        for i in range(num_views):
            self.img_cell[i] = np.tile(self.img_cell[i], (1, num_tiles, 1))
        
        width = original_width * num_tiles
        subpixel_width = width * num_channels

        # 子像素化
        all_subpixels = []
        for v in range(num_views):
            subpixels = self.img_cell[v].reshape(height, -1)
            all_subpixels.append(subpixels.astype(np.uint8))

        # 融合
        self.log("正在计算融合图...")
        fusion_subpixels = np.zeros((height, subpixel_width), dtype=np.uint8)

        # 关键修复：生成精确的视点分配表
        view_assignment_table = np.zeros((height, subpixel_width), dtype=int)
        
        if is_balanced_mode:
            # 平衡模式：生成LR序列
            lr_sequence = generate_balanced_lr_sequence(P, subpixel_width)
            L_view, R_view = 0, 1
            
            for r in range(height):
                row_offset = r * offset_per_row_float
                
                positions = np.arange(subpixel_width)
                seq_positions = np.floor(positions - row_offset).astype(int) % subpixel_width
                view_assignment_table[r, :] = (lr_sequence[seq_positions] == 'L').astype(int)
        else:
            # 非平衡模式：生成视点索引序列
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

        # 验证所有视点都被使用
        used_views = np.unique(view_assignment_table)
        self.log(f"使用的视点索引: {used_views}")
        if len(used_views) < num_views:
            self.log(f"警告: 只使用了 {len(used_views)} 个视点，预期 {num_views} 个")
            self.log(f"缺失的视点: {set(range(num_views)) - set(used_views)}")
        else:
            self.log(f"✓ 所有 {num_views} 个视点都被正确使用")

        # 根据视点分配表进行融合
        for r in range(height):
            for v in range(num_views):
                mask = (view_assignment_table[r, :] == v)
                if np.any(mask):
                    fusion_subpixels[r, mask] = all_subpixels[v][r, mask]

        # 还原图像
        fusion_rgb = fusion_subpixels.reshape(height, width, num_channels)
        start_col = (num_tiles // 2) * original_width
        end_col = start_col + original_width
        fusion_rgb = fusion_rgb[:, start_col:end_col, :]

        # 4. 保存逻辑
        final_save_path = self.save_path
        
        if not final_save_path:
            self.log("正在请求保存路径...")
            final_save_path = self.root.after(0, lambda: filedialog.asksaveasfilename(
                title="选择保存位置",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
                initialfile=f"fusion_P{P}_V{num_views}_A{angle_deg}.png"
            ))
        
        if not final_save_path:
            self.log("用户取消了保存操作。")
            return

        save_dir = os.path.dirname(final_save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imencode('.png', cv2.cvtColor(fusion_rgb, cv2.COLOR_RGB2BGR))[1].tofile(final_save_path)
        
        self.log(f"处理完成！已保存至: {final_save_path}")
        self.log("正在显示预览...")
        
        # 显示预览
        plt.figure(figsize=(10, 6))
        plt.imshow(fusion_rgb)
        plt.title(f"融合结果 - 周期: {P}, 视点数: {num_views}, 角度: {angle_deg}°")
        plt.axis('off')
        plt.show()

    def process_video_logic(self):
        """视频处理逻辑（新增功能）"""
        self.log("=== 视频处理模式 ===\n")
        
        # 1. 获取参数
        width = int(self.width_var.get())
        height = int(self.height_var.get())
        period_input = float(self.period_var.get())
        num_views = int(self.views_var.get())
        angle_deg = float(self.angle_var.get())
        output_fps = float(self.fps_var.get())
        codec_str = self.codec_var.get()
        
        # 2. 验证视频数量
        if len(self.video_captures) < num_views:
            raise ValueError(f"需要 {num_views} 个视频，但只选择了 {len(self.video_captures)} 个")
        
        # 3. 获取视频信息
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
        
        # 使用第一个视频的帧数作为总帧数
        total_frames = video_frames[0]
        self.log(f"\n总帧数: {total_frames}")
        
        # 4. 算法逻辑（与图片模式相同）
        if num_views == 2 and period_input != int(period_input):
            is_balanced_mode = True
            P = period_input
            self.log(">>> 启用非整数周期平衡模式")
        else:
            is_balanced_mode = False
            P = round(period_input)
            if P % num_views != 0:
                raise ValueError(f'周期 {P} 必须能被视点数 {num_views} 整除')

        # 扩展图像（使用第一帧进行预处理）
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

        # 生成视点分配表（与图片模式相同）
        view_assignment_table = np.zeros((height, subpixel_width), dtype=int)
        
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

        # 5. 准备视频写入器
        final_save_path = self.save_path
        
        if not final_save_path:
            self.log("正在请求保存路径...")
            final_save_path = self.root.after(0, lambda: filedialog.asksaveasfilename(
                title="选择保存位置",
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")],
                initialfile=f"fusion_video_P{P}_V{num_views}_A{angle_deg}.mp4"
            ))
        
        if not final_save_path:
            self.log("用户取消了保存操作。")
            return

        save_dir = os.path.dirname(final_save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 根据扩展名选择编解码器
        file_ext = os.path.splitext(final_save_path)[1].lower()
        if file_ext == '.avi':
            fourcc_code = cv2.VideoWriter_fourcc(*codec_str)
        else:  # .mp4
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

        # 6. 逐帧处理
        self.log("开始逐帧处理...")
        start_time = time.time()
        
        for frame_idx in range(total_frames):
            if self.cancel_flag:
                self.log("用户取消了处理")
                break
            
            # 读取所有视频的当前帧
            frame_list = []
            for cap in self.video_captures[:num_views]:
                ret, frame = cap.read()
                if not ret:
                    self.log(f"警告: 视频读取结束于帧 {frame_idx}")
                    break
                
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 调整尺寸（如果需要）
                if frame_rgb.shape[:2] != (height, original_width):
                    frame_rgb = cv2.resize(frame_rgb, (original_width, height))
                
                # 扩展图像
                frame_rgb = np.tile(frame_rgb, (1, num_tiles, 1))
                
                frame_list.append(frame_rgb)
            
            if len(frame_list) < num_views:
                self.log("视频帧读取不完整，停止处理")
                break
            
            # 子像素化
            all_subpixels = []
            for v in range(num_views):
                subpixels = frame_list[v].reshape(height, -1)
                all_subpixels.append(subpixels.astype(np.uint8))
            
            # 融合（与图片模式相同）
            fusion_subpixels = np.zeros((height, subpixel_width), dtype=np.uint8)
            
            for r in range(height):
                for v in range(num_views):
                    mask = (view_assignment_table[r, :] == v)
                    if np.any(mask):
                        fusion_subpixels[r, mask] = all_subpixels[v][r, mask]
            
            # 还原图像
            fusion_rgb = fusion_subpixels.reshape(height, width, num_channels)
            start_col = (num_tiles // 2) * original_width
            end_col = start_col + original_width
            fusion_rgb = fusion_rgb[:, start_col:end_col, :]
            
            # 转换为BGR并写入视频
            fusion_bgr = cv2.cvtColor(fusion_rgb, cv2.COLOR_RGB2BGR)
            self.video_writer.write(fusion_bgr)
            
            # 更新进度
            if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                self.update_progress(frame_idx + 1, total_frames, frame_idx + 1)
                elapsed_time = time.time() - start_time
                fps_current = (frame_idx + 1) / elapsed_time if elapsed_time > 0 else 0
                self.log(f"处理帧 {frame_idx + 1}/{total_frames} ({fps_current:.1f} fps)")
        
        # 7. 完成处理
        self.video_writer.release()
        elapsed_time = time.time() - start_time
        avg_fps = total_frames / elapsed_time if elapsed_time > 0 else 0
        
        self.log("\n" + "="*50)
        self.log(f"视频处理完成！")
        self.log(f"总帧数: {total_frames}")
        self.log(f"处理时间: {elapsed_time:.2f}秒")
        self.log(f"平均速度: {avg_fps:.2f} fps")
        self.log(f"已保存至: {final_save_path}")
        self.log("="*50)

if __name__ == "__main__":
    root = tk.Tk()
    app = StereoFusionApp(root)
    root.mainloop()