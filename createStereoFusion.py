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
import json

# --- 配置文件路径 ---
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stereo_fusion_config.json")

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

class StereoFusionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多视点融合图生成器 (GUI版)")
        self.root.geometry("900x700")
        
        self.img_cell = []
        self.file_paths = []
        self.save_path = ""
        self.dialog_queue = queue.Queue()
        self.preview_queue = queue.Queue()
        
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
            "angle": "18.435"
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
                "angle": self.angle_var.get()
            }
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置文件失败: {e}")

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

        # 第三行：角度
        ttk.Label(param_frame, text="斜向偏移角度:").grid(row=1, column=0, sticky="e", pady=(5,0))
        self.angle_var = tk.StringVar(value=self.config["angle"])
        ttk.Entry(param_frame, textvariable=self.angle_var, width=8).grid(row=1, column=1, padx=2, pady=(5,0))
        ttk.Label(param_frame, text="度").grid(row=1, column=2, pady=(5,0))

        # 图片选择区域
        file_frame = ttk.LabelFrame(self.root, text="图片输入", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)

        self.file_label = ttk.Label(file_frame, text="未选择图片", foreground="gray")
        self.file_label.pack(side="left", fill="x", expand=True)

        ttk.Button(file_frame, text="选择图片", command=self.select_files).pack(side="right", padx=5)
        
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

        self.save_label = ttk.Label(save_frame, text="保存路径: 未选择 (将在处理时询问)", foreground="gray")
        self.save_label.pack(side="left", fill="x", expand=True)

        ttk.Button(save_frame, text="选择保存位置", command=self.select_save_path).pack(side="right", padx=5)

        # 【修改】连续视点融合按钮区域 - 只保留这一个按钮
        continuous_frame = ttk.LabelFrame(self.root, text="连续视点融合", padding=10)
        continuous_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(continuous_frame, text="输出一个周期内所有视点顺序的融合图", foreground="blue").pack(side="left", padx=5)
        self.continuous_btn = ttk.Button(continuous_frame, text="输出连续视点融合图(一个周期)", 
                                        command=self.start_continuous_processing)
        self.continuous_btn.pack(side="right", padx=5)

        # 状态显示区域
        status_frame = ttk.LabelFrame(self.root, text="运行状态", padding=10)
        status_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=10, state="disabled", bg="#f0f0f0")
        self.status_text.pack(fill="both", expand=True)
        
        # 【修改】底部按钮 - 删除了重复的连续视点按钮
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        self.run_btn = ttk.Button(btn_frame, text="开始生成融合图", command=self.start_processing)
        self.run_btn.pack(side="right")

    def log(self, message):
        self.status_text.config(state="normal")
        self.status_text.insert("end", message + "\n")
        self.status_text.see("end")
        self.status_text.config(state="disabled")

    def select_files(self):
        paths = filedialog.askopenfilenames(
            title="选择视点图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if paths:
            self.file_paths = list(paths)
            self.file_label.config(text=f"已选择 {len(paths)} 张图片: {os.path.basename(paths[0])}...", foreground="black")
            self.log(f"已加载图片: {paths[0]}")
            self.img_cell = []

    def select_save_path(self):
        path = filedialog.asksaveasfilename(
            title="选择保存位置及文件名",
            defaultextension=".bmp",
            filetypes=[("BMP Image", "*.bmp"), ("PNG Image", "*.png"), ("All Files", "*.*")],
            initialfile=f"fusion_P{self.period_var.get()}_V{self.views_var.get()}"
        )
        if path:
            if not path.lower().endswith('.bmp'):
                path = os.path.splitext(path)[0] + '.bmp'
            self.save_path = path
            self.save_label.config(text=f"保存路径: {path}", foreground="black")

    def generate_black_white_images(self, mode='black_white'):
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
        self.file_label.config(text=f"使用生成的黑白测试图片 ({display_text})", foreground="blue")
        self.log(f"黑白测试图片生成完毕: {display_text}")

    def generate_red_blue_images(self):
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
        self.file_label.config(text="使用生成的红蓝测试图片 (左红右蓝)", foreground="blue")
        self.log("红蓝测试图片生成完毕: 左红右蓝")

    def start_continuous_processing(self):
        if not self.img_cell and not self.file_paths:
            messagebox.showwarning("警告", "请先选择图片或生成测试图！")
            return
        
        self.continuous_btn.config(state="disabled", text="处理中...")
        self.run_btn.config(state="disabled")
        self.log("\n=== 开始连续视点融合处理 ===")
        
        threading.Thread(target=self.process_continuous_logic, daemon=True).start()

    def start_processing(self):
        if not self.img_cell and not self.file_paths:
            messagebox.showwarning("警告", "请先选择图片或生成测试图！")
            return
        
        self.run_btn.config(state="disabled", text="处理中...")
        self.continuous_btn.config(state="disabled")
        self.log("\n=== 开始处理 ===")
        
        threading.Thread(target=self.process_logic, daemon=True).start()

    def process_continuous_logic(self):
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            period_input = float(self.period_var.get())
            num_views = int(self.views_var.get())
            angle_deg = float(self.angle_var.get())

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

                h_ref, w_ref = self.img_cell[0].shape[:2]
                for i in range(1, len(self.img_cell)):
                    if self.img_cell[i].shape[:2] != (h_ref, w_ref):
                        self.img_cell[i] = cv2.resize(self.img_cell[i], (w_ref, h_ref))
                
                width, height = w_ref, h_ref
                self.log(f"成功加载 {len(self.img_cell)} 张图片，分辨率: {width}x{height}")

            self.log("正在请求保存目录...")
            
            while not self.dialog_queue.empty():
                try:
                    self.dialog_queue.get_nowait()
                except:
                    pass
            
            self.root.after(0, lambda: self.dialog_queue.put(
                filedialog.askdirectory(title="选择保存目录（用于保存多张融合图）")
            ))
            
            try:
                save_dir = self.dialog_queue.get(timeout=60)
            except queue.Empty:
                self.log("超时：用户未在规定时间内选择目录")
                return
            
            if not save_dir or not os.path.exists(save_dir):
                self.log("用户取消了保存操作或目录不存在。")
                return

            self.log(f"\n开始生成 {num_views} 张连续视点融合图...")
            
            for shift in range(num_views):
                self.log(f"\n--- 处理第 {shift + 1}/{num_views} 张融合图 (视点顺序: {[(i + shift) % num_views + 1 for i in range(num_views)]}) ---")
                
                shifted_views = [self.img_cell[(i + shift) % num_views] for i in range(num_views)]
                
                original_img_cell = self.img_cell.copy()
                self.img_cell = shifted_views
                
                try:
                    fusion_rgb = self.generate_single_fusion(width, height, period_input, num_views, angle_deg)
                    
                    filename = f"fusion_shift{shift+1}_P{period_input}_V{num_views}_A{angle_deg}.bmp"
                    save_path = os.path.join(save_dir, filename)
                    
                    cv2.imencode('.bmp', cv2.cvtColor(fusion_rgb, cv2.COLOR_RGB2BGR))[1].tofile(save_path)
                    
                    self.log(f"✓ 第 {shift + 1} 张融合图已保存: {filename}")
                    
                except Exception as e:
                    self.log(f"✗ 第 {shift + 1} 张融合图处理失败: {str(e)}")
                    continue
                finally:
                    self.img_cell = original_img_cell

            self.log(f"\n=== 连续视点融合处理完成！共生成 {num_views} 张融合图 ===")
            self.log(f"所有文件已保存至: {save_dir}")
            
            self.root.after(0, lambda: messagebox.showinfo(
                "完成", 
                f"连续视点融合处理完成！\n共生成 {num_views} 张融合图\n保存位置: {save_dir}"
            ))

        except Exception as e:
            self.log(f"发生错误: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.root.after(0, lambda: self.run_btn.config(state="normal", text="开始生成融合图"))
            self.root.after(0, lambda: self.continuous_btn.config(state="normal", text="输出连续视点融合图(一个周期)"))

    def generate_single_fusion(self, width, height, period_input, num_views, angle_deg):
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
        for i in range(num_views):
            self.img_cell[i] = np.tile(self.img_cell[i], (1, num_tiles, 1))
        
        width = original_width * num_tiles
        subpixel_width = width * num_channels

        all_subpixels = []
        for v in range(num_views):
            subpixels = self.img_cell[v].reshape(height, -1)
            all_subpixels.append(subpixels.astype(np.uint8))

        self.log("正在计算融合图...")
        fusion_subpixels = np.zeros((height, subpixel_width), dtype=np.uint8)

        view_assignment_table = np.zeros((height, subpixel_width), dtype=int)
        
        if is_balanced_mode:
            lr_sequence = generate_balanced_lr_sequence(P, subpixel_width)
            L_view, R_view = 0, 1
            
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
            self.log(f"缺失的视点: {set(range(num_views)) - set(used_views)}")
        else:
            self.log(f"✓ 所有 {num_views} 个视点都被正确使用")

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

    def process_logic(self):
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            period_input = float(self.period_var.get())
            num_views = int(self.views_var.get())
            angle_deg = float(self.angle_var.get())

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

                h_ref, w_ref = self.img_cell[0].shape[:2]
                for i in range(1, len(self.img_cell)):
                    if self.img_cell[i].shape[:2] != (h_ref, w_ref):
                        self.img_cell[i] = cv2.resize(self.img_cell[i], (w_ref, h_ref))
                
                width, height = w_ref, h_ref
                self.log(f"成功加载 {len(self.img_cell)} 张图片，分辨率: {width}x{height}")

            fusion_rgb = self.generate_single_fusion(width, height, period_input, num_views, angle_deg)

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
                        defaultextension=".bmp",
                        filetypes=[("BMP Image", "*.bmp"), ("PNG Image", "*.png"), ("All Files", "*.*")],
                        initialfile=f"fusion_P{period_input}_V{num_views}_A{angle_deg}"
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

            if not final_save_path.lower().endswith('.bmp'):
                final_save_path = os.path.splitext(final_save_path)[0] + '.bmp'

            save_dir = os.path.dirname(final_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            cv2.imencode('.bmp', cv2.cvtColor(fusion_rgb, cv2.COLOR_RGB2BGR))[1].tofile(final_save_path)
            
            self.log(f"处理完成！已保存至: {final_save_path}")
            self.log("正在显示预览...")
            
            preview_data = {
                'image': fusion_rgb,
                'period': period_input,
                'views': num_views,
                'angle': angle_deg
            }
            self.preview_queue.put(preview_data)

        except Exception as e:
            self.log(f"发生错误: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.root.after(0, lambda: self.run_btn.config(state="normal", text="开始生成融合图"))
            self.root.after(0, lambda: self.continuous_btn.config(state="normal", text="输出连续视点融合图(一个周期)"))

if __name__ == "__main__":
    root = tk.Tk()
    app = StereoFusionApp(root)
    root.mainloop()