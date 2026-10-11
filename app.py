"""
Academic Frontier — CuPy Neural Net Explorer v13.0
Advanced dashboard featuring validation tracking, spectral norm generalization bounds, 
FGSM adversarial attacks, and polar coordinate manifolds.
"""

import sys
import time
import numpy as np
import io
import imageio.v2 as imageio

try:
    import cupy as cp
    USING_GPU = True
    _to_np = cp.asnumpy
except ImportError:
    import numpy as cp
    USING_GPU = False
    _to_np = np.asarray
    cp.asnumpy = np.asarray

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QSlider, QPushButton, QTextEdit,
                               QComboBox, QGroupBox, QDialog, QScrollArea, QSpinBox,
                               QDoubleSpinBox, QCheckBox, QFileDialog, QGridLayout,
                               QTabWidget, QFrame, QSizePolicy, QListWidget, QListWidgetItem,
                               QLineEdit)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QCursor

# ==========================================
# HELPERS
# ==========================================
def array_to_pixmap(arr, colormap='Grayscale'):
    arr = np.clip(arr, 0, 1)
    if colormap == 'Grayscale':
        img = (arr * 255).astype(np.uint8)
        img = np.ascontiguousarray(img)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        cmap = plt.get_cmap(colormap)
        colored = cmap(arr)[:, :, :3]
        img = (colored * 255).astype(np.uint8)
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def style_mpl_axes(ax, dark=True):
    color = '#cccccc' if dark else '#333333'
    ax.tick_params(colors=color, labelsize=8)
    for s in ax.spines.values(): s.set_color(color)

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100.0
    return 10 * np.log10(1.0 / mse)

def ease_linear(t): return t
def ease_smoothstep(t): return t * t * (3 - 2 * t)
def ease_in_out_cubic(t): return 4*t**3 if t < 0.5 else 1 - (-2*t + 2)**3 / 2

# ==========================================
# 1. CUPY NEURAL NETWORK
# ==========================================
class BaseNet:
    def __init__(self, layer_shapes, activation='relu'):
        self.layers = []
        self.layer_shapes = layer_shapes
        self.activation = activation
        self.cache = None
        self.quant_bits = 16
        self.dropout_rate = 0.0
        self.grad_noise_std = 0.0
        self.arch_text = self._generate_arch_text()
        self.randomize_weights('he')

    def _generate_arch_text(self):
        text = "Architecture Details\n" + "═" * 38 + "\n"
        total_params = 0
        for i, (in_f, out_f) in enumerate(self.layer_shapes):
            params = (in_f * out_f) + out_f
            total_params += params
            text += f"Layer {i+1}: Linear({in_f} → {out_f})\n"
            text += f"  Weights: [{out_f}, {in_f}]  Bias: [{out_f}]\n"
            text += f"  Params: {params:,}\n"
            act = self.activation if i < len(self.layer_shapes) - 1 else "Sigmoid"
            text += f"  Activation: {act}\n" + "─" * 38 + "\n"
        text += f"Total Parameters: {total_params:,}\n"
        return text

    def _act(self, z, last=False):
        if last: return 1.0 / (1.0 + cp.exp(-cp.clip(z, -50, 50)))
        if self.activation == 'relu': return cp.maximum(0, z)
        elif self.activation == 'leaky_relu': return cp.where(z > 0, z, 0.01 * z).astype(z.dtype)
        elif self.activation == 'tanh': return cp.tanh(z)
        elif self.activation == 'sigmoid': return 1.0 / (1.0 + cp.exp(-cp.clip(z, -50, 50)))
        return cp.maximum(0, z)

    def _act_grad(self, z, last=False):
        if last:
            a = 1.0 / (1.0 + cp.exp(-cp.clip(z, -50, 50)))
            return a * (1 - a)
        if self.activation == 'relu': return (z > 0).astype(z.dtype)
        elif self.activation == 'leaky_relu': return cp.where(z > 0, 1.0, 0.01).astype(z.dtype)
        elif self.activation == 'tanh': return 1 - cp.tanh(z) ** 2
        elif self.activation == 'sigmoid':
            a = 1.0 / (1.0 + cp.exp(-cp.clip(z, -50, 50)))
            return a * (1 - a)
        return (z > 0).astype(z.dtype)

    def _quantize(self, W):
        if self.quant_bits >= 16: return W
        max_val = 2**(self.quant_bits - 1) - 1
        return cp.round(W * max_val) / max_val

    def randomize_weights(self, init_type='he'):
        self.layers = []
        for in_f, out_f in self.layer_shapes:
            if init_type == 'he':
                scale = cp.sqrt(2.0 / in_f) if self.activation in ('relu', 'leaky_relu') else cp.sqrt(1.0 / in_f)
                W = cp.random.randn(out_f, in_f).astype(cp.float32) * scale
            elif init_type == 'xavier':
                scale = cp.sqrt(1.0 / in_f)
                W = cp.random.randn(out_f, in_f).astype(cp.float32) * scale
            elif init_type == 'uniform':
                W = cp.random.uniform(-0.1, 0.1, (out_f, in_f)).astype(cp.float32)
            elif init_type == 'zeros':
                W = cp.zeros((out_f, in_f), dtype=cp.float32)
            b = cp.zeros(out_f, dtype=cp.float32)
            self.layers.append({'W': W, 'b': b})
        self.reset_optim_state()

    def reset_optim_state(self):
        self.optim_state = [{'m': cp.zeros_like(l['W']), 'v': cp.zeros_like(l['W']),
                             'm_b': cp.zeros_like(l['b']), 'v_b': cp.zeros_like(l['b'])} for l in self.layers]
        self.t = 0

    def get_spectral_norms(self):
        norms = []
        for l in self.layers:
            # Compute largest singular value (spectral norm)
            s = cp.linalg.svd(l['W'], compute_uv=False)
            norms.append(float(s[0]))
        return norms

    def forward(self, z, cache=False):
        a = z
        activations = [a]
        pre_acts = []
        for i, layer in enumerate(self.layers):
            W = self._quantize(layer['W'])
            b = self._quantize(layer['b'])
            z_pre = cp.dot(a, W.T) + b
            pre_acts.append(z_pre)
            a = self._act(z_pre, last=(i == len(self.layers) - 1))
            
            if cache and self.dropout_rate > 0 and i < len(self.layers) - 1:
                mask = (cp.random.rand(*a.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
                a = a * mask
                
            activations.append(a)
        if cache: self.cache = {'activations': activations, 'pre_acts': pre_acts}
        return a

    def backward(self, target):
        activations = self.cache['activations']
        pre_acts = self.cache['pre_acts']
        N = activations[0].shape[0]
        L = len(self.layers)
        grads = [{'W': None, 'b': None} for _ in range(L)]
        da = (2.0 / N) * (activations[-1] - target)
        dz = da * self._act_grad(pre_acts[-1], last=True)
        grad_norms = []
        grad_input = None
        
        for i in range(L - 1, -1, -1):
            grads[i]['W'] = cp.dot(dz.T, activations[i])
            grads[i]['b'] = cp.sum(dz, axis=0)
            grad_norms.append(float(cp.linalg.norm(grads[i]['W'])))
            if i > 0:
                da = cp.dot(dz, self.layers[i]['W'])
                dz = da * self._act_grad(pre_acts[i - 1], last=False)
            else:
                grad_input = cp.dot(dz, self.layers[0]['W'])
                
        return grads, grad_norms[::-1], grad_input

    def step(self, grads, lr, optim='sgd', freeze_flags=None):
        for i, layer in enumerate(self.layers):
            if freeze_flags and freeze_flags[i]: continue
            if self.grad_noise_std > 0:
                grads[i]['W'] += cp.random.randn(*grads[i]['W'].shape).astype(cp.float32) * self.grad_noise_std
                
            if optim == 'sgd':
                layer['W'] -= lr * grads[i]['W']
                layer['b'] -= lr * grads[i]['b']
            elif optim == 'adam':
                self.t += 1
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                self.optim_state[i]['m'] = beta1 * self.optim_state[i]['m'] + (1 - beta1) * grads[i]['W']
                self.optim_state[i]['v'] = beta2 * self.optim_state[i]['v'] + (1 - beta2) * (grads[i]['W']**2)
                m_hat = self.optim_state[i]['m'] / (1 - beta1**self.t)
                v_hat = self.optim_state[i]['v'] / (1 - beta2**self.t)
                layer['W'] -= lr * m_hat / (cp.sqrt(v_hat) + eps)
                
                self.optim_state[i]['m_b'] = beta1 * self.optim_state[i]['m_b'] + (1 - beta1) * grads[i]['b']
                self.optim_state[i]['v_b'] = beta2 * self.optim_state[i]['v_b'] + (1 - beta2) * (grads[i]['b']**2)
                m_b_hat = self.optim_state[i]['m_b'] / (1 - beta1**self.t)
                v_b_hat = self.optim_state[i]['v_b'] / (1 - beta2**self.t)
                layer['b'] -= lr * m_b_hat / (cp.sqrt(v_b_hat) + eps)

    def train_batch(self, z_batch, target_batch, lr=0.01, optim='sgd', freeze_flags=None, l1=0.0, tv=0.0):
        out = self.forward(z_batch, cache=True)
        mse_loss = cp.mean((out - target_batch) ** 2)
        l1_loss = sum(cp.mean(cp.abs(l['W'])) for l in self.layers) if l1 > 0 else 0.0
        tv_loss = cp.mean((out[:, 1:, :] - out[:, :-1, :])**2) + cp.mean((out[:, :, 1:] - out[:, :, :-1])**2) if tv > 0 else 0.0
        loss = float(mse_loss + l1 * l1_loss + tv * tv_loss)
        
        grads, grad_norms, _ = self.backward(target_batch)
        if l1 > 0:
            for i in range(len(self.layers)):
                grads[i]['W'] += l1 * cp.sign(self.layers[i]['W'])
                
        self.step(grads, lr, optim, freeze_flags)
        return loss, grad_norms

    def prune_weights(self, percentile=10):
        all_vals = cp.concatenate([l['W'].ravel() for l in self.layers])
        threshold = cp.percentile(cp.abs(all_vals), percentile)
        total_pruned = 0
        for layer in self.layers:
            mask = cp.abs(layer['W']) < threshold
            layer['W'][mask] = 0
            total_pruned += int(cp.sum(mask))
        return total_pruned

    def save_weights(self, path):
        save_dict = {'layer_shapes': np.array(self.layer_shapes), 'activation': np.array([self.activation])}
        for i, l in enumerate(self.layers):
            save_dict[f'W_{i}'] = _to_np(l['W']); save_dict[f'b_{i}'] = _to_np(l['b'])
        np.savez(path, **save_dict)

    def load_weights(self, path):
        data = np.load(path, allow_pickle=True)
        self.layers = []
        for i in range(len(self.layer_shapes)):
            self.layers.append({'W': cp.asarray(data[f'W_{i}']), 'b': cp.asarray(data[f'b_{i}'])})
        self.reset_optim_state()

class SmallMLP(BaseNet):
    def __init__(self, activation='relu'): super().__init__([(2, 16), (16, 784)], activation)
class MediumMLP(BaseNet):
    def __init__(self, activation='relu'): super().__init__([(2, 64), (64, 128), (128, 784)], activation)
class LargeMLP(BaseNet):
    def __init__(self, activation='relu'): super().__init__([(2, 256), (256, 128), (128, 64), (64, 784)], activation)

# ==========================================
# 2. TARGET PATTERN GENERATOR
# ==========================================
def generate_target_batch(z_batch, pattern_type='gaussian_blob', custom_img=None):
    N = z_batch.shape[0]
    if pattern_type == 'custom' and custom_img is not None:
        return cp.repeat(custom_img, N, axis=0)

    z1, z2 = z_batch[:, 0], z_batch[:, 1]
    y_c, x_c = cp.meshgrid(cp.arange(28, dtype=cp.float32), cp.arange(28, dtype=cp.float32), indexing='ij')
    cx = (14.0 + z1 * 2.0)[:, None, None]; cy = (14.0 + z2 * 2.0)[:, None, None]

    if pattern_type == 'gaussian_blob':
        sigma = 3.0 + 0.3 * cp.sqrt(z1**2 + z2**2)[:, None, None]
        img = cp.exp(-((x_c[None] - cx)**2 + (y_c[None] - cy)**2) / (2 * sigma**2))
    elif pattern_type == 'ring':
        r = 4.0 + cp.abs(z1)[:, None, None] * 3.0
        img = cp.exp(-((cp.sqrt((x_c[None] - cx)**2 + (y_c[None] - cy)**2) - r) / 1.5)**2)
    elif pattern_type == 'cross':
        img = cp.maximum(cp.exp(-((x_c[None] - cx) / 1.5)**2), cp.exp(-((y_c[None] - cy) / 1.5)**2))
    elif pattern_type == 'spiral':
        dx, dy = x_c[None] - cx, y_c[None] - cy
        r = cp.sqrt(dx**2 + dy**2); theta = cp.arctan2(dy, dx)
        img = 0.5 + 0.5 * cp.cos(theta * 3 + r * 0.5 - z1[:, None, None] * 2) * cp.exp(-r / (8 + z2[:, None, None] * 2))
    elif pattern_type == 'sine_wave':
        img = 0.5 + 0.5 * cp.sin(x_c[None] * (0.5 + z1[:, None, None]) + y_c[None] * (0.5 + z2[:, None, None]))
    elif pattern_type == 'checkerboard':
        size = 2 + cp.abs(z1)[:, None, None] * 3
        img = ((x_c[None] // size) + (y_c[None] // size)) % 2
        img = img * (0.3 + 0.7 * cp.abs(z2)[:, None, None])
    else: img = cp.zeros((N, 28, 28), dtype=cp.float32)
    return cp.clip(img, 0, 1).reshape(N, 784)

# ==========================================
# 3. MAIN APPLICATION
# ==========================================
class AcademicFrontierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Academic Frontier — CuPy Neural Net Explorer v13.0")
        self.resize(1650, 950)

        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            self.gpu_name = props['name']
            if isinstance(self.gpu_name, bytes): self.gpu_name = self.gpu_name.decode('utf-8').strip('\x00')
        except Exception: self.gpu_name = "NumPy CPU Backend"

        self.architectures = {
            "Small MLP (2→16→784)": SmallMLP,
            "Medium MLP (2→64→128→784)": MediumMLP,
            "Large MLP (2→256→128→64→784)": LargeMLP,
        }
        self.current_activation = 'relu'
        self.current_model = list(self.architectures.values())[0](self.current_activation)

        mock = cp.random.randn(800, 2).astype(cp.float32) * 2.5
        self.mock_latents = _to_np(mock)

        self.click_mode = 'navigate'; self.point_a_pos = None; self.point_b_pos = None
        self.auto_rotate_t = 0.0; self.loss_history = []; self.val_loss_history = []
        self.grad_norm_history = []; self.spec_norm_history = []
        self.last_forward_ms = 0.0; self.interp_step = 0; self.z_optimal = None
        self.trail_history = []; self.heatmap_artist = None
        self.custom_target_img = None; self.gif_frames = []

        self.auto_rotate_timer = QTimer(self); self.auto_rotate_timer.timeout.connect(self.auto_rotate_tick)
        self.interp_timer = QTimer(self); self.interp_timer.timeout.connect(self.interp_tick)
        self.colormaps = ['Grayscale', 'viridis', 'plasma', 'magma', 'inferno', 'cividis', 'turbo', 'coolwarm']

        self.init_ui()
        self.refresh_all_tabs()
        self.update_status_bar()

    def get_stylesheet(self):
        return """
            QMainWindow { background-color: #1e1e1e; color: #d4d4d4; }
            QScrollArea { border: none; background-color: #1e1e1e; }
            QScrollArea > QWidget > QWidget { background-color: #1e1e1e; }
            QTabWidget::pane { border: 1px solid #3c3c3c; background: #1e1e1e; }
            QTabBar::tab { background: #2d2d2d; color: #888888; padding: 8px 16px; border: 1px solid #3c3c3c; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #1e1e1e; color: #4ec9b0; border-bottom: 2px solid #4ec9b0; }
            QGroupBox { border: 1px solid #3c3c3c; border-radius: 6px; margin-top: 14px; padding: 14px 10px 10px 10px; font-weight: bold; font-size: 12px; background-color: #252525; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #4ec9b0; }
            QPushButton { background-color: #0e639c; color: white; border: none; padding: 7px 12px; border-radius: 4px; font-size: 12px; }
            QPushButton:hover { background-color: #1177bb; } QPushButton:pressed { background-color: #0a4d7a; }
            QPushButton:checked { background-color: #0a4d7a; border: 1px solid #4ec9b0; } QPushButton:disabled { background-color: #3c3c3c; color: #888888; }
            QComboBox { background-color: #2d2d2d; color: #d4d4d4; border: 1px solid #3c3c3c; padding: 5px 8px; border-radius: 4px; font-size: 12px; }
            QComboBox::drop-down { border: none; width: 20px; } QComboBox QAbstractItemView { background-color: #2d2d2d; color: #d4d4d4; selection-background-color: #0e639c; border: 1px solid #3c3c3c; }
            QTextEdit, QLineEdit { background-color: #1a1a1a; color: #d4d4d4; border: 1px solid #3c3c3c; font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 11px; border-radius: 4px; }
            QSlider::groove:horizontal { border: 1px solid #3c3c3c; height: 4px; background: #2d2d2d; border-radius: 2px; }
            QSlider::handle:horizontal { background: #0e639c; width: 16px; margin: -7px 0; border-radius: 8px; } QSlider::handle:horizontal:hover { background: #1177bb; }
            QDoubleSpinBox, QSpinBox { background-color: #2d2d2d; color: #d4d4d4; border: 1px solid #3c3c3c; padding: 4px; border-radius: 3px; font-size: 12px; }
            QCheckBox { color: #d4d4d4; font-size: 12px; spacing: 6px; } QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #3c3c3c; border-radius: 3px; background: #2d2d2d; }
            QCheckBox::indicator:checked { background: #0e639c; border: 1px solid #4ec9b0; }
            QListWidget { background-color: #1a1a1a; color: #d4d4d4; border: 1px solid #3c3c3c; font-size: 12px; border-radius: 4px; }
            QListWidget::item:selected { background-color: #0e639c; color: white; }
            QLabel { color: #d4d4d4; font-size: 12px; } QStatusBar { background-color: #007acc; color: white; font-size: 11px; } QStatusBar QLabel { color: white; }
        """

    def init_ui(self):
        self.setStyleSheet(self.get_stylesheet())
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central); main_layout.setSpacing(10); main_layout.setContentsMargins(10, 10, 10, 10)

        # ===== LEFT CONTROL PANEL =====
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setFixedWidth(350)
        left_widget = QWidget(); left_layout = QVBoxLayout(left_widget); left_layout.setSpacing(10)
        
        gpu_group = QGroupBox("⚡ Compute Backend"); gpu_layout = QVBoxLayout(gpu_group)
        self.gpu_label = QLabel(f"Backend: {'CuPy (CUDA GPU)' if USING_GPU else 'NumPy (CPU)'}\nDevice: {self.gpu_name}")
        self.gpu_label.setStyleSheet("font-family: monospace; font-size: 11px; color: #4ec9b0;"); gpu_layout.addWidget(self.gpu_label)
        self.timing_label = QLabel("Forward pass: — ms"); self.timing_label.setStyleSheet("font-size: 11px; color: #cccccc;"); gpu_layout.addWidget(self.timing_label)
        exp_row = QHBoxLayout()
        self.export_btn = QPushButton("📸 Export Tab as PNG"); self.export_btn.clicked.connect(self.export_tab); exp_row.addWidget(self.export_btn)
        gpu_layout.addLayout(exp_row); left_layout.addWidget(gpu_group)

        arch_group = QGroupBox("🏗️ Network Architecture"); arch_layout = QVBoxLayout(arch_group)
        arch_row = QHBoxLayout(); arch_row.addWidget(QLabel("Model:"))
        self.arch_combo = QComboBox(); self.arch_combo.addItems(self.architectures.keys()); self.arch_combo.currentTextChanged.connect(self.change_architecture); arch_row.addWidget(self.arch_combo); arch_layout.addLayout(arch_row)
        act_row = QHBoxLayout(); act_row.addWidget(QLabel("Activation:"))
        self.act_combo = QComboBox(); self.act_combo.addItems(['relu', 'leaky_relu', 'tanh', 'sigmoid']); self.act_combo.currentTextChanged.connect(self.change_activation); act_row.addWidget(self.act_combo); arch_layout.addLayout(act_row)
        init_row = QHBoxLayout(); init_row.addWidget(QLabel("Init:"))
        self.init_combo = QComboBox(); self.init_combo.addItems(['he', 'xavier', 'uniform', 'zeros']); self.init_combo.currentTextChanged.connect(lambda _: self.current_model.randomize_weights(self.init_combo.currentText()))
        init_row.addWidget(self.init_combo); arch_layout.addLayout(init_row)
        btn_row1 = QHBoxLayout()
        self.rand_btn = QPushButton("🎲 Randomize"); self.rand_btn.clicked.connect(self.randomize_weights); btn_row1.addWidget(self.rand_btn)
        self.save_btn = QPushButton("💾 Save"); self.save_btn.clicked.connect(self.save_weights); btn_row1.addWidget(self.save_btn)
        self.load_btn = QPushButton("📂 Load"); self.load_btn.clicked.connect(self.load_weights); btn_row1.addWidget(self.load_btn); arch_layout.addLayout(btn_row1)
        
        reset_opt_row = QHBoxLayout()
        self.reset_opt_btn = QPushButton("♻ Reset Optimizer State"); self.reset_opt_btn.clicked.connect(lambda: self.current_model.reset_optim_state())
        reset_opt_row.addWidget(self.reset_opt_btn); arch_layout.addLayout(reset_opt_row)
        
        self.arch_text = QTextEdit(); self.arch_text.setReadOnly(True); self.arch_text.setMaximumHeight(110); self.arch_text.setPlainText(self.current_model.arch_text); arch_layout.addWidget(self.arch_text)
        left_layout.addWidget(arch_group)

        latent_group = QGroupBox("🎯 Latent Vector Control"); latent_layout = QVBoxLayout(latent_group)
        self.z1_label = QLabel("Z₁ = 0.00  (range: -5.0 to 5.0)"); latent_layout.addWidget(self.z1_label)
        self.slider_z1 = QSlider(Qt.Horizontal); self.slider_z1.setRange(-50, 50); self.slider_z1.setValue(0)
        self.slider_z1.valueChanged.connect(self.update_visualization); self.slider_z1.sliderReleased.connect(self.refresh_dynamic_tabs); latent_layout.addWidget(self.slider_z1)
        self.z2_label = QLabel("Z₂ = 0.00  (range: -5.0 to 5.0)"); latent_layout.addWidget(self.z2_label)
        self.slider_z2 = QSlider(Qt.Horizontal); self.slider_z2.setRange(-50, 50); self.slider_z2.setValue(0)
        self.slider_z2.valueChanged.connect(self.update_visualization); self.slider_z2.sliderReleased.connect(self.refresh_dynamic_tabs); latent_layout.addWidget(self.slider_z2)
        
        quant_row = QHBoxLayout(); quant_row.addWidget(QLabel("Weight Bits:"))
        self.quant_slider = QSlider(Qt.Horizontal); self.quant_slider.setRange(2, 16); self.quant_slider.setValue(16)
        self.quant_slider.valueChanged.connect(self.update_quantization); quant_row.addWidget(self.quant_slider)
        latent_layout.addLayout(quant_row)

        noise_row = QHBoxLayout(); noise_row.addWidget(QLabel("Noise σ:"))
        self.noise_slider = QSlider(Qt.Horizontal); self.noise_slider.setRange(0, 100); self.noise_slider.setValue(0)
        self.noise_slider.valueChanged.connect(self.update_visualization); noise_row.addWidget(self.noise_slider)
        latent_layout.addLayout(noise_row)

        nav_row = QHBoxLayout()
        self.set_a_btn = QPushButton("◎ Set A"); self.set_a_btn.clicked.connect(lambda: self.set_click_mode('set_a')); nav_row.addWidget(self.set_a_btn)
        self.set_b_btn = QPushButton("◻ Set B"); self.set_b_btn.clicked.connect(lambda: self.set_click_mode('set_b')); nav_row.addWidget(self.set_b_btn)
        self.interp_btn = QPushButton("→ Interp"); self.interp_btn.clicked.connect(self.start_interpolation); nav_row.addWidget(self.interp_btn)
        self.auto_btn = QPushButton("▶ Auto"); self.auto_btn.setCheckable(True); self.auto_btn.clicked.connect(self.toggle_auto_rotate); nav_row.addWidget(self.auto_btn)
        latent_layout.addLayout(nav_row)
        
        ease_row = QHBoxLayout(); ease_row.addWidget(QLabel("Easing:"))
        self.ease_combo = QComboBox(); self.ease_combo.addItems(['Linear', 'Smoothstep', 'Cubic'])
        ease_row.addWidget(self.ease_combo); latent_layout.addLayout(ease_row)

        rec_row = QHBoxLayout()
        self.record_btn = QPushButton("⏺ Record GIF"); self.record_btn.setCheckable(True); self.record_btn.clicked.connect(self.toggle_recording)
        rec_row.addWidget(self.record_btn); latent_layout.addLayout(rec_row)

        heat_row = QHBoxLayout(); heat_row.addWidget(QLabel("Heatmap:"))
        self.heat_combo = QComboBox(); self.heat_combo.addItems(['None', 'Jac. Vol', 'Hess. Trace', 'Hess. Det', 'Brightness', 'Cond. Number'])
        self.heat_combo.currentTextChanged.connect(self.update_heatmap); heat_row.addWidget(self.heat_combo)
        latent_layout.addLayout(heat_row)

        samp_row = QHBoxLayout()
        self.sample_btn = QPushButton("🎲 Random"); self.sample_btn.clicked.connect(self.random_sample); samp_row.addWidget(self.sample_btn)
        self.sample_normal_btn = QPushButton("~ N(0,1)"); self.sample_normal_btn.clicked.connect(self.sample_normal); samp_row.addWidget(self.sample_normal_btn)
        self.fgsm_btn = QPushButton("⚡ FGSM"); self.fgsm_btn.clicked.connect(self.fgsm_attack); samp_row.addWidget(self.fgsm_btn)
        latent_layout.addLayout(samp_row)
        left_layout.addWidget(latent_group)

        train_group = QGroupBox("🎓 Training & Pruning"); train_layout = QVBoxLayout(train_group)
        pat_row = QHBoxLayout(); pat_row.addWidget(QLabel("Pattern:"))
        self.pattern_combo = QComboBox(); self.pattern_combo.addItems(['gaussian_blob', 'ring', 'cross', 'spiral', 'sine_wave', 'checkerboard', 'custom'])
        self.pattern_combo.currentTextChanged.connect(lambda _: self.update_visualization()); pat_row.addWidget(self.pattern_combo); train_layout.addLayout(pat_row)
        custom_row = QHBoxLayout()
        self.load_img_btn = QPushButton("🖼️ Load Custom Target"); self.load_img_btn.clicked.connect(self.load_custom_image); custom_row.addWidget(self.load_img_btn)
        train_layout.addLayout(custom_row)
        opt_row = QHBoxLayout(); opt_row.addWidget(QLabel("Optimizer:"))
        self.optim_combo = QComboBox(); self.optim_combo.addItems(['sgd', 'adam']); opt_row.addWidget(self.optim_combo); train_layout.addLayout(opt_row)
        sch_row = QHBoxLayout(); sch_row.addWidget(QLabel("Scheduler:"))
        self.sch_combo = QComboBox(); self.sch_combo.addItems(['None', 'Step', 'Cosine', 'Exponential']); sch_row.addWidget(self.sch_combo); train_layout.addLayout(sch_row)
        
        param_row = QHBoxLayout(); param_row.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 5000); self.epochs_spin.setValue(100); param_row.addWidget(self.epochs_spin)
        param_row.addWidget(QLabel("LR:"))
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(0.0001, 1.0); self.lr_spin.setSingleStep(0.01); self.lr_spin.setValue(0.05); param_row.addWidget(self.lr_spin)
        param_row.addWidget(QLabel("Batch:"))
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(16, 2048); self.batch_spin.setSingleStep(16); self.batch_spin.setValue(256); param_row.addWidget(self.batch_spin); train_layout.addLayout(param_row)

        freeze_row = QHBoxLayout(); freeze_row.addWidget(QLabel("Freeze:"))
        self.freeze_checks = []
        for i in range(4):
            chk = QCheckBox(f"L{i+1}"); freeze_row.addWidget(chk); self.freeze_checks.append(chk)
        train_layout.addLayout(freeze_row)

        reg_row = QHBoxLayout()
        reg_row.addWidget(QLabel("L1:"))
        self.l1_spin = QDoubleSpinBox(); self.l1_spin.setRange(0.0, 1.0); self.l1_spin.setSingleStep(0.001); self.l1_spin.setValue(0.0); reg_row.addWidget(self.l1_spin)
        reg_row.addWidget(QLabel("TV:"))
        self.tv_spin = QDoubleSpinBox(); self.tv_spin.setRange(0.0, 1.0); self.tv_spin.setSingleStep(0.01); self.tv_spin.setValue(0.0); reg_row.addWidget(self.tv_spin)
        train_layout.addLayout(reg_row)

        drop_row = QHBoxLayout(); drop_row.addWidget(QLabel("Dropout:"))
        self.drop_slider = QSlider(Qt.Horizontal); self.drop_slider.setRange(0, 90); self.drop_slider.setValue(0)
        drop_row.addWidget(self.drop_slider); train_layout.addLayout(drop_row)

        gn_row = QHBoxLayout(); gn_row.addWidget(QLabel("Grad Noise:"))
        self.gn_slider = QSlider(Qt.Horizontal); self.gn_slider.setRange(0, 100); self.gn_slider.setValue(0)
        gn_row.addWidget(self.gn_slider); train_layout.addLayout(gn_row)

        self.train_btn = QPushButton("▶ Train Network"); self.train_btn.clicked.connect(self.run_training); train_layout.addWidget(self.train_btn)
        prune_row = QHBoxLayout()
        self.prune_spin = QSpinBox(); self.prune_spin.setRange(1, 90); self.prune_spin.setValue(10); prune_row.addWidget(self.prune_spin)
        self.prune_btn = QPushButton("✂ Prune Weights %"); self.prune_btn.clicked.connect(self.prune_weights); prune_row.addWidget(self.prune_btn)
        train_layout.addLayout(prune_row)
        left_layout.addWidget(train_group)
        left_layout.addStretch()
        left_scroll.setWidget(left_widget)
        main_layout.addWidget(left_scroll)

        # ===== RIGHT TABBED VISUALIZATION AREA =====
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        self._build_tab_latent_space()
        self._build_tab_loss_landscape()
        self._build_tab_topology()
        self._build_tab_arithmetic()
        self._build_tab_maximization()
        self._build_tab_internals()
        self._build_tab_training()
        self._build_tab_manifold()

    # ==========================================
    # TAB BUILDERS
    # ==========================================
    def _build_tab_latent_space(self):
        tab = QWidget(); layout = QHBoxLayout(tab)
        self.fig_latent, self.ax_latent = plt.subplots(figsize=(7, 6), facecolor='#1e1e1e')
        self.canvas_latent = FigureCanvas(self.fig_latent)
        self.canvas_latent.mpl_connect('button_press_event', self.handle_plot_click)
        layout.addWidget(self.canvas_latent, stretch=1)

        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel); right_layout.setFixedWidth(300)
        out_group = QGroupBox("🖼️ Output vs Target (28×28)"); out_layout = QVBoxLayout(out_group)
        img_row = QHBoxLayout()
        out_col = QVBoxLayout(); out_col.addWidget(QLabel("Decoded:"))
        self.image_label = QLabel(); self.image_label.setAlignment(Qt.AlignCenter); self.image_label.setMinimumSize(160, 160)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid #3c3c3c; border-radius: 4px;")
        out_col.addWidget(self.image_label); img_row.addLayout(out_col)
        tar_col = QVBoxLayout(); tar_col.addWidget(QLabel("Target:"))
        self.target_label = QLabel(); self.target_label.setAlignment(Qt.AlignCenter); self.target_label.setMinimumSize(160, 160)
        self.target_label.setStyleSheet("background-color: black; border: 1px solid #3c3c3c; border-radius: 4px;")
        tar_col.addWidget(self.target_label); img_row.addLayout(tar_col)
        out_layout.addLayout(img_row)
        self.metric_label = QLabel("MSE: — | PSNR: — dB")
        self.metric_label.setStyleSheet("font-family: monospace; font-size: 13px; color: #4ec9b0; padding: 5px;")
        self.metric_label.setAlignment(Qt.AlignCenter); out_layout.addWidget(self.metric_label)
        cmap_row = QHBoxLayout(); cmap_row.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox(); self.cmap_combo.addItems(self.colormaps); self.cmap_combo.currentTextChanged.connect(lambda _: self.update_visualization()); cmap_row.addWidget(self.cmap_combo)
        out_layout.addLayout(cmap_row); right_layout.addWidget(out_group)
        bm_group = QGroupBox("📌 Concept Bookmarks"); bm_layout = QVBoxLayout(bm_group)
        bm_row = QHBoxLayout()
        self.bm_name_input = QTextEdit(); self.bm_name_input.setPlaceholderText("Bookmark name..."); self.bm_name_input.setMaximumHeight(30)
        bm_row.addWidget(self.bm_name_input)
        self.bm_add_btn = QPushButton("Add"); self.bm_add_btn.clicked.connect(self.add_bookmark); bm_row.addWidget(self.bm_add_btn)
        bm_layout.addLayout(bm_row)
        self.bookmark_list = QListWidget(); self.bookmark_list.itemDoubleClicked.connect(self.load_bookmark)
        bm_layout.addWidget(self.bookmark_list); right_layout.addWidget(bm_group)
        right_layout.addStretch(); layout.addWidget(right_panel)
        self.tabs.addTab(tab, "Latent Space")

    def _build_tab_loss_landscape(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        ctrl_row = QHBoxLayout()
        self.landscape_btn = QPushButton("📈 Compute Landscape (vs Current Target)")
        self.landscape_btn.clicked.connect(self.refresh_landscape_tab); ctrl_row.addWidget(self.landscape_btn)
        ctrl_row.addStretch(); layout.addLayout(ctrl_row)
        self.fig_land = plt.figure(figsize=(15, 5), facecolor='#1e1e1e')
        self.ax_land_2d = self.fig_land.add_subplot(1, 3, 1)
        self.ax_land_3d = self.fig_land.add_subplot(1, 3, 2, projection='3d')
        self.ax_land_1d = self.fig_land.add_subplot(1, 3, 3)
        self.canvas_land = FigureCanvas(self.fig_land)
        self.canvas_land.mpl_connect('button_press_event', self.handle_landscape_click)
        layout.addWidget(self.canvas_land); self.tabs.addTab(tab, "Loss Landscape & Gradients")

    def _build_tab_topology(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Hessian Objective:"))
        self.obj_combo = QComboBox(); self.obj_combo.addItems(["Total Output Brightness (L2)", "Center Pixel (14,14)"])
        self.obj_combo.currentTextChanged.connect(self.refresh_topology_tab); ctrl_row.addWidget(self.obj_combo)
        ctrl_row.addStretch(); layout.addLayout(ctrl_row)
        self.topology_text = QTextEdit(); self.topology_text.setReadOnly(True); self.topology_text.setMaximumHeight(160)
        layout.addWidget(self.topology_text)
        self.fig_topo, self.axes_topo = plt.subplots(1, 3, figsize=(12, 4), facecolor='#1e1e1e')
        self.canvas_topo = FigureCanvas(self.fig_topo); layout.addWidget(self.canvas_topo)
        self.tabs.addTab(tab, "Latent Topology & Saliency")

    def _build_tab_arithmetic(self):
        tab = QWidget(); layout = QHBoxLayout(tab)
        ctrl_widget = QWidget(); ctrl_layout = QVBoxLayout(ctrl_widget)
        a_group = QGroupBox("Vector A"); a_lay = QVBoxLayout(a_group)
        self.z1a_spin = QDoubleSpinBox(); self.z1a_spin.setRange(-5, 5); self.z1a_spin.setSingleStep(0.1); self.z1a_spin.setValue(2.0)
        self.z2a_spin = QDoubleSpinBox(); self.z2a_spin.setRange(-5, 5); self.z2a_spin.setSingleStep(0.1); self.z2a_spin.setValue(1.0)
        a_lay.addWidget(QLabel("Z₁:")); a_lay.addWidget(self.z1a_spin); a_lay.addWidget(QLabel("Z₂:")); a_lay.addWidget(self.z2a_spin)
        ctrl_layout.addWidget(a_group)
        b_group = QGroupBox("Vector B"); b_lay = QVBoxLayout(b_group)
        self.z1b_spin = QDoubleSpinBox(); self.z1b_spin.setRange(-5, 5); self.z1b_spin.setSingleStep(0.1); self.z1b_spin.setValue(-2.0)
        self.z2b_spin = QDoubleSpinBox(); self.z2b_spin.setRange(-5, 5); self.z2b_spin.setSingleStep(0.1); self.z2b_spin.setValue(-1.0)
        b_lay.addWidget(QLabel("Z₁:")); b_lay.addWidget(self.z1b_spin); b_lay.addWidget(QLabel("Z₂:")); b_lay.addWidget(self.z2b_spin)
        ctrl_layout.addWidget(b_group)
        op_group = QGroupBox("Operation"); op_lay = QVBoxLayout(op_group)
        self.op_combo = QComboBox()
        self.op_combo.addItems(["(A + B) / 2", "A + B", "A - B", "B - A", "0.7A + 0.3B", "0.3A + 0.7B", "A * B (element-wise)"])
        self.op_combo.currentTextChanged.connect(self.refresh_arithmetic_tab); op_lay.addWidget(self.op_combo)
        ctrl_layout.addWidget(op_group); ctrl_layout.addStretch()
        img_widget = QWidget(); img_layout = QHBoxLayout(img_widget)
        def make_img_box(title):
            col = QVBoxLayout()
            lbl_title = QLabel(f"<b>{title}</b>"); lbl_title.setAlignment(Qt.AlignCenter)
            lbl = QLabel(); lbl.setAlignment(Qt.AlignCenter); lbl.setStyleSheet("background-color: black; border: 1px solid #3c3c3c;")
            lbl.setMinimumSize(200, 200); col.addWidget(lbl_title); col.addWidget(lbl)
            return col, lbl
        col_a, self.img_a = make_img_box("Vector A"); col_b, self.img_b = make_img_box("Vector B"); col_c, self.img_c = make_img_box("Result")
        img_layout.addLayout(col_a); img_layout.addLayout(col_b); img_layout.addLayout(col_c)
        layout.addWidget(ctrl_widget); layout.addWidget(img_widget, stretch=1); self.tabs.addTab(tab, "Latent Arithmetic")

    def _build_tab_maximization(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Layer:"))
        self.max_layer_combo = QComboBox(); self.update_max_layers()
        self.max_layer_combo.currentIndexChanged.connect(self.update_max_neuron_spin); ctrl_row.addWidget(self.max_layer_combo)
        ctrl_row.addWidget(QLabel("Neuron:"))
        self.max_neuron_spin = QSpinBox(); self.max_neuron_spin.setMinimum(0); self.update_max_neuron_spin()
        ctrl_row.addWidget(self.max_neuron_spin)
        ctrl_row.addWidget(QLabel("TV Weight:"))
        self.tv_spin = QDoubleSpinBox(); self.tv_spin.setRange(0.0, 10.0); self.tv_spin.setSingleStep(0.1); self.tv_spin.setValue(1.0)
        ctrl_row.addWidget(self.tv_spin)
        self.max_run_btn = QPushButton("▶ Run Gradient Ascent"); self.max_run_btn.clicked.connect(self.run_maximization)
        ctrl_row.addWidget(self.max_run_btn); ctrl_row.addStretch(); layout.addLayout(ctrl_row)
        self.max_status = QLabel("Status: Idle"); self.max_status.setStyleSheet("font-family: monospace; color: #4ec9b0;")
        layout.addWidget(self.max_status)
        img_row = QHBoxLayout()
        self.max_img_label = QLabel(); self.max_img_label.setAlignment(Qt.AlignCenter)
        self.max_img_label.setStyleSheet("background-color: black; border: 1px solid #3c3c3c;")
        self.max_img_label.setMinimumSize(300, 300); img_row.addWidget(self.max_img_label, stretch=1)
        self.max_apply_btn = QPushButton("🎯 Set Main Z\n to Optimal"); self.max_apply_btn.clicked.connect(self.apply_maximization)
        img_row.addWidget(self.max_apply_btn); layout.addLayout(img_row); self.tabs.addTab(tab, "Feature Visualization")

    def _build_tab_internals(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        self.fig_int, self.axes_int = plt.subplots(2, 3, figsize=(14, 7), facecolor='#1e1e1e')
        self.canvas_int = FigureCanvas(self.fig_int); layout.addWidget(self.canvas_int); self.tabs.addTab(tab, "Network Internals & SVD")

    def _build_tab_training(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        self.fig_train, self.axes_train = plt.subplots(2, 2, figsize=(12, 8), facecolor='#1e1e1e')
        self.canvas_train = FigureCanvas(self.fig_train); layout.addWidget(self.canvas_train); self.tabs.addTab(tab, "Training & Gradient Flow")

    def _build_tab_manifold(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Grid Min:"))
        self.grid_min_spin = QDoubleSpinBox(); self.grid_min_spin.setRange(-5, 5); self.grid_min_spin.setValue(-5.0); ctrl_row.addWidget(self.grid_min_spin)
        ctrl_row.addWidget(QLabel("Grid Max:"))
        self.grid_max_spin = QDoubleSpinBox(); self.grid_max_spin.setRange(-5, 5); self.grid_max_spin.setValue(5.0); ctrl_row.addWidget(self.grid_max_spin)
        self.polar_chk = QCheckBox("Polar Grid"); ctrl_row.addWidget(self.polar_chk)
        gen_btn = QPushButton("🔲 Generate Manifold Grid & FFT"); gen_btn.clicked.connect(self.generate_manifold_grid); ctrl_row.addWidget(gen_btn)
        ctrl_row.addStretch(); layout.addLayout(ctrl_row)
        self.fig_man, self.axes_man = plt.subplots(1, 2, figsize=(12, 5), facecolor='#1e1e1e')
        self.canvas_man = FigureCanvas(self.fig_man); layout.addWidget(self.canvas_man); self.tabs.addTab(tab, "Manifold Grid & Spectral FFT")

    # ==========================================
    # STATE & REFRESH LOGIC
    # ==========================================
    def toggle_recording(self, checked):
        if checked:
            self.gif_frames = []
            self.record_btn.setText("⏹ Stop & Save GIF")
            self.statusBar().showMessage("Recording started...", 2000)
        else:
            self.record_btn.setText("⏺ Record GIF")
            if self.gif_frames:
                path, _ = QFileDialog.getSaveFileName(self, "Save GIF", "animation.gif", "GIF Image (*.gif)")
                if path:
                    QApplication.setOverrideCursor(Qt.WaitCursor)
                    imageio.mimsave(path, self.gif_frames, duration=0.05)
                    QApplication.restoreOverrideCursor()
                    self.statusBar().showMessage(f"GIF saved to {path}", 5000)
            self.gif_frames = []

    def export_tab(self):
        idx = self.tabs.currentIndex()
        canvases = [self.canvas_latent, self.canvas_land, self.canvas_topo, None, None, self.canvas_int, self.canvas_train, self.canvas_man]
        if canvases[idx]:
            path, _ = QFileDialog.getSaveFileName(self, "Save Figure", f"tab_{idx}.png", "PNG Image (*.png)")
            if path: canvases[idx].figure.savefig(path, dpi=300, facecolor='#1e1e1e')

    def load_custom_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Target Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            qimg = QImage(path).scaled(28, 28, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            qimg = qimg.convertToFormat(QImage.Format_Grayscale8)
            ptr = qimg.constBits(); ptr.setsize(qimg.sizeInBytes())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(28, 28).astype(np.float32) / 255.0
            self.custom_target_img = cp.asarray(arr).reshape(1, 784)
            self.pattern_combo.setCurrentText("custom"); self.update_visualization()

    def update_max_layers(self):
        self.max_layer_combo.clear()
        for i in range(len(self.current_model.layers) - 1): self.max_layer_combo.addItem(f"Layer {i+1}")

    def update_max_neuron_spin(self, _=None):
        idx = self.max_layer_combo.currentIndex()
        if idx >= 0: self.max_neuron_spin.setMaximum(self.current_model.layer_shapes[idx][1] - 1)

    def update_quantization(self, bits):
        self.current_model.quant_bits = bits
        self.update_visualization()
        self.refresh_dynamic_tabs()

    def refresh_all_tabs(self):
        self.update_latent_plot(); self.update_visualization()
        self.refresh_dynamic_tabs(); self.refresh_arithmetic_tab()

    def refresh_dynamic_tabs(self):
        self.refresh_topology_tab(); self.refresh_internals_tab(); self.refresh_training_tab()

    def update_latent_plot(self):
        self.ax_latent.clear()
        self.ax_latent.scatter(self.mock_latents[:, 0], self.mock_latents[:, 1], c='#4ec9b0', alpha=0.2, s=12, edgecolors='none')
        self.heatmap_artist = None
        self.ax_latent.set_title("Latent Space (2D) — Click to Navigate", color='#cccccc')
        self.ax_latent.set_xlim(-5, 5); self.ax_latent.set_ylim(-5, 5)
        self.ax_latent.set_facecolor('#2d2d2d'); style_mpl_axes(self.ax_latent)
        self.ax_latent.grid(True, alpha=0.15, color='#cccccc')
        self.current_point = self.ax_latent.scatter([0], [0], c='red', s=140, marker='x', linewidths=2.5, zorder=5)
        self.trail_line, = self.ax_latent.plot([], [], '-', color='red', alpha=0.4, linewidth=1.5, zorder=4)
        self.opt_path_line, = self.ax_latent.plot([], [], '-', color='cyan', alpha=0.8, linewidth=2, zorder=6)
        self.canvas_latent.draw(); self.update_heatmap()

    def update_visualization(self):
        z1 = self.slider_z1.value() / 10.0; z2 = self.slider_z2.value() / 10.0
        noise = self.noise_slider.value() / 100.0
        z_noisy = z1 + np.random.randn() * noise if noise > 0 else z1
        z_noisy_y = z2 + np.random.randn() * noise if noise > 0 else z2
        self.z1_label.setText(f"Z₁ = {z1:+.2f}  (Noise: {noise:.2f})"); self.z2_label.setText(f"Z₂ = {z2:+.2f}  (Noise: {noise:.2f})")
        self.trail_history.append((z1, z2))
        if len(self.trail_history) > 30: self.trail_history.pop(0)
        if hasattr(self, 'current_point') and self.current_point: self.current_point.remove()
        if hasattr(self, 'trail_line') and self.trail_line: 
            self.trail_line.set_data([p[0] for p in self.trail_history], [p[1] for p in self.trail_history])
        self.current_point = self.ax_latent.scatter([z1], [z2], c='red', s=140, marker='x', linewidths=2.5, zorder=5)
        self.canvas_latent.draw_idle()

        cmap = self.cmap_combo.currentText()
        z = cp.array([[z_noisy, z_noisy_y]], dtype=cp.float32)
        if USING_GPU: cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter(); img_cp = self.current_model.forward(z)
        if USING_GPU: cp.cuda.Stream.null.synchronize()
        self.last_forward_ms = (time.perf_counter() - t0) * 1000

        img = _to_np(img_cp.squeeze()).reshape(28, 28)
        self.image_label.setPixmap(array_to_pixmap(img, cmap).scaled(160, 160, Qt.KeepAspectRatio))
        
        if self.record_btn.isChecked():
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            if cmap != 'Grayscale':
                colored = plt.get_cmap(cmap)(img)[:, :, :3]
                pil_img = Image.fromarray((colored * 255).astype(np.uint8))
            self.gif_frames.append(np.array(pil_img))
        
        target_z = cp.array([[z1, z2]], dtype=cp.float32)
        target = _to_np(generate_target_batch(target_z, self.pattern_combo.currentText(), self.custom_target_img).squeeze()).reshape(28, 28)
        self.target_label.setPixmap(array_to_pixmap(target, cmap).scaled(160, 160, Qt.KeepAspectRatio))
        mse = np.mean((img - target)**2); psnr = calculate_psnr(img, target)
        self.metric_label.setText(f"MSE: {mse:.4f} | PSNR: {psnr:.2f} dB")
        self.update_status_bar()

    def update_heatmap(self, _=None):
        if self.heatmap_artist: self.heatmap_artist.remove(); self.heatmap_artist = None
        heat_type = self.heat_combo.currentText()
        if heat_type == 'None': self.canvas_latent.draw(); return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        n = 25; grid_x = cp.linspace(-5, 5, n, dtype=cp.float32); grid_y = cp.linspace(-5, 5, n, dtype=cp.float32)
        xx, yy = cp.meshgrid(grid_x, grid_y); z_batch = cp.stack([xx.ravel(), yy.ravel()], axis=1)
        if heat_type == 'Brightness':
            out = self.current_model.forward(z_batch); data = _to_np(cp.mean(out, axis=1)).reshape(n, n); cmap = 'viridis'
        elif heat_type in ['Jac. Vol', 'Cond. Number']:
            eps = 1e-4
            z_p1 = z_batch.copy(); z_p1[:, 0] += eps; z_m1 = z_batch.copy(); z_m1[:, 0] -= eps
            grad_z1 = (self.current_model.forward(z_p1) - self.current_model.forward(z_m1)) / (2 * eps)
            z_p2 = z_batch.copy(); z_p2[:, 1] += eps; z_m2 = z_batch.copy(); z_m2[:, 1] -= eps
            grad_z2 = (self.current_model.forward(z_p2) - self.current_model.forward(z_m2)) / (2 * eps)
            s1_sq = cp.sum(grad_z1**2, axis=1); s2_sq = cp.sum(grad_z2**2, axis=1); s12 = cp.sum(grad_z1*grad_z2, axis=1)
            trace = s1_sq + s2_sq; det = s1_sq*s2_sq - s12**2
            sqrt_disc = cp.sqrt(cp.maximum((trace/2)**2 - det, 1e-12))
            sigma1 = cp.sqrt(trace/2 + sqrt_disc); sigma2 = cp.sqrt(trace/2 - sqrt_disc)
            if heat_type == 'Jac. Vol': data = _to_np(sigma1**2 + sigma2**2).reshape(n, n); cmap = 'inferno'
            else: data = _to_np(sigma1 / (sigma2 + 1e-8)).reshape(n, n); cmap = 'plasma'
        elif heat_type in ['Hess. Trace', 'Hess. Det']:
            h = 0.2
            def L(z): return cp.mean(self.current_model.forward(z)**2, axis=1)
            L_b = L(z_batch)
            z_xp = z_batch.copy(); z_xp[:,0] += h; z_xm = z_batch.copy(); z_xm[:,0] -= h
            L_xx = (L(z_xp) - 2*L_b + L(z_xm)) / h**2
            z_yp = z_batch.copy(); z_yp[:,1] += h; z_ym = z_batch.copy(); z_ym[:,1] -= h
            L_yy = (L(z_yp) - 2*L_b + L(z_ym)) / h**2
            z_xpyp = z_batch.copy(); z_xpyp[:,0] += h; z_xpyp[:,1] += h; z_xpym = z_batch.copy(); z_xpym[:,0] += h; z_xpym[:,1] -= h
            z_xmyp = z_batch.copy(); z_xmyp[:,0] -= h; z_xmyp[:,1] += h; z_xmym = z_batch.copy(); z_xmym[:,0] -= h; z_xmym[:,1] -= h
            L_xy = (L(z_xpyp) - L(z_xpym) - L(z_xmyp) + L(z_xmym)) / (4*h**2)
            if heat_type == 'Hess. Trace': data = _to_np(L_xx + L_yy).reshape(n, n)
            else: data = _to_np(L_xx * L_yy - L_xy**2).reshape(n, n)
            cmap = 'coolwarm'
        self.heatmap_artist = self.ax_latent.imshow(data, extent=[-5, 5, -5, 5], origin='lower', cmap=cmap, alpha=0.6, zorder=0)
        self.canvas_latent.draw(); QApplication.restoreOverrideCursor()

    def add_bookmark(self):
        z1 = self.slider_z1.value() / 10.0; z2 = self.slider_z2.value() / 10.0
        name = self.bm_name_input.toPlainText().strip()
        if not name: name = f"Z=[{z1:.1f},{z2:.1f}]"
        item = QListWidgetItem(f"{name}  (Z=[{z1:.2f}, {z2:.2f}])"); item.setData(Qt.UserRole, (z1, z2))
        self.bookmark_list.addItem(item); self.bm_name_input.clear()

    def load_bookmark(self, item):
        z1, z2 = item.data(Qt.UserRole)
        self.slider_z1.setValue(int(z1 * 10)); self.slider_z2.setValue(int(z2 * 10))

    def refresh_landscape_tab(self):
        QApplication.setOverrideCursor(Qt.WaitCursor); QApplication.processEvents()
        z1, z2 = self.slider_z1.value() / 10.0, self.slider_z2.value() / 10.0
        target_img = generate_target_batch(cp.array([[z1, z2]], dtype=cp.float32), self.pattern_combo.currentText(), self.custom_target_img)
        n = 40; grid_x = cp.linspace(-5, 5, n, dtype=cp.float32); grid_y = cp.linspace(-5, 5, n, dtype=cp.float32)
        xx, yy = cp.meshgrid(grid_x, grid_y); z_batch = cp.stack([xx.ravel(), yy.ravel()], axis=1)
        out = self.current_model.forward(z_batch)
        mse = cp.mean((out - target_img)**2, axis=1).reshape(n, n); mse_np = _to_np(mse)
        
        self.ax_land_2d.clear()
        im = self.ax_land_2d.imshow(np.log(mse_np + 1e-6), extent=[-5, 5, -5, 5], origin='lower', cmap='inferno')
        def L(z): return float(cp.mean((self.current_model.forward(z) - target_img)**2))
        hh = 0.5
        X_q, Y_q = np.meshgrid(np.linspace(-4, 4, 15), np.linspace(-4, 4, 15))
        U, V = np.zeros_like(X_q), np.zeros_like(Y_q)
        for i in range(X_q.shape[0]):
            for j in range(X_q.shape[1]):
                gx = (L(cp.array([[X_q[i,j]+hh, Y_q[i,j]]], dtype=cp.float32)) - L(cp.array([[X_q[i,j]-hh, Y_q[i,j]]], dtype=cp.float32))) / (2*hh)
                gy = (L(cp.array([[X_q[i,j], Y_q[i,j]+hh]], dtype=cp.float32)) - L(cp.array([[X_q[i,j], Y_q[i,j]-hh]], dtype=cp.float32))) / (2*hh)
                U[i,j], V[i,j] = -gx, -gy
        self.ax_land_2d.quiver(X_q, Y_q, U, V, color='cyan', alpha=0.7)
        self.ax_land_2d.plot(z1, z2, 'rx', markersize=10, markeredgewidth=2)
        self.ax_land_2d.set_title("Log(MSE) & Gradient Field (-∇L) [Click to Teleport]", color='#cccccc'); style_mpl_axes(self.ax_land_2d)
        if hasattr(self, 'cbar_land'): self.cbar_land.remove()
        self.cbar_land = self.fig_land.colorbar(im, ax=self.ax_land_2d, fraction=0.046, pad=0.04); self.cbar_land.ax.tick_params(colors='#cccccc')
        
        self.ax_land_3d.clear()
        X, Y = np.meshgrid(_to_np(grid_x), _to_np(grid_y))
        self.ax_land_3d.plot_surface(X, Y, mse_np, cmap='inferno', alpha=0.8)
        self.ax_land_3d.set_title("3D Loss Surface", color='#cccccc'); self.ax_land_3d.tick_params(colors='#cccccc')
        self.ax_land_3d.set_xlabel('Z1', color='#cccccc'); self.ax_land_3d.set_ylabel('Z2', color='#cccccc'); self.ax_land_3d.set_zlabel('MSE', color='#cccccc')
        
        self.ax_land_1d.clear(); style_mpl_axes(self.ax_land_1d)
        self.ax_land_1d.set_title("A→B Path Elevation Profile", color='#cccccc')
        self.ax_land_1d.set_xlabel("Interpolation Step (t)", color='#cccccc'); self.ax_land_1d.set_ylabel("MSE", color='#cccccc')
        if self.point_a_pos is not None and self.point_b_pos is not None:
            path_zs = [self.point_a_pos + (self.point_b_pos - self.point_a_pos) * t for t in np.linspace(0, 1, 50)]
            path_losses = [L(cp.array([z], dtype=cp.float32)) for z in path_zs]
            self.ax_land_2d.plot([z[0] for z in path_zs], [z[1] for z in path_zs], 'r--', linewidth=2)
            self.ax_land_1d.plot(path_losses, color='red', linewidth=2)
        
        self.fig_land.tight_layout(); self.canvas_land.draw(); QApplication.restoreOverrideCursor()

    def handle_landscape_click(self, event):
        if event.inaxes == self.ax_land_2d:
            x, y = event.xdata, event.ydata
            self.slider_z1.setValue(int(x * 10)); self.slider_z2.setValue(int(y * 10))
            self.tabs.setCurrentIndex(0)

    def refresh_arithmetic_tab(self):
        a = np.array([self.z1a_spin.value(), self.z2a_spin.value()]); b = np.array([self.z1b_spin.value(), self.z2b_spin.value()])
        op = self.op_combo.currentText()
        if op == "(A + B) / 2":   c = (a + b) / 2
        elif op == "A + B":        c = a + b
        elif op == "A - B":        c = a - b
        elif op == "B - A":        c = b - a
        elif op == "0.7A + 0.3B":  c = 0.7 * a + 0.3 * b
        elif op == "0.3A + 0.7B":  c = 0.3 * a + 0.7 * b
        elif op == "A * B (element-wise)": c = a * b
        c = np.clip(c, -5, 5)
        def decode(z):
            out = _to_np(self.current_model.forward(cp.array([z], dtype=cp.float32)).squeeze()).reshape(28, 28)
            return array_to_pixmap(out, self.cmap_combo.currentText()).scaled(200, 200, Qt.KeepAspectRatio)
        self.img_a.setPixmap(decode(a)); self.img_b.setPixmap(decode(b)); self.img_c.setPixmap(decode(c))

    def run_maximization(self):
        layer_idx = self.max_layer_combo.currentIndex() + 1; neuron_idx = self.max_neuron_spin.value()
        tv_weight = self.tv_spin.value()
        z = cp.random.randn(1, 2).astype(cp.float32) * 0.5; lr = 0.05; z_history = []
        self.max_run_btn.setEnabled(False)
        def total_loss(z_val):
            out = self.current_model.forward(z_val).reshape(28, 28)
            act = self.current_model.cache['activations'][layer_idx][0, neuron_idx]
            tv = cp.sum((out[1:, :] - out[:-1, :])**2) + cp.sum((out[:, 1:] - out[:, :-1])**2)
            return act - tv_weight * float(tv)
        for step in range(100):
            eps = 1e-3
            z_p1 = z.copy(); z_p1[0,0] += eps; z_m1 = z.copy(); z_m1[0,0] -= eps
            l_p1 = total_loss(z_p1); l_m1 = total_loss(z_m1); grad_z1 = (l_p1 - l_m1) / (2 * eps)
            z_p2 = z.copy(); z_p2[0,1] += eps; z_m2 = z.copy(); z_m2[0,1] -= eps
            l_p2 = total_loss(z_p2); l_m2 = total_loss(z_m2); grad_z2 = (l_p2 - l_m2) / (2 * eps)
            z[0,0] += lr * grad_z1; z[0,1] += lr * grad_z2; z = cp.clip(z, -5.0, 5.0)
            z_history.append(_to_np(z[0]))
            if step % 10 == 0:
                self.max_status.setText(f"Step {step}/100 | Loss: {float(l_p1):.4f} | Z: [{float(z[0,0]):.2f}, {float(z[0,1]):.2f}]")
                self.opt_path_line.set_data([p[0] for p in z_history], [p[1] for p in z_history])
                self.canvas_latent.draw_idle(); QApplication.processEvents()
        self.z_optimal = z
        self.max_status.setText(f"Complete! Z = [{float(z[0,0]):.3f}, {float(z[0,1]):.3f}]")
        out = _to_np(self.current_model.forward(z).squeeze()).reshape(28, 28)
        self.max_img_label.setPixmap(array_to_pixmap(out, 'viridis').scaled(300, 300, Qt.KeepAspectRatio))
        self.max_run_btn.setEnabled(True)

    def apply_maximization(self):
        if self.z_optimal is None: return
        z1, z2 = _to_np(self.z_optimal[0])
        self.slider_z1.setValue(int(z1 * 10)); self.slider_z2.setValue(int(z2 * 10))
        self.tabs.setCurrentIndex(0)

    def refresh_topology_tab(self):
        z1 = self.slider_z1.value() / 10.0; z2 = self.slider_z2.value() / 10.0; z = cp.array([[z1, z2]], dtype=cp.float32)
        h = 1e-4
        f_dx = (self.current_model.forward(z + cp.array([[h, 0]], dtype=cp.float32)) - self.current_model.forward(z - cp.array([[h, 0]], dtype=cp.float32))).squeeze() / (2*h)
        f_dy = (self.current_model.forward(z + cp.array([[0, h]], dtype=cp.float32)) - self.current_model.forward(z - cp.array([[0, h]], dtype=cp.float32))).squeeze() / (2*h)
        J = cp.stack([f_dx, f_dy], axis=1); M = J.T @ J
        s_vals = cp.sqrt(cp.maximum(cp.linalg.eigvalsh(M), 1e-12))
        def L(z1, z2):
            out = self.current_model.forward(cp.array([[z1, z2]], dtype=cp.float32)).squeeze()
            if self.obj_combo.currentText().startswith("Total"): return float(cp.sum(out**2))
            return float(out[14*28 + 14]**2)
        hh = 1e-3
        Lxx = (L(z1+hh, z2) - 2*L(z1, z2) + L(z1-hh, z2)) / (hh**2)
        Lyy = (L(z1, z2+hh) - 2*L(z1, z2) + L(z1, z2-hh)) / (hh**2)
        Lxy = (L(z1+hh, z2+hh) - L(z1+hh, z2-hh) - L(z1-hh, z2+hh) + L(z1-hhh, z2-hh)) / (4*hh**2)
        H = np.array([[Lxx, Lxy], [Lxy, Lyy]]); h_eigvals, h_eigvecs = np.linalg.eigh(H)
        grad_x = (L(z1+hh, z2) - L(z1-hh, z2)) / (2*hh); grad_y = (L(z1, z2+hh) - L(z1, z2-hh)) / (2*hh)
        nature = "Local Minimum" if (h_eigvals[0] > 0 and h_eigvals[1] > 0) else "Local Maximum" if (h_eigvals[0] < 0 and h_eigvals[1] < 0) else "Saddle Point"
        txt = f"Mathematical Analysis at Z = [{z1:.3f}, {z2:.3f}]\n" + "═"*50 + "\n"
        txt += "1. JACOBIAN MATRIX (First-Order Derivatives)\n"
        txt += f"   Pullback Metric M = JᵀJ:\n   ┌ {M[0,0]:>8.4f}  {M[0,1]:>8.4f} ┐\n   └ {M[1,0]:>8.4f}  {M[1,1]:>8.4f} ┘\n"
        txt += f"   Singular Values (σ₁, σ₂): {s_vals[1]:.4f}, {s_vals[0]:.4f}\n\n"
        txt += "2. HESSIAN MATRIX (Second-Order Derivatives)\n"
        txt += f"   ┌ {Lxx:>8.4f}  {Lxy:>8.4f} ┐\n   └ {Lxy:>8.4f}  {Lyy:>8.4f} ┘\n"
        txt += f"   Eigenvalues (λ₁, λ₂): {h_eigvals[1]:.4f}, {h_eigvals[0]:.4f}\n"
        txt += f"   Gradient ∇L: [{grad_x:.4f}, {grad_y:.4f}]\n\n"
        txt += f"3. CRITICAL POINT CLASSIFICATION: {nature}\n"
        self.topology_text.setText(txt)
        for ax in self.axes_topo: ax.clear(); ax.set_facecolor('#2d2d2d'); style_mpl_axes(ax)
        self.axes_topo[0].set_title("Hessian Quadratic Form & Gradient", color='#cccccc')
        X, Y = np.meshgrid(np.linspace(z1-2, z1+2, 40), np.linspace(z2-2, z2+2, 40))
        Z_quad = Lxx*(X-z1)**2 + 2*Lxy*(X-z1)*(Y-z2) + Lyy*(Y-z2)**2
        self.axes_topo[0].contourf(X, Y, Z_quad, levels=15, cmap='coolwarm', alpha=0.6)
        self.axes_topo[0].plot(z1, z2, 'ro', markersize=8)
        self.axes_topo[0].quiver(z1, z2, grad_x, grad_y, color='yellow', scale=5, label='Gradient')
        v1 = h_eigvecs[:, 1]; v2 = h_eigvecs[:, 0]
        self.axes_topo[0].quiver(z1, z2, v1[0], v1[1], color='cyan', scale=5, label=f'Eigvec λ={h_eigvals[1]:.2f}')
        self.axes_topo[0].quiver(z1, z2, v2[0], v2[1], color='magenta', scale=5, label=f'Eigvec λ={h_eigvals[0]:.2f}')
        self.axes_topo[0].legend(loc='upper right', fontsize=7, facecolor='#2d2d2d', edgecolor='#555555', labelcolor='#cccccc')
        self.axes_topo[1].imshow(_to_np(f_dx).reshape(28, 28), cmap='magma'); self.axes_topo[1].set_title("∂Output/∂Z₁", color='#cccccc')
        self.axes_topo[2].imshow(_to_np(f_dy).reshape(28, 28), cmap='magma'); self.axes_topo[2].set_title("∂Output/∂Z₂", color='#cccccc')
        for ax in self.axes_topo[1:]: ax.set_xticks([]); ax.set_yticks([])
        self.fig_topo.tight_layout(); self.canvas_topo.draw()

    def refresh_internals_tab(self):
        z1 = self.slider_z1.value() / 10.0; z2 = self.slider_z2.value() / 10.0
        self.current_model.forward(cp.array([[z1, z2]], dtype=cp.float32), cache=True)
        for ax in self.axes_int.flatten(): ax.clear(); ax.set_facecolor('#2d2d2d'); style_mpl_axes(ax); ax.set_visible(True)
        n_layers = len(self.current_model.layers); cols = min(n_layers, 3)
        for i in range(cols):
            layer = self.current_model.layers[i]; W = _to_np(layer['W'])
            vmax = np.abs(W).max() + 1e-8
            self.axes_int[0, i].imshow(W, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            self.axes_int[0, i].set_title(f"Layer {i+1} Weights ({W.shape[0]}×{W.shape[1]})", color='#cccccc', fontsize=9)
            try:
                U, s, Vh = np.linalg.svd(W, full_matrices=False)
                if W.shape[0] == 784:
                    eigen_img = U[:, 0].reshape(28, 28)
                    self.axes_int[1, i].imshow(eigen_img, cmap='magma')
                    self.axes_int[1, i].set_title(f"Layer {i+1} 1st Eigenimage", color='#cccccc', fontsize=9)
                else:
                    self.axes_int[1, i].plot(s, color='#f44747', marker='.', markersize=3)
                    self.axes_int[1, i].set_title(f"Layer {i+1} Singular Values", color='#cccccc', fontsize=9)
            except Exception: pass
        for i in range(cols, 3): self.axes_int[0, i].set_visible(False); self.axes_int[1, i].set_visible(False)
        self.fig_int.tight_layout(); self.canvas_int.draw()

    def refresh_training_tab(self):
        for ax in self.axes_train.flatten(): ax.clear(); ax.set_facecolor('#2d2d2d'); style_mpl_axes(ax)
        
        ax_loss = self.axes_train[0, 0]
        if self.loss_history:
            ax_loss.plot(self.loss_history, color='#4ec9b0', linewidth=1.5, label='Train')
            if self.val_loss_history:
                ax_loss.plot(self.val_loss_history, color='#f44747', linewidth=1.5, label='Val')
            ax_loss.set_yscale('log'); ax_loss.set_title(f"Train/Val Loss ({len(self.loss_history)} steps)", color='#cccccc')
            ax_loss.legend(facecolor='#2d2d2d', edgecolor='#555555', labelcolor='#cccccc', fontsize=7)

        ax_grad = self.axes_train[0, 1]
        if self.grad_norm_history:
            gn = np.array(self.grad_norm_history).T
            im = ax_grad.imshow(np.log10(gn + 1e-8), aspect='auto', cmap='magma', interpolation='nearest')
            ax_grad.set_title("Gradient Norm Heatmap (log10)", color='#cccccc')
            ax_grad.set_xlabel("Epoch", color='#cccccc'); ax_grad.set_ylabel("Layer", color='#cccccc')
            if hasattr(self, 'cbar_train'): self.cbar_train.remove()
            self.cbar_train = self.fig_train.colorbar(im, ax=ax_grad, fraction=0.046, pad=0.04); self.cbar_train.ax.tick_params(colors='#cccccc')
            
        ax_spec = self.axes_train[1, 0]
        if self.spec_norm_history:
            sn = np.array(self.spec_norm_history)
            for i in range(sn.shape[1]): ax_spec.plot(sn[:, i], label=f'Layer {i+1}')
            ax_spec.set_title("Spectral Norm (σ_max) over Epochs", color='#cccccc')
            ax_spec.legend(facecolor='#2d2d2d', edgecolor='#555555', labelcolor='#cccccc', fontsize=7)

        ax_dead = self.axes_train[1, 1]
        z_batch = cp.random.randn(500, 2).astype(cp.float32) * 2.0
        self.current_model.forward(z_batch, cache=True)
        dead_pcts = []
        for i in range(len(self.current_model.layers)):
            act = self.current_model.cache['activations'][i+1]
            dead_pct = float((act == 0).all(axis=0).mean()) * 100 if self.current_model.activation in ['relu', 'leaky_relu'] else 0
            dead_pcts.append(dead_pct)
        ax_dead.bar(range(1, len(self.current_model.layers)+1), dead_pcts, color='#569cd6')
        ax_dead.set_title("Dead ReLU Percentage per Layer", color='#cccccc')
        ax_dead.set_xlabel("Layer", color='#cccccc'); ax_dead.set_ylabel("% Dead", color='#cccccc')
        
        self.fig_train.tight_layout(); self.canvas_train.draw()

    # ==========================================
    # EVENTS & ACTIONS
    # ==========================================
    def change_architecture(self, text):
        self.current_model = self.architectures[text](self.current_activation)
        self.arch_text.setPlainText(self.current_model.arch_text)
        self.loss_history.clear(); self.val_loss_history.clear()
        self.grad_norm_history.clear(); self.spec_norm_history.clear()
        self.update_max_layers(); self.refresh_all_tabs()

    def change_activation(self, text):
        self.current_activation = text
        self.current_model = list(self.architectures.values())[self.arch_combo.currentIndex()](self.current_activation)
        self.arch_text.setPlainText(self.current_model.arch_text)
        self.loss_history.clear(); self.val_loss_history.clear()
        self.grad_norm_history.clear(); self.spec_norm_history.clear()
        self.refresh_all_tabs()

    def randomize_weights(self):
        self.current_model.randomize_weights(self.init_combo.currentText())
        self.loss_history.clear(); self.val_loss_history.clear()
        self.grad_norm_history.clear(); self.spec_norm_history.clear()
        self.refresh_all_tabs()

    def save_weights(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Weights", "model_weights.npz", "NumPy Archives (*.npz)")
        if path: self.current_model.save_weights(path); self.statusBar().showMessage(f"Saved to {path}", 3000)

    def load_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Weights", "", "NumPy Archives (*.npz)")
        if path:
            try: self.current_model.load_weights(path); self.refresh_all_tabs(); self.statusBar().showMessage(f"Loaded from {path}", 3000)
            except Exception as e: self.statusBar().showMessage(f"Error: {e}", 3000)

    def set_click_mode(self, mode):
        self.click_mode = mode
        if mode == 'set_a': self.statusBar().showMessage("Click on plot to set Point A", 5000)
        elif mode == 'set_b': self.statusBar().showMessage("Click on plot to set Point B", 5000)

    def handle_plot_click(self, event):
        if event.inaxes != self.ax_latent: return
        x, y = max(-5.0, min(5.0, event.xdata)), max(-5.0, min(5.0, event.ydata))
        if self.click_mode == 'set_a':
            self.point_a_pos = np.array([x, y]); self.click_mode = 'navigate'
            self.z1a_spin.setValue(x); self.z2a_spin.setValue(y)
            self.statusBar().showMessage(f"Point A set to ({x:.2f}, {y:.2f})", 2000)
        elif self.click_mode == 'set_b':
            self.point_b_pos = np.array([x, y]); self.click_mode = 'navigate'
            self.z1b_spin.setValue(x); self.z2b_spin.setValue(y)
            self.statusBar().showMessage(f"Point B set to ({x:.2f}, {y:.2f})", 2000)
        else:
            self.slider_z1.setValue(int(x * 10)); self.slider_z2.setValue(int(y * 10))

    def random_sample(self):
        idx = np.random.randint(0, len(self.mock_latents))
        self.slider_z1.setValue(int(self.mock_latents[idx, 0] * 10)); self.slider_z2.setValue(int(self.mock_latents[idx, 1] * 10))

    def sample_normal(self):
        z = np.clip(np.random.randn(2), -5, 5)
        self.slider_z1.setValue(int(z[0] * 10)); self.slider_z2.setValue(int(z[1] * 10))

    def fgsm_attack(self):
        z1 = self.slider_z1.value() / 10.0; z2 = self.slider_z2.value() / 10.0
        z = cp.array([[z1, z2]], dtype=cp.float32)
        target = generate_target_batch(z, self.pattern_combo.currentText(), self.custom_target_img)
        self.current_model.forward(z, cache=True)
        _, _, grad_input = self.current_model.backward(target)
        eps = 0.5
        z_adv = z + eps * cp.sign(grad_input)
        z_adv = cp.clip(z_adv, -5.0, 5.0)
        z1_adv, z2_adv = _to_np(z_adv[0])
        self.slider_z1.setValue(int(z1_adv * 10)); self.slider_z2.setValue(int(z2_adv * 10))
        self.statusBar().showMessage(f"FGSM Attack applied! Z moved to [{z1_adv:.2f}, {z2_adv:.2f}]", 5000)

    def toggle_auto_rotate(self):
        if self.auto_rotate_timer.isActive():
            self.auto_rotate_timer.stop(); self.auto_btn.setText("▶ Auto"); self.auto_btn.setChecked(False)
        else:
            self.auto_rotate_t = 0.0; self.auto_rotate_timer.start(40); self.auto_btn.setText("⏸ Stop"); self.auto_btn.setChecked(True)

    def auto_rotate_tick(self):
        self.auto_rotate_t += 0.04
        self.slider_z1.setValue(int(3.5 * np.sin(self.auto_rotate_t) * 10))
        self.slider_z2.setValue(int(3.5 * np.sin(self.auto_rotate_t * 1.3 + 0.7) * 10))

    def start_interpolation(self):
        if self.point_a_pos is None or self.point_b_pos is None:
            self.statusBar().showMessage("Set both Point A and Point B first!", 3000); return
        self.interp_step = 0; n_steps = 60; self.interp_path = []
        ease_map = {'Linear': ease_linear, 'Smoothstep': ease_smoothstep, 'Cubic': ease_in_out_cubic}
        easing_fn = ease_map.get(self.ease_combo.currentText(), ease_linear)
        for i in range(n_steps + 1):
            t = i / n_steps; t_eased = easing_fn(t)
            self.interp_path.append((1 - t_eased) * self.point_a_pos + t_eased * self.point_b_pos)
        self.interp_timer.start(40)

    def interp_tick(self):
        self.interp_step += 1
        if self.interp_step >= len(self.interp_path): self.interp_timer.stop(); return
        z = self.interp_path[self.interp_step]
        self.slider_z1.setValue(int(z[0] * 10)); self.slider_z2.setValue(int(z[1] * 10))

    def generate_manifold_grid(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        g_min = self.grid_min_spin.value(); g_max = self.grid_max_spin.value()
        n = 20
        if self.polar_chk.isChecked():
            r = cp.linspace(0, g_max, n, dtype=cp.float32)
            theta = cp.linspace(0, 2 * cp.pi, n, dtype=cp.float32)
            rr, tt = cp.meshgrid(r, theta)
            xx = rr * cp.cos(tt); yy = rr * cp.sin(tt)
        else:
            grid_x = cp.linspace(g_min, g_max, n, dtype=cp.float32); grid_y = cp.linspace(g_min, g_max, n, dtype=cp.float32)
            xx, yy = cp.meshgrid(grid_x, grid_y)
            
        z_batch = cp.stack([xx.ravel(), yy.ravel()], axis=1)
        img_batch = _to_np(self.current_model.forward(z_batch)).reshape(n, n, 28, 28)
        grid_image = img_batch.swapaxes(1, 2).reshape(n * 28, n * 28)
        
        for ax in self.axes_man: ax.clear(); ax.set_facecolor('#2d2d2d'); style_mpl_axes(ax)
        self.axes_man[0].imshow(grid_image, cmap=self.cmap_combo.currentText())
        self.axes_man[0].set_title(f"Latent Grid [{'Polar' if self.polar_chk.isChecked() else 'Cartesian'}]", color='#cccccc')
        self.axes_man[0].set_xticks([]); self.axes_man[0].set_yticks([])
        fft_vals = np.fft.fft2(grid_image); fft_shifted = np.fft.fftshift(fft_vals)
        magnitude_spectrum = 20 * np.log(np.abs(fft_shifted) + 1e-8)
        self.axes_man[1].imshow(magnitude_spectrum, cmap='inferno')
        self.axes_man[1].set_title("2D FFT Magnitude", color='#cccccc')
        self.axes_man[1].set_xticks([]); self.axes_man[1].set_yticks([])
        self.fig_man.tight_layout(); self.canvas_man.draw()
        QApplication.restoreOverrideCursor()

    def run_training(self):
        epochs = self.epochs_spin.value(); lr = self.lr_spin.value(); batch_size = self.batch_spin.value()
        pattern = self.pattern_combo.currentText(); optim = self.optim_combo.currentText()
        scheduler = self.sch_combo.currentText(); l1 = self.l1_spin.value(); tv = self.tv_spin.value()
        
        freeze_flags = [chk.isChecked() for chk in self.freeze_checks[:len(self.current_model.layers)]]
        self.current_model.dropout_rate = self.drop_slider.value() / 100.0
        self.current_model.grad_noise_std = self.gn_slider.value() / 1000.0
            
        # Fixed validation set
        val_z = cp.random.randn(64, 2).astype(cp.float32) * 2.0
        val_target = generate_target_batch(val_z, pattern, self.custom_target_img)
        self.val_loss_history = []
        self.spec_norm_history = []
        
        self.train_btn.setEnabled(False); self.train_btn.setText("Training..."); QApplication.processEvents()
        for epoch in range(epochs):
            if scheduler == 'Step': curr_lr = lr * (0.5 ** (epoch // 50))
            elif scheduler == 'Cosine': curr_lr = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
            elif scheduler == 'Exponential': curr_lr = lr * (0.99 ** epoch)
            else: curr_lr = lr
            
            # Train batch
            z_batch = cp.random.randn(batch_size, 2).astype(cp.float32) * 2.0
            target_batch = generate_target_batch(z_batch, pattern, self.custom_target_img)
            loss, g_norms = self.current_model.train_batch(z_batch, target_batch, curr_lr, optim, freeze_flags, l1, tv)
            self.loss_history.append(loss); self.grad_norm_history.append(g_norms)
            
            # Validation & Metrics
            self.current_model.dropout_rate = 0.0 # Turn off dropout for val
            val_out = self.current_model.forward(val_z)
            val_loss = float(cp.mean((val_out - val_target)**2))
            self.val_loss_history.append(val_loss)
            self.spec_norm_history.append(self.current_model.get_spectral_norms())
            self.current_model.dropout_rate = self.drop_slider.value() / 100.0 # Restore
            
            if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
                self.refresh_training_tab(); self.update_visualization(); QApplication.processEvents()
                
        self.current_model.dropout_rate = 0.0
        self.train_btn.setEnabled(True); self.train_btn.setText("▶ Train Network")
        self.statusBar().showMessage(f"Training complete. Train Loss: {self.loss_history[-1]:.6f} | Val Loss: {self.val_loss_history[-1]:.6f}", 5000)
        self.refresh_dynamic_tabs()

    def prune_weights(self):
        pruned = self.current_model.prune_weights(self.prune_spin.value())
        self.statusBar().showMessage(f"Pruned {pruned} weights globally.", 5000)
        self.refresh_dynamic_tabs()

    def update_status_bar(self):
        self.statusBar().showMessage(f"  Backend: {'CuPy GPU' if USING_GPU else 'NumPy CPU'}  |  Device: {self.gpu_name}  |  Forward: {self.last_forward_ms:.3f} ms  |  Click mode: {self.click_mode}")

# ==========================================
# 4. RUN
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("CuPy Neural Net Explorer")
    window = AcademicFrontierApp()
    window.show()
    sys.exit(app.exec())