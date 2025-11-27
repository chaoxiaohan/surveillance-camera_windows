import sys
import os
import cv2
import time
import threading
import glob
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

MAX_TOTAL_BYTES = 20 * 1024 ** 3  # 20 GB
SEGMENT_SECONDS = 30 * 60  # 30 minutes


def human_size(n):
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def total_dir_size(path):
    total = 0
    for p in Path(path).rglob('*'):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def cleanup_old_files(path, limit_bytes=MAX_TOTAL_BYTES):
    files = [p for p in Path(path).iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    removed = 0
    while total > limit_bytes and files:
        oldest = files.pop(0)
        try:
            size = oldest.stat().st_size
            oldest.unlink()
            total -= size
            removed += 1
        except Exception:
            break
    return removed


class CaptureWorker(QtCore.QThread):
    frame_received = QtCore.Signal(object)
    status = QtCore.Signal(str)

    def __init__(self, device_index=0, resolution=(1280, 720), fps=30, encoder='mp4v', parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.resolution = resolution
        self.fps = fps
        self.encoder = encoder
        self._running = False
        self._record = False
        self.out_dir = str(Path.cwd() / 'recordings')
        self.segment_seconds = SEGMENT_SECONDS
        self.writer = None
        self.segment_start = None
        self.frame_size = None
        self._cap = None

    def run(self):
        # 优先尝试 DSHOW (Windows)，否则默认
        if os.name == 'nt':
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.device_index)

        if not self._cap or not self._cap.isOpened():
            self.status.emit('无法打开摄像头')
            return

        # 设置摄像头参数
        # 注意：部分摄像头可能不支持特定分辨率或FPS，设置可能无效，需以实际读取为准
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._running = True
        self.status.emit('摄像头预览中...')
        last_cleanup = time.time()

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self.status.emit('读取帧失败')
                # 尝试重连或退出？这里选择退出循环
                break

            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
            # 叠加时间戳
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 发送预览帧
            self.frame_received.emit(frame.copy())

            # 录制逻辑
            if self._record:
                if self.writer is None:
                    ensure_dir(self.out_dir)
                    cleanup_old_files(self.out_dir)
                    filename = now.strftime('%Y%m%d_%H%M%S') + '.mp4'
                    path = str(Path(self.out_dir) / filename)
                    h, w = frame.shape[:2]
                    self.frame_size = (w, h)
                    
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*self.encoder)
                        # 如果是 H264，有时需要 openh264 dll，或者系统安装了 ffmpeg
                        self.writer = cv2.VideoWriter(path, fourcc, self.fps, self.frame_size)
                        if not self.writer.isOpened():
                            self.status.emit(f'无法初始化编码器 {self.encoder}，尝试默认 mp4v')
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            self.writer = cv2.VideoWriter(path, fourcc, self.fps, self.frame_size)
                        
                        self.segment_start = time.time()
                        self.status.emit(f'正在录制: {filename}')
                    except Exception as e:
                        self.status.emit(f'录制初始化错误: {e}')
                        self.writer = None

                # 写入帧
                if self.writer:
                    try:
                        self.writer.write(frame)
                    except Exception:
                        pass

                # 分段检查
                if self.writer and (time.time() - (self.segment_start or 0) >= self.segment_seconds):
                    try:
                        self.writer.release()
                    except Exception:
                        pass
                    self.writer = None
                    self.segment_start = None
                    self.status.emit('分段保存完成，准备新文件')

                # 定期清理检查 (每60s)
                if time.time() - last_cleanup > 60:
                    cleanup_old_files(self.out_dir)
                    last_cleanup = time.time()

            else:
                # 如果不在录制状态但 writer 存在，说明刚停止录制
                if self.writer:
                    try:
                        self.writer.release()
                    except Exception:
                        pass
                    self.writer = None
                    self.segment_start = None
                    self.status.emit('录制已停止，继续预览')

            # 控制循环速度，避免占用过高 CPU，同时尽量接近目标 FPS
            # 简单的 sleep，实际帧率由摄像头采集速率决定
            QtCore.QThread.msleep(1)

        if self._cap and self._cap.isOpened():
            self._cap.release()
        if self.writer:
            try:
                self.writer.release()
            except Exception:
                pass
        self.status.emit('摄像头已断开')

    def stop(self):
        self._running = False

    def start_record(self):
        self._record = True

    def stop_record(self):
        self._record = False


class VideoWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)
        self.setStyleSheet('background: black; border: 2px solid #333;')

    def set_frame(self, frame):
        if frame is None:
            self._image = None
        else:
            h, w = frame.shape[:2]
            # OpenCV is BGR, Qt is RGB. Convert once per frame update.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # QImage(data, ...) requires the data to stay alive. .copy() ensures QImage owns the data.
            self._image = QtGui.QImage(frame_rgb.data, w, h, frame_rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        
        if self._image is None:
            painter.setPen(QtGui.QColor('#666'))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "无信号")
            return

        rect = self.rect()
        # Scale
        scaled_img = self._image.scaled(rect.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        
        # Center
        x = (rect.width() - scaled_img.width()) // 2
        y = (rect.height() - scaled_img.height()) // 2
        
        painter.drawImage(x, y, scaled_img)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('精美视频监控 (FFmpeg/OpenCV)')
        self.resize(1000, 700)

        self.worker = None

        # --- UI Components ---
        
        # 摄像头选择
        self.camera_combo = QtWidgets.QComboBox()
        self.scan_btn = QtWidgets.QPushButton('刷新设备')
        
        # 画质 (分辨率)
        self.res_combo = QtWidgets.QComboBox()
        self.res_combo.addItem("1920x1080 (1080p)", (1920, 1080))
        self.res_combo.addItem("1280x720 (720p)", (1280, 720))
        self.res_combo.addItem("640x480 (480p)", (640, 480))
        self.res_combo.setCurrentIndex(1) # 默认 720p

        # 帧率
        self.fps_combo = QtWidgets.QComboBox()
        for fps in [30, 25, 20, 15, 10, 5]:
            self.fps_combo.addItem(f"{fps} FPS", fps)
        self.fps_combo.setCurrentIndex(0) # 默认 30

        # 编码器
        self.enc_combo = QtWidgets.QComboBox()
        self.enc_combo.addItem("MPEG-4 (mp4v)", "mp4v")
        self.enc_combo.addItem("H.264 (avc1)", "avc1")
        self.enc_combo.addItem("H.264 (h264)", "h264")
        self.enc_combo.addItem("H.264 (x264)", "x264")
        self.enc_combo.setToolTip("如果 H.264 无法使用，将自动回退到 mp4v")

        # 录制控制
        self.start_btn = QtWidgets.QPushButton('开始录制')
        self.stop_btn = QtWidgets.QPushButton('停止录制')
        self.stop_btn.setEnabled(False)
        self.stop_btn.setObjectName("stop") # for stylesheet

        # 预览区域
        self.preview_label = VideoWidget()

        # 目录设置
        self.out_dir_edit = QtWidgets.QLineEdit(str(Path.cwd() / 'recordings'))
        self.dir_btn = QtWidgets.QPushButton('...')
        self.dir_btn.setFixedWidth(40)

        # 状态栏
        self.status_bar = QtWidgets.QLabel('就绪')
        self.status_bar.setStyleSheet("padding: 5px; color: #666;")

        # --- Layout ---
        
        # Settings Bar
        settings_layout = QtWidgets.QHBoxLayout()
        settings_layout.addWidget(QtWidgets.QLabel('摄像头:'))
        settings_layout.addWidget(self.camera_combo, 1)
        settings_layout.addWidget(self.scan_btn)
        settings_layout.addSpacing(20)
        settings_layout.addWidget(QtWidgets.QLabel('画质:'))
        settings_layout.addWidget(self.res_combo)
        settings_layout.addWidget(QtWidgets.QLabel('帧率:'))
        settings_layout.addWidget(self.fps_combo)
        settings_layout.addWidget(QtWidgets.QLabel('编码:'))
        settings_layout.addWidget(self.enc_combo)

        # Control Bar
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.addWidget(QtWidgets.QLabel('保存路径:'))
        ctrl_layout.addWidget(self.out_dir_edit, 1)
        ctrl_layout.addWidget(self.dir_btn)
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)

        # Main Layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(settings_layout)
        main_layout.addWidget(self.preview_label)
        main_layout.addLayout(ctrl_layout)
        main_layout.addWidget(self.status_bar)

        w = QtWidgets.QWidget()
        w.setLayout(main_layout)
        self.setCentralWidget(w)

        # --- Connections ---
        self.scan_btn.clicked.connect(self.scan_cameras)
        self.dir_btn.clicked.connect(self.choose_dir)
        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)
        
        # 任何设置改变都重启摄像头预览
        self.camera_combo.currentIndexChanged.connect(self.restart_camera)
        self.res_combo.currentIndexChanged.connect(self.restart_camera)
        self.fps_combo.currentIndexChanged.connect(self.restart_camera)
        self.enc_combo.currentIndexChanged.connect(self.update_encoder_only)

        # Timer for UI update
        self.ui_timer = QtCore.QTimer()
        self.ui_timer.timeout.connect(self.update_preview)
        self.last_frame = None
        self.ui_timer.start(30) # ~30fps UI refresh

        # Initial Scan
        self.scan_cameras()

    def scan_cameras(self):
        current_idx = self.camera_combo.currentData()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        
        # 简单的扫描逻辑
        found = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY)
            if cap.isOpened():
                found.append(i)
                cap.release()
        
        if not found:
            self.camera_combo.addItem('未检测到摄像头', -1)
        else:
            for idx in found:
                self.camera_combo.addItem(f'摄像头 {idx}', idx)
        
        # 尝试恢复之前的选择
        if current_idx is not None and current_idx in found:
            idx = self.camera_combo.findData(current_idx)
            self.camera_combo.setCurrentIndex(idx)
        
        self.camera_combo.blockSignals(False)
        
        # 扫描完如果有摄像头，自动启动预览
        if found:
            self.restart_camera()

    def restart_camera(self):
        # 停止旧的
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        idx = self.camera_combo.currentData()
        if idx is None or idx < 0:
            self.status_bar.setText("请选择有效的摄像头")
            return

        res = self.res_combo.currentData()
        fps = self.fps_combo.currentData()
        enc = self.enc_combo.currentData()

        self.worker = CaptureWorker(device_index=idx, resolution=res, fps=fps, encoder=enc)
        self.worker.out_dir = self.out_dir_edit.text()
        self.worker.frame_received.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.start()
        
        # 恢复录制状态？不，切换设置应停止录制
        self.stop_recording_ui()

    def update_encoder_only(self):
        # 编码器改变不需要重启摄像头，只需要更新 worker 参数
        if self.worker:
            self.worker.encoder = self.enc_combo.currentData()
            self.status_bar.setText(f"编码器已更新为: {self.worker.encoder}")

    def choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, '选择输出目录', str(Path.cwd()))
        if d:
            self.out_dir_edit.setText(d)
            if self.worker:
                self.worker.out_dir = d

    def start_recording(self):
        if not self.worker or not self.worker.isRunning():
            self.restart_camera()
            # 等待一下启动？
            QtCore.QTimer.singleShot(500, self.start_recording)
            return

        self.worker.start_record()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # 禁用设置，防止录制中更改导致崩溃
        self.camera_combo.setEnabled(False)
        self.res_combo.setEnabled(False)
        self.fps_combo.setEnabled(False)
        self.enc_combo.setEnabled(False)

    def stop_recording(self):
        if self.worker:
            self.worker.stop_record()
        self.stop_recording_ui()

    def stop_recording_ui(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.res_combo.setEnabled(True)
        self.fps_combo.setEnabled(True)
        self.enc_combo.setEnabled(True)

    @QtCore.Slot(object)
    def on_frame(self, frame):
        self.last_frame = frame

    @QtCore.Slot(str)
    def on_status(self, s):
        self.status_bar.setText(s)

    def update_preview(self):
        if self.last_frame is None:
            return
        self.preview_label.set_frame(self.last_frame)

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion") # 现代风格
    
    # Dark Theme Palette
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

    app.setStyleSheet('''
        QPushButton { 
            background-color: #2b5b84; 
            color: white; 
            border-radius: 4px; 
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #3a7ca5; }
        QPushButton:pressed { background-color: #1e4261; }
        QPushButton:disabled { background-color: #555; color: #888; }
        QPushButton#stop { background-color: #c62828; }
        QPushButton#stop:hover { background-color: #e53935; }
        
        QComboBox { 
            padding: 4px; 
            border: 1px solid #555; 
            border-radius: 4px; 
            background: #333; 
            color: white; 
        }
        QLineEdit { 
            padding: 4px; 
            border: 1px solid #555; 
            border-radius: 4px; 
            background: #333; 
            color: white; 
        }
        QLabel { color: #ddd; }
    ''')

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
