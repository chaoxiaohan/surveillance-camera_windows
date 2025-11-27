import sys
import os
import cv2
import time
import threading
import glob
from datetime import datetime
from pathlib import Path
import subprocess
import shutil
import tempfile
import numpy as np

# optional audio libs
try:
    import sounddevice as sd
    import soundfile as sf
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

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


class AudioRecorder:
    """Simple audio recorder using sounddevice + soundfile.
    Writes to WAV file until stopped. Non-blocking start/stop.
    """
    def __init__(self, path, samplerate=44100, channels=1):
        self.path = path
        self.samplerate = samplerate
        self.channels = channels
        self._stream = None
        self._file = None

    def start(self):
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError('sounddevice/soundfile not available')

        self._file = sf.SoundFile(self.path, mode='w', samplerate=self.samplerate, channels=self.channels)

        def callback(indata, frames, time_info, status):
            if status:
                # ignore
                pass
            # write a copy to avoid referencing temporary buffer
            try:
                self._file.write(indata.copy())
            except Exception:
                pass

        self._stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels, callback=callback)
        self._stream.start()

    def stop(self):
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception:
            pass
        try:
            if self._file is not None:
                self._file.close()
                self._file = None
        except Exception:
            pass


def finalize_segment(temp_video, temp_audio, final_path, status_signal=None, actual_fps=None):
    """If ffmpeg is available and temp_audio exists, mux audio+video into final_path.
    Always re-encode video with actual measured FPS to ensure correct playback speed.
    Otherwise move/rename temp_video -> final_path and keep audio aside.
    status_signal: optional Qt Signal to emit status strings.
    actual_fps: measured FPS during recording (frames / duration)
    """
    try:
        if temp_audio and os.path.exists(temp_audio) and shutil.which('ffmpeg'):
            # 始终使用实际测量的帧率重新编码，确保播放速度正确
            if actual_fps and actual_fps > 0:
                if status_signal:
                    status_signal.emit(f'合并中 (实际帧率: {actual_fps:.1f} FPS)...')
                
                # 使用 ffmpeg 重新编码视频，设置正确的帧率
                # 只在输入端指定帧率，让 ffmpeg 重新解释时间戳
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                    '-r', str(actual_fps),  # 输入帧率（告诉 ffmpeg 读取视频时按此帧率解释）
                    '-i', temp_video,
                    '-i', temp_audio,
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '128k',
                    '-shortest',  # 以较短的流为准（避免音视频时长不一致）
                    final_path
                ]
            else:
                # 没有有效的帧率信息，直接复制
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', temp_video, '-i', temp_audio,
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                    '-shortest',
                    final_path
                ]
            
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 0:
                # remove temp files
                try:
                    os.remove(temp_video)
                except Exception:
                    pass
                try:
                    os.remove(temp_audio)
                except Exception:
                    pass
                if status_signal:
                    status_signal.emit('分段已合并音频')
                return True
            else:
                err_msg = p.stderr.decode('utf-8', errors='ignore') if p.stderr else ''
                if status_signal:
                    status_signal.emit(f'ffmpeg 合并失败: {err_msg[:100]}')
                # fallthrough to move
        # fallback: move temp_video to final
        try:
            shutil.move(temp_video, final_path)
        except Exception:
            # attempt copy
            try:
                shutil.copy2(temp_video, final_path)
                os.remove(temp_video)
            except Exception:
                pass
        if status_signal:
            status_signal.emit('分段保存为文件')
    except Exception as e:
        if status_signal:
            status_signal.emit(f'合并/保存段失败: {e}')
    return False


class CaptureWorker(QtCore.QThread):
    frame_received = QtCore.Signal(object)
    status = QtCore.Signal(str)

    def __init__(self, device_index=0, resolution=(1280, 720), fps=30, encoder='mp4v', audio_enabled=False, parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.resolution = resolution
        self.fps = fps
        self.encoder = encoder
        self.audio_enabled = audio_enabled and SOUNDDEVICE_AVAILABLE
        self.audio_samplerate = 44100
        self.audio_channels = 1
        self._audio_recorder = None
        self._running = False
        self._record = False
        self.out_dir = str(Path.cwd() / 'recordings')
        self.segment_seconds = SEGMENT_SECONDS
        self.writer = None
        self.segment_start = None
        self.frame_size = None
        self._cap = None
        # 用于计算实际 FPS
        self._frame_count = 0
        self._record_start_time = None

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

            # 录制逻辑（包含音频录制与分段合并）
            if self._record:
                if self.writer is None:
                    ensure_dir(self.out_dir)
                    cleanup_old_files(self.out_dir)
                    filename = now.strftime('%Y%m%d_%H%M%S') + '.mp4'
                    final_path = str(Path(self.out_dir) / filename)
                    temp_video = final_path + '.vtmp.mp4'
                    temp_audio = final_path + '.atmp.wav' if self.audio_enabled else None
                    h, w = frame.shape[:2]
                    self.frame_size = (w, h)
                    
                    # 使用用户设定的 FPS 作为初始值（实际 FPS 会在录制完成后重新计算）
                    # 摄像头的 CAP_PROP_FPS 往往不准确，所以使用固定值先写入
                    initial_fps = self.fps
                    
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*self.encoder)
                        self.writer = cv2.VideoWriter(temp_video, fourcc, initial_fps, self.frame_size)
                        if not self.writer.isOpened():
                            self.status.emit(f'无法初始化编码器 {self.encoder}，尝试默认 mp4v')
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            self.writer = cv2.VideoWriter(temp_video, fourcc, initial_fps, self.frame_size)

                        # start audio recorder if requested
                        if self.audio_enabled:
                            try:
                                self._audio_recorder = AudioRecorder(temp_audio, samplerate=self.audio_samplerate, channels=self.audio_channels)
                                self._audio_recorder.start()
                            except Exception as e:
                                self.status.emit(f'音频启动失败: {e}')
                                self._audio_recorder = None

                        self._current_final = final_path
                        self._current_temp_video = temp_video
                        self._current_temp_audio = temp_audio
                        self._current_initial_fps = initial_fps
                        self.segment_start = time.time()
                        self._record_start_time = time.time()
                        self._frame_count = 0
                        self.status.emit(f'正在录制: {filename}')
                    except Exception as e:
                        self.status.emit(f'录制初始化错误: {e}')
                        self.writer = None

                # 写入帧
                if self.writer:
                    try:
                        self.writer.write(frame)
                        self._frame_count += 1
                    except Exception:
                        pass

                # 分段检查
                if self.writer and (time.time() - (self.segment_start or 0) >= self.segment_seconds):
                    try:
                        self.writer.release()
                    except Exception:
                        pass

                    # stop audio recorder for this segment
                    if self._audio_recorder:
                        try:
                            self._audio_recorder.stop()
                        except Exception:
                            pass

                    # 计算实际录制时长和帧率
                    actual_duration = time.time() - self._record_start_time if self._record_start_time else 0
                    actual_fps_measured = self._frame_count / actual_duration if actual_duration > 0 else self._current_initial_fps
                    
                    # finalize (mux) this segment，传递实际测量的 FPS
                    try:
                        finalize_segment(
                            self._current_temp_video, 
                            self._current_temp_audio, 
                            self._current_final, 
                            self.status,
                            actual_fps=actual_fps_measured
                        )
                    except Exception:
                        pass

                    self.writer = None
                    self._audio_recorder = None
                    self.segment_start = None
                    self._frame_count = 0
                    self._record_start_time = None
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
                    # stop audio recorder and finalize
                    if self._audio_recorder:
                        try:
                            self._audio_recorder.stop()
                        except Exception:
                            pass
                        
                        # 计算实际录制时长和帧率
                        actual_duration = time.time() - self._record_start_time if self._record_start_time else 0
                        actual_fps_measured = self._frame_count / actual_duration if actual_duration > 0 else self._current_initial_fps
                        
                        try:
                            finalize_segment(
                                self._current_temp_video, 
                                self._current_temp_audio, 
                                self._current_final, 
                                self.status,
                                actual_fps=actual_fps_measured
                            )
                        except Exception:
                            pass

                    self.writer = None
                    self._audio_recorder = None
                    self.segment_start = None
                    self._frame_count = 0
                    self._record_start_time = None
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
        self.setWindowTitle('视频监控电脑端@choshokan')
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
        # 默认只添加兼容性更好的 MPEG-4 编码器 (mp4v)
        # 仅在检测到 FFmpeg 时才展示 H.264 相关选项，以避免加载系统上不兼容的 OpenH264 库
        self.enc_combo.addItem("MPEG-4 (mp4v)", "mp4v")
        if shutil.which('ffmpeg'):
            self.enc_combo.addItem("H.264 (avc1)", "avc1")
            self.enc_combo.addItem("H.264 (h264)", "h264")
            self.enc_combo.addItem("H.264 (x264)", "x264")
            self.enc_combo.setToolTip("检测到 FFmpeg，可选择 H.264；否则使用 mp4v")
        else:
            self.enc_combo.setToolTip("系统未检测到 FFmpeg，已隐藏 H.264 选项以避免 OpenH264 冲突")
        # 默认选择第一个 (mp4v)
        self.enc_combo.setCurrentIndex(0)
        
        # 音频录制开关
        self.audio_chk = QtWidgets.QCheckBox('录音')
        if not SOUNDDEVICE_AVAILABLE:
            self.audio_chk.setEnabled(False)
            self.audio_chk.setToolTip('未检测到 sounddevice/soundfile，无法录音')
        else:
            self.audio_chk.setChecked(True)
            if not shutil.which('ffmpeg'):
                self.audio_chk.setToolTip('未检测到 FFmpeg，录音将保存为单独的 .wav 文件')

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
        settings_layout.addWidget(self.audio_chk)

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
        self.audio_chk.stateChanged.connect(self.restart_camera)

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
        audio_enabled = self.audio_chk.isChecked()

        self.worker = CaptureWorker(device_index=idx, resolution=res, fps=fps, encoder=enc, audio_enabled=audio_enabled)
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
