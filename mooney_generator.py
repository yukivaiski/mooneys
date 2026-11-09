import os
import sys
from glob import glob

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QLineEdit, QFileDialog, QListWidget,
    QProgressBar, QGroupBox, QMessageBox, QAbstractItemView
)


def generate_mooney(image_path, output_dir, blur_strength=15, white_pixel_ratio=0.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, f"Couldn't read image: {image_path}"

    ksize = int(blur_strength)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(1, ksize)

    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)

    percentile = (1.0 - np.clip(white_pixel_ratio, 0.0, 1.0)) * 100
    threshold_value = np.percentile(blurred, percentile)

    _, mooney = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_mooney{ext}")

    cv2.imwrite(output_path, mooney)
    return True, output_path


class ProcessingWorker(QThread):
    progress = pyqtSignal(int, int)  # (current, total)
    log_message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, inputs, output_dir, blur_strength, white_pixel_ratio):
        super().__init__()
        self.inputs = inputs
        self.output_dir = output_dir
        self.blur_strength = blur_strength
        self.white_pixel_ratio = white_pixel_ratio

    def run(self):
        all_files = []
        for path in self.inputs:
            if os.path.isdir(path):
                extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
                for ext in extensions:
                    all_files.extend(glob(os.path.join(path, ext)))
                    all_files.extend(glob(os.path.join(path, ext.upper())))
            elif os.path.isfile(path):
                all_files.append(path)

        all_files = list(set(all_files))
        total = len(all_files)

        if total == 0:
            self.log_message.emit("No valid images found here")
            self.finished.emit()
            return

        self.log_message.emit(f"Processing {total} images...")

        for idx, file_path in enumerate(all_files, start=1):
            success, msg = generate_mooney(file_path, self.output_dir, self.blur_strength, self.white_pixel_ratio)
            if success:
                self.log_message.emit(f"[{idx}/{total}] Saved: {os.path.basename(msg)}")
            else:
                self.log_message.emit(f"[{idx}/{total}] Error: {msg}")
            self.progress.emit(idx, total)

        self.finished.emit()


class MooneyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mooney Image Generator")
        self.resize(1000, 800)

        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QGroupBox { font-weight: bold; color: #cdd6f4; border: 1px solid #45475a; border-radius: 8px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #cdd6f4; font-size: 13px; }
            QLineEdit, QListWidget { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 5px; padding: 5px; }
            QPushButton { background-color: #89b4fa; color: #11111b; font-weight: bold; border-radius: 6px; padding: 8px 15px; }
            QPushButton:hover { background-color: #b4befe; }
            QPushButton:pressed { background-color: #74c7ec; }
            QProgressBar { border: 1px solid #45475a; border-radius: 5px; text-align: center; color: #cdd6f4; background-color: #313244; }
            QProgressBar::chunk { background-color: #a6e3a1; border-radius: 4px; }
            QSlider::groove:horizontal { height: 6px; background: #313244; border-radius: 3px; }
            QSlider::handle:horizontal { background: #89b4fa; width: 16px; margin: -5px 0; border-radius: 8px; }
            QMessageBox { background-color: #1e1e2e; }
            QMessageBox QLabel { color: #cdd6f4; font-size: 14px; }
            QMessageBox QPushButton { background-color: #89b4fa; color: #11111b; min-width: 70px; }
        """)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        input_group = QGroupBox("1. Input Files / Directories")
        input_layout = QVBoxLayout()

        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(180)
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        btn_layout = QHBoxLayout()
        self.btn_add_files = QPushButton("Add Images")
        self.btn_add_files.clicked.connect(self.add_files)

        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.clicked.connect(self.add_folder)

        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_remove_selected.setStyleSheet("background-color: #fab387; color: #11111b;")
        self.btn_remove_selected.clicked.connect(self.remove_selected_item)

        self.btn_clear_inputs = QPushButton("Clear All")
        self.btn_clear_inputs.setStyleSheet("background-color: #f38ba8;")
        self.btn_clear_inputs.clicked.connect(self.file_list.clear)

        btn_layout.addWidget(self.btn_add_files)
        btn_layout.addWidget(self.btn_add_folder)
        btn_layout.addWidget(self.btn_remove_selected)
        btn_layout.addWidget(self.btn_clear_inputs)

        input_layout.addWidget(self.file_list)
        input_layout.addLayout(btn_layout)
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        param_group = QGroupBox("2. Parameters")
        param_layout = QVBoxLayout()
        param_layout.setSpacing(12)

        blur_box = QHBoxLayout()
        blur_label = QLabel("Blur Strength:")
        blur_label.setFixedWidth(140)
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(1, 99)
        self.blur_slider.setValue(21)
        self.blur_val_label = QLabel("21")
        self.blur_val_label.setFixedWidth(30)
        self.blur_slider.valueChanged.connect(lambda v: self.blur_val_label.setText(str(v)))

        blur_box.addWidget(blur_label)
        blur_box.addWidget(self.blur_slider)
        blur_box.addWidget(self.blur_val_label)

        ratio_box = QHBoxLayout()
        ratio_label = QLabel("White Pixel Ratio:")
        ratio_label.setFixedWidth(140)
        self.ratio_slider = QSlider(Qt.Horizontal)
        self.ratio_slider.setRange(0, 100)
        self.ratio_slider.setValue(45)
        self.ratio_val_label = QLabel("0.45")
        self.ratio_val_label.setFixedWidth(30)
        self.ratio_slider.valueChanged.connect(lambda v: self.ratio_val_label.setText(f"{v / 100:.2f}"))

        ratio_box.addWidget(ratio_label)
        ratio_box.addWidget(self.ratio_slider)
        ratio_box.addWidget(self.ratio_val_label)

        param_layout.addLayout(blur_box)
        param_layout.addLayout(ratio_box)
        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)

        output_group = QGroupBox("3. Output Destination")
        output_layout = QHBoxLayout()

        self.output_path_input = QLineEdit("mooneys")
        self.btn_browse_output = QPushButton("Browse")
        self.btn_browse_output.clicked.connect(self.browse_output)

        output_layout.addWidget(self.output_path_input)
        output_layout.addWidget(self.btn_browse_output)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)

        self.btn_generate = QPushButton("Generate Mooney Images")
        self.btn_generate.setFixedHeight(50)
        self.btn_generate.setStyleSheet("font-size: 16px; background-color: #a6e3a1; color: #11111b;")
        self.btn_generate.clicked.connect(self.start_processing)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        main_layout.addWidget(self.btn_generate)
        main_layout.addWidget(self.progress_bar)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Image Files", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        for f in files:
            self.file_list.addItem(f)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Images")
        if folder:
            self.file_list.addItem(folder)

    def remove_selected_item(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_path_input.setText(folder)

    def start_processing(self):
        inputs = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not inputs:
            QMessageBox.warning(self, "No Input", "Please select at least one file or folder.")
            return

        output_dir = self.output_path_input.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please specify an output folder")
            return

        blur_strength = self.blur_slider.value()
        white_ratio = self.ratio_slider.value() / 100.0

        self.btn_generate.setEnabled(False)
        self.progress_bar.setValue(0)

        self.worker = ProcessingWorker(inputs, output_dir, blur_strength, white_ratio)
        self.worker.progress.connect(self.update_progress)
        self.worker.log_message.connect(lambda msg: self.statusBar().showMessage(msg, 3000))
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()

    def update_progress(self, current, total):
        percentage = int((current / total) * 100)
        self.progress_bar.setValue(percentage)

    def processing_finished(self):
        self.btn_generate.setEnabled(True)
        QMessageBox.information(self, "Complete", "Mooney images processing finished successfully!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MooneyApp()
    window.show()
    sys.exit(app.exec_())
