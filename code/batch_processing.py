#批量图像处理工具
import os
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QAction, QMessageBox, QWidget, QHeaderView,
                            QVBoxLayout, QLabel, QFileDialog, QListWidget, QSplitter,
                            QCheckBox, QSpinBox, QGridLayout, QPushButton, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QSize, QSettings, pyqtSignal, QThread
from PyQt5.QtGui import QIcon, QFont
from glcm_calculator import calculate_features, rescale_image
import traceback
import csv
from PIL import Image
from image_loader import rescale_image
from PyQt5.QtCore import pyqtSignal, QObject
# 批量图像分析主窗口类
class BatchAnalysisWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from resources import QSS_STYLE, get_icon_from_base64
        self.setStyleSheet(QSS_STYLE)
        self.setWindowIcon(get_icon_from_base64())
        app_font = QFont()
        app_font.setFamily("Microsoft YaHei")
        app_font.setPointSize(10)
        self.setFont(app_font)
        self.setWindowTitle("批量图像处理")
        self.setGeometry(600, 400, 900, 700)
        self.setMinimumSize(800, 600)
        self.imported_images = []
        self.settings = QSettings("MyCompany", "TextureAnalysis")
        self.create_menu()
        self.init_ui()
        self.imported_images = self.settings.value("batch/imported_images", [])
        self.statusBar().showMessage("就绪")
    # 创建主菜单系统
    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        import_action = QAction("导入图像 (Ctrl+I)", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self.import_images)
        file_menu.addAction(import_action)
        export_action = QAction("导出特征表格 (Ctrl+E)", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_features)
        file_menu.addSeparator()
        file_menu.addAction(export_action)
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        process_menu = menubar.addMenu("处理")
        process_menu.addAction(QAction("开始处理", self, triggered=self.start_processing)) 
        clear_menu = menubar.addMenu("数据管理")
        clear_action = QAction("清空数据", self)
        clear_action.triggered.connect(self.clear_data)
        clear_menu.addAction(clear_action)
        help_menu = menubar.addMenu("帮助")
        help_menu.addAction(QAction("关于", self, triggered=self.show_about))
    # 导出特征数据到CSV文件
    def export_features(self):
            row_count = self.result_table.rowCount()
            if row_count == 0:
                QMessageBox.warning(self, "警告", "没有可导出的数据")
                return
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存特征数据", "", "CSV文件 (*.csv)"
            )
            if not save_path:
                return
            if not save_path.endswith(".csv"):
                save_path += ".csv"
            try:
                with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f, delimiter=',')
                    column_count = self.result_table.columnCount()
                    headers = [
                                  "Index",
                                  "File Path",
                                  "Contrast",
                                  "Energy",
                                  "Homogeneity",
                                  "Correlation",
                                  "Dissimilarity",
                                  "Autocorrelation"
                              ][:column_count]
                    writer.writerow(headers)
                    for row in range(row_count):
                        row_data = []
                        for col in range(column_count):
                            item = self.result_table.item(row, col)
                            row_data.append(item.text() if item else '')
                        writer.writerow(row_data)
                QMessageBox.information(self, "成功", f"数据已导出至：{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")
                traceback.print_exc()
    # 开始批量处理图像特征
    def start_processing(self):
            if not self.imported_images:
                QMessageBox.warning(self, "警告", "请先导入图像文件")
                return
            params = {
                'step': self.step_input.value(),
                'distance': self.distance_input.value(),
                'angle': self.angle_input.value(),
                'grayscale': self.gray_input.value()
            }
            self.processing_thread = self.ProcessingThread(self.imported_images, params)
            self.result_table.setRowCount(len(self.imported_images))
            self.processing_thread.result_ready.connect(self.update_table)
            self.processing_thread.start()
    # 显示关于对话框
    def show_about(self):
            QMessageBox.about(
                self, 
                "关于批量图像处理",
                """<h3>批量图像处理模块 v1.0</h3>
                <p>支持GLCM纹理特征批量计算与分析</p>
                <p><b>核心功能：</b></p>
                <ul>
                <li>多格式图像批量导入（BMP/PNG/JPG/TIF等）</li>
                <li>统一GLCM参数配置（步长/距离/角度/灰度级）</li>
                <li>6种纹理特征批量提取（对比度、能量、同质性等）</li>
                <li>批量结果导出（CSV格式）</li>
                </ul>
                <p><b>操作步骤：</b></p>
                <ol>
                <li>通过「文件」→「导入图像」选择多个图像文件</li>
                <li>在参数面板选择需要计算的特征</li>
                <li>设置GLCM参数（步长、距离、角度、灰度级）</li>
                <li>点击「处理」→「开始处理」执行批量分析</li>
                <li>通过「文件」→「导出特征表格」保存结果</li>
                </ol>
                <p><b>快捷键：</b></p>
                <ul>
                <li>Ctrl+I：批量导入图像</li>
                <li>Ctrl+E：导出特征表格</li>
                </ul>
                <p><b>参数说明：</b></p>
                <ul>
                <li>步长：图像重采样间隔（1-10像素）</li>
                <li>距离：GLCM计算距离（1-20像素）</li>
                <li>角度：GLCM计算角度（0-180度）</li>
                <li>灰度级：图像灰度量化级别（2-512）</li>
              """
            )
    # 初始化用户界面
    def init_ui(self):
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            file_list_widget = QWidget()
            file_list_layout = QVBoxLayout(file_list_widget)
            file_list_layout.addWidget(QLabel("已导入文件列表", alignment=Qt.AlignCenter))
            self.file_list = QListWidget()
            self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
            file_list_layout.addWidget(self.file_list)
            param_widget = QWidget()
            param_layout = QGridLayout(param_widget)
            self.feature_checkboxes = {
                '对比度': QCheckBox('对比度'),
                '能量': QCheckBox('能量'),
                '同质性': QCheckBox('同质性'),
                '相关性': QCheckBox('相关性'),
                '差异性': QCheckBox('差异性'),
                '自相关性': QCheckBox('自相关性')
            }
            param_layout.addWidget(QLabel("选择特征："), 0, 0, 1, 2)
            for row, (feature, cb) in enumerate(self.feature_checkboxes.items(), start=1):
                param_layout.addWidget(cb, row, 0)
            self.step_input = QSpinBox()
            self.step_input.setRange(1, 10)
            self.step_input.setValue(1)
            self.distance_input = QSpinBox()
            self.distance_input.setRange(1, 20)
            self.distance_input.setValue(5)
            self.angle_input = QSpinBox()
            self.angle_input.setRange(0, 180)
            self.angle_input.setValue(0)
            self.gray_input = QSpinBox()
            self.gray_input.setRange(2, 512)
            self.gray_input.setValue(256)
            param_layout.addWidget(QLabel("步长："), 1, 2)
            param_layout.addWidget(self.step_input, 1, 3)
            param_layout.addWidget(QLabel("距离："), 2, 2)
            param_layout.addWidget(self.distance_input, 2, 3)
            param_layout.addWidget(QLabel("角度 (°)："), 3, 2)
            param_layout.addWidget(self.angle_input, 3, 3)
            param_layout.addWidget(QLabel("灰度级："), 4, 2)
            param_layout.addWidget(self.gray_input, 4, 3)
            reset_btn = QPushButton("恢复默认参数")
            reset_btn.clicked.connect(self.restore_defaults)
            param_layout.addWidget(reset_btn, 5, 2, 1, 2)
            self.result_table = QTableWidget()
            self.result_table.setColumnCount(2)
            self.result_table.setHorizontalHeaderLabels([
            "索引", "文件名"
            ])
            self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            main_layout.addWidget(file_list_widget)
            main_layout.addWidget(param_widget)
            main_layout.addWidget(self.result_table)
    def update_table(self, row_idx, features):
        selected_features = [feature for feature, cb in self.feature_checkboxes.items() if cb.isChecked()]
        self.result_table.setColumnCount(2 + len(selected_features))
        headers = ["索引", "文件名"] + selected_features
        self.result_table.setHorizontalHeaderLabels(headers)
        self.result_table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
        file_name = os.path.basename(self.imported_images[row_idx])
        path_item = QTableWidgetItem(file_name)
        path_item.setToolTip(self.imported_images[row_idx])
        self.result_table.setItem(row_idx, 1, path_item)
        feature_mapping = {
            '对比度': 'contrast',
            '能量': 'energy',
            '同质性': 'homogeneity',
            '相关性': 'correlation', 
            '差异性': 'dissimilarity',
            '自相关性': 'autocorrelation'
        }
        for col, feature in enumerate(selected_features, start=2):
            value = features.get(feature_mapping[feature], 0.0)
            item = QTableWidgetItem(f"{value:.4f}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.result_table.setItem(row_idx, col, item)
        self.result_table.resizeColumnsToContents()
    def import_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图像文件",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.heic);;所有文件 (*.*)"
        )
        if not files:
            return
        valid_files = []
        error_messages = []
        failed_files = []
        for file_path in files:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.heic'}:
                error_messages.append(f"不支持的格式：{os.path.basename(file_path)}")
                failed_files.append(file_path)
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
                valid_files.append(file_path)
            except Exception as e:
                error_messages.append(f"导入失败：{os.path.basename(file_path)} - {str(e)}")
                failed_files.append(file_path)
        self.imported_images = valid_files
        self.file_list.clear()
        for file_name in [os.path.basename(p) for p in valid_files]:
            self.file_list.addItem(file_name)
        if error_messages:
            self._show_error_dialog(error_messages, failed_files)
        self.statusBar().showMessage(f"成功导入 {len(valid_files)} 个文件", 5000)
    def _show_error_dialog(self, messages, files):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("导入错误")
        dialog.setIcon(QMessageBox.Warning)
        dialog.setText(f"以下文件导入失败（共 {len(files)} 个）：")
        dialog.setInformativeText("\n".join(messages))
        dialog.setDetailedText("失败文件路径：\n" + "\n".join(files))
        dialog.exec_()
    def restore_defaults(self):
        self.step_input.setValue(1)
        self.distance_input.setValue(5)
        self.angle_input.setValue(0)
        self.gray_input.setValue(256)
        for cb in self.feature_checkboxes.values():
            cb.setChecked(False)
    # 多线程处理类 - 在后台线程中执行GLCM特征计算
    class ProcessingThread(QThread):
        progress = pyqtSignal(int)
        result_ready = pyqtSignal(int, dict)
        glcm_ready = pyqtSignal(object)
        def __init__(self, image_paths, params):
            super().__init__()
            self.image_paths = image_paths
            self.params = params
            self._is_running = True
        def run(self):
            for idx, path in enumerate(self.image_paths):
                try:
                    features, glcm_matrix = calculate_features(
                        path,
                        step=self.params['step'],
                        distance=self.params['distance'],
                        angle=np.deg2rad(self.params['angle']),
                        grayscale=self.params['grayscale']
                    )
                    self.result_ready.emit(idx, features)
                    self.glcm_ready.emit(glcm_matrix)
                except Exception as e:
                    print(f"处理失败：{path}，错误：{str(e)}")
                self.progress.emit(int((idx + 1) / len(self.image_paths) * 100))
    def _handle_glcm(self, glcm_matrix):
        self.last_glcm = glcm_matrix
    def closeEvent(self, event):
        if hasattr(self, 'processing_thread') and self.processing_thread.isRunning():
            self.processing_thread.terminate()
            self.processing_thread.wait()
        self.settings.setValue("batch/imported_images", self.imported_images)
        super().closeEvent(event)
    def clear_data(self):
        self.imported_images.clear()
        self.file_list.clear()
        self.result_table.setRowCount(0)
        self.statusBar().showMessage("数据已清空", 3000)
        self.settings.remove("batch/imported_images")
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    window = BatchAnalysisWindow()
    window.show()
    app.exec_()