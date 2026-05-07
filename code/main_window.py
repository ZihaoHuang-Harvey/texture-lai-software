# 单张图像处理工具
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit, \
    QCheckBox, QPushButton, QFormLayout, QMenuBar, QMenu, QAction, QFileDialog, QInputDialog, QMessageBox, QGridLayout, \
    QFrame, QDialog, QSizePolicy
from model_inference import ModelBuilderWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import QSizePolicy
class ImageTextureAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.default_params = {
            "step": "1",
            "distance": "5",
            "angle": "0",
            "grayscale": "256"
        }
        self.current_image = None
        self.original_image_copy = None
        self.recent_files = []
        self.max_recent_files = 5
        self._pixmap_cache = {}
        self._is_resizing = False
        self.setWindowTitle("图像纹理分析系统")
        self.setGeometry(100, 100, 1024, 768)
        self.setMinimumSize(900, 700)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        resample_menu = menubar.addMenu("重采样")
        analysis_menu = menubar.addMenu("分析")
        view_menu = menubar.addMenu("视图")
        help_menu = menubar.addMenu("帮助")
        open_image_action = QAction("打开图像", self)
        open_image_action.setShortcut("Ctrl+O")
        open_image_action.triggered.connect(self.open_image)
        file_menu.addAction(open_image_action)
        self.recent_files_menu = QMenu("最近文件", self)
        file_menu.addMenu(self.recent_files_menu)
        file_menu.addSeparator()
        export_image_action = QAction("导出图像", self)
        export_image_action.setShortcut("Ctrl+S")
        export_image_action.triggered.connect(self.export_image)
        file_menu.addAction(export_image_action)
        export_data_action = QAction("导出数据", self)
        export_data_action.setShortcut("Ctrl+E")
        export_data_action.triggered.connect(self.export_data)
        file_menu.addAction(export_data_action)
        export_glcm_data_action = QAction("导出GLCM矩阵", self)
        export_glcm_data_action.setShortcut("Ctrl+M")
        export_glcm_data_action.triggered.connect(self.export_glcm_data)
        file_menu.addAction(export_glcm_data_action)
        export_glcm_image_action = QAction("导出GLCM图像", self)
        export_glcm_image_action.setShortcut("Ctrl+I")
        export_glcm_image_action.triggered.connect(self.export_glcm_image)
        file_menu.addAction(export_glcm_image_action)
        export_all_action = QAction("一键导出全部", self)
        export_all_action.setShortcut("Ctrl+A")
        export_all_action.triggered.connect(self.export_all_data)
        file_menu.addAction(export_all_action)
        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        resample_action = QAction("重采样...", self)
        resample_action.setShortcut("Ctrl+R")
        resample_action.triggered.connect(self.resample_image)
        resample_menu.addAction(resample_action)
        process_action = QAction("处理图像", self)
        process_action.setShortcut("F5")
        process_action.triggered.connect(self.process_image)
        analysis_menu.addAction(process_action)
        reset_view_action = QAction("重置视图", self)
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)
        clear_all_action = QAction("一键清除", self)
        clear_all_action.setShortcut("Ctrl+Z")
        clear_all_action.triggered.connect(self.clear_all)
        view_menu.addAction(clear_all_action)
        usage_guide_action = QAction("使用指南", self)
        usage_guide_action.triggered.connect(self.show_usage_guide)
        help_menu.addAction(usage_guide_action)
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_area = QHBoxLayout()
        main_area.setSpacing(10)
        preview_group = QGroupBox("预览窗口")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_image = QLabel()
        self.preview_image.setMinimumSize(300, 200)
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setObjectName("previewImage")
        self.preview_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self.preview_image)
        main_area.addWidget(preview_group, stretch=5)
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout(param_group)
        param_layout.setVerticalSpacing(10)
        self.step_length = QLineEdit(self.default_params["step"])
        self.distance = QLineEdit(self.default_params["distance"])
        self.angle = QLineEdit(self.default_params["angle"])
        self.grayscale = QLineEdit(self.default_params["grayscale"])
        param_layout.addWidget(QLabel("步长:"), 0, 0)
        param_layout.addWidget(self.step_length, 0, 1)
        param_layout.addWidget(QLabel("距离:"), 1, 0)
        param_layout.addWidget(self.distance, 1, 1)
        param_layout.addWidget(QLabel("角度:"), 2, 0)
        param_layout.addWidget(self.angle, 2, 1)
        param_layout.addWidget(QLabel("灰度:"), 3, 0)
        param_layout.addWidget(self.grayscale, 3, 1)
        reset_btn = QPushButton("恢复默认")
        reset_btn.clicked.connect(self.reset_parameters)
        param_layout.addWidget(reset_btn, 4, 0, 1, 2)
        main_area.addWidget(param_group, stretch=1)
        main_layout.addLayout(main_area, stretch=12)
        bottom_area = QHBoxLayout()
        left_bottom = QHBoxLayout()
        original_group = QGroupBox("原始图像")
        original_layout = QVBoxLayout(original_group)
        self.original_image = QLabel()
        self.original_image.setMinimumHeight(120)
        self.original_image.setAlignment(Qt.AlignCenter)
        self.original_image.setObjectName("originalImage")
        self.original_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        original_layout.addWidget(self.original_image)
        left_bottom.addWidget(original_group, stretch=1)
        glcm_group = QGroupBox("灰度共生矩阵")
        glcm_layout = QVBoxLayout(glcm_group)
        self.glcm_image = QLabel()
        self.glcm_image.setMinimumHeight(120)
        self.glcm_image.setAlignment(Qt.AlignCenter)
        self.glcm_image.setObjectName("glcmImage")
        self.glcm_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        glcm_layout.addWidget(self.glcm_image)
        left_bottom.addWidget(glcm_group, stretch=1)
        result_group = QGroupBox("结果数据")
        result_layout = QVBoxLayout(result_group)
        self.result_image = QLabel()
        self.result_image.setMinimumHeight(120)
        self.result_image.setAlignment(Qt.AlignCenter)
        self.result_image.setObjectName("resultData")
        self.result_image.setWordWrap(True)
        self.result_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        result_layout.addWidget(self.result_image)
        left_bottom.addWidget(result_group, stretch=1)
        bottom_area.addLayout(left_bottom, stretch=3)
        feature_group = QGroupBox("选择特征")
        feature_layout = QVBoxLayout(feature_group)
        self.select_all_checkbox = QCheckBox("全选特征")
        self.select_all_checkbox.stateChanged.connect(self.toggle_all_features)
        feature_layout.addWidget(self.select_all_checkbox)
        features = ["对比度", "同质性", "相关性", "差异性", "能量", "自相关性"]
        self.feature_checkboxes = {}
        for feature in features:
            checkbox = QCheckBox(feature)
            # 添加特征复选框状态变化的联动
            checkbox.stateChanged.connect(self.update_select_all_state)
            feature_layout.addWidget(checkbox)
            self.feature_checkboxes[feature] = checkbox
        process_button = QPushButton("处理图片")
        process_button.clicked.connect(self.process_image)
        feature_layout.addWidget(process_button)
        bottom_area.addWidget(feature_group, stretch=1)
        main_layout.addLayout(bottom_area, stretch=1)
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", "图像文件 (*.jpg *.png *.bmp)")
        if file_path:
            self.current_image = Image.open(file_path).convert("RGB")
            self.original_image_copy = self.current_image.copy()
            self.update_image_display(self.original_image, self.current_image)
            self.update_image_display(self.preview_image, self.current_image)
            self.add_to_recent_files(file_path)
    def export_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "没有图像可导出")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "导出图像", "", "图像文件 (*.jpg *.png *.bmp)")
        if file_path:
            self.current_image.save(file_path)
    def export_data(self):
        if not hasattr(self, 'last_calculated_features') or not self.last_calculated_features:
            QMessageBox.warning(self, "警告", "请先完成图像处理")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存特征数据", "", "Excel文件 (*.xlsx)")
        if file_path:
            try:
                from texture_features_extractor import save_features_to_excel
                save_features_to_excel(self.last_calculated_features, file_path)
                QMessageBox.information(self, "导出成功", "数据已保存到: " + file_path)
            except Exception as e:
                QMessageBox.critical(self, "导出错误", f"保存失败: {str(e)}")
    def export_glcm_data(self):
        if not hasattr(self, 'last_glcm') or self.last_glcm is None:
            QMessageBox.warning(self, "警告", "请先计算GLCM矩阵")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存GLCM矩阵", "", "Excel文件 (*.xlsx);;所有文件 (*)"
        )
        if path:
            try:
                import pandas as pd
                if len(self.last_glcm.shape) > 2:
                    matrix_2d = self.last_glcm[:, :, 0, 0]
                else:
                    matrix_2d = self.last_glcm
                rows, cols = matrix_2d.shape
                QMessageBox.information(self, "GLCM矩阵信息", f"GLCM矩阵大小: {rows}×{cols}\n这与您设置的灰度级别相对应")
                df = pd.DataFrame(matrix_2d)
                df.to_excel(path, index=False, header=False)
                QMessageBox.information(self, "成功", "GLCM矩阵已保存至Excel文件")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    def export_glcm_image(self):
        if not hasattr(self, 'last_glcm') or self.last_glcm is None:
            QMessageBox.warning(self, "警告", "请先计算GLCM矩阵")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存GLCM图像", "", "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)"
        )
        if path:
            try:
                from glcm_calculator import glcm_to_image
                glcm_img = glcm_to_image(self.last_glcm)
                Image.fromarray(glcm_img).save(path)
                QMessageBox.information(self, "成功", "GLCM图像已保存")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    def export_all_data(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", "")
        if not export_dir:
            return
        import os
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_folder = os.path.join(export_dir, f"纹理分析_{timestamp}")
        try:
            os.makedirs(export_folder, exist_ok=True)
            if self.original_image_copy is not None:
                original_path = os.path.join(export_folder, "原始图像.png")
                self.original_image_copy.save(original_path)
            if self.current_image is not None:
                processed_path = os.path.join(export_folder, "处理后图像.png")
                self.current_image.save(processed_path)
            if hasattr(self, 'last_glcm') and self.last_glcm is not None:
                from glcm_calculator import glcm_to_image
                glcm_img = glcm_to_image(self.last_glcm)
                glcm_path = os.path.join(export_folder, "GLCM矩阵图像.png")
                Image.fromarray(glcm_img).save(glcm_path)
                import pandas as pd
                if len(self.last_glcm.shape) > 2:
                    matrix_2d = self.last_glcm[:, :, 0, 0]
                else:
                    matrix_2d = self.last_glcm
                df = pd.DataFrame(matrix_2d)
                glcm_data_path = os.path.join(export_folder, "GLCM矩阵数据.xlsx")
                df.to_excel(glcm_data_path, index=False, header=False)
            if hasattr(self, 'last_calculated_features') and self.last_calculated_features:
                from texture_features_extractor import save_features_to_excel
                features_path = os.path.join(export_folder, "纹理特征数据.xlsx")
                save_features_to_excel(self.last_calculated_features, features_path)
            # 创建导出报告
            report_path = os.path.join(export_folder, "导出报告.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"纹理分析导出报告\n")
                f.write(f"导出时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("导出文件列表:\n")
                if self.original_image_copy is not None:
                    f.write("- 原始图像.png\n")
                if self.current_image is not None:
                    f.write("- 处理后图像.png\n")
                if hasattr(self, 'last_glcm') and self.last_glcm is not None:
                    f.write("- GLCM矩阵图像.png\n")
                    f.write("- GLCM矩阵数据.xlsx\n")
                if hasattr(self, 'last_calculated_features') and self.last_calculated_features:
                    f.write("- 纹理特征数据.xlsx\n")
                f.write("\n参数设置:\n")
                f.write(f"- 步长: {self.step_length.text()}\n")
                f.write(f"- 距离: {self.distance.text()}\n")
                f.write(f"- 角度: {self.angle.text()}\n")
                f.write(f"- 灰度: {self.grayscale.text()}\n")
            QMessageBox.information(self, "导出成功", f"所有数据已成功导出到:\n{export_folder}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出过程中发生错误: {str(e)}")
    def resample_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        # 创建对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("图像重采样")
        dialog.setMinimumWidth(350)
        layout = QVBoxLayout(dialog)
        # 添加说明标签
        info_label = QLabel(f"当前图像尺寸: {self.current_image.width} × {self.current_image.height}")
        layout.addWidget(info_label)
        # 尺寸输入
        size_layout = QFormLayout()
        width_input = QLineEdit(str(self.current_image.width))
        height_input = QLineEdit(str(self.current_image.height))
        size_layout.addRow("宽度:", width_input)
        size_layout.addRow("高度:", height_input)
        layout.addLayout(size_layout)
        # 预设尺寸按钮
        presets_label = QLabel("常用尺寸:")
        layout.addWidget(presets_label)
        presets_layout = QHBoxLayout()
        presets = [(256, 256), (512, 512), (1024, 1024)]
        for w, h in presets:
            btn = QPushButton(f"{w}×{h}")
            btn.clicked.connect(lambda checked, w=w, h=h: (width_input.setText(str(w)), height_input.setText(str(h))))
            presets_layout.addWidget(btn)
        layout.addLayout(presets_layout)
        # 比例缩放选项
        scale_label = QLabel("按比例缩放:")
        layout.addWidget(scale_label)
        scale_layout = QHBoxLayout()
        scales = [0.25, 0.5, 0.75, 1.5, 2.0]
        for scale in scales:
            btn = QPushButton(f"{scale:.2f}x")
            btn.clicked.connect(lambda checked, s=scale: (
                width_input.setText(str(int(self.current_image.width * s)) if self.current_image is not None else ""),
                height_input.setText(str(int(self.current_image.height * s)) if self.current_image is not None else "")
            ))
            scale_layout.addWidget(btn)
        layout.addLayout(scale_layout)
        buttons_layout = QHBoxLayout()
        apply_btn = QPushButton("应用")
        apply_btn.clicked.connect(lambda: self._apply_resample(width_input.text(), height_input.text(), dialog))
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        buttons_layout.addWidget(apply_btn)
        buttons_layout.addWidget(cancel_btn)
        layout.addLayout(buttons_layout)
        dialog.exec_()
    def _apply_resample(self, width_text, height_text, dialog):
        try:
            if self.current_image is None:
                QMessageBox.warning(self, "警告", "请先打开图像")
                return
            width = int(width_text)
            height = int(height_text)
            if width < 1 or height < 1:
                QMessageBox.warning(self, "参数错误", "宽度和高度必须大于0")
                return
            if width > 10000 or height > 10000:
                if QMessageBox.question(self, "确认", "尺寸过大可能导致处理缓慢，是否继续？", 
                                      QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                    return
            self.current_image = self.current_image.resize((width, height), Image.Resampling.LANCZOS)
            self.update_image_display(self.preview_image, self.current_image)
            dialog.accept()
            QMessageBox.information(self, "成功", f"已将图像重采样为 {width}×{height}")
        except ValueError:
            QMessageBox.warning(self, "参数错误", "请输入有效的整数")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重采样失败: {str(e)}")
    def process_image(self):
        """核心图像处理函数 - 执行GLCM纹理特征计算"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        try:
            try:
                step_size = int(self.step_length.text() or 1)
                distance = int(self.distance.text() or 1)
                angle_deg = float(self.angle.text() or 0.0)
                gray_levels_str = self.grayscale.text() or "256"
                gray_levels = int(gray_levels_str)
                if step_size < 1 or distance < 1:
                    raise ValueError("步长和距离必须大于0")
                if gray_levels < 2 or gray_levels > 256:
                    raise ValueError("灰度级必须在2-256之间")
                gray_levels = int(np.clip(gray_levels, 2, 256))
            except ValueError as e:
                QMessageBox.warning(self, "参数错误", f"无效参数: {str(e)}")
                return
            distances = [distance]
            angles = [np.deg2rad(angle_deg)]
        except ValueError:
            QMessageBox.warning(self, "参数错误", "请输入有效的数值参数")
            return
        # 特征选择映射 - 中文到英文
        selected_features = [f for f, cb in self.feature_checkboxes.items() if cb.isChecked()]
        feature_mapping = {
            "对比度": "contrast",
            "同质性": "homogeneity",
            "相关性": "correlation",
            "差异性": "dissimilarity",
            "能量": "energy",
            "自相关性": "autocorrelation"
        }
        english_features = [feature_mapping[chn] for chn in selected_features]
        if not english_features:
            QMessageBox.warning(self, "警告", "请至少选择一个特征")
            return
        # 图像预处理 - 转换为灰度图
        gray_image = self.current_image.convert('L')
        img_array = np.array(gray_image)
        # 灰度量化处理
        try:
            from image_loader import rescale_image
            gray_levels_int = int(gray_levels)
            quantized_array = rescale_image(img_array, gray_levels_int)
            quantized_image = Image.fromarray(quantized_array)
        except Exception as e:
            QMessageBox.warning(self, "量化错误", f"灰度量化失败: {str(e)}")
            return
        unique_values = np.unique(np.array(quantized_image))
        if len(unique_values) > gray_levels:
            QMessageBox.warning(self, "参数错误", f"量化后的不同灰度值数量({len(unique_values)})超过设定灰度级({gray_levels})")
            return
        img_array = np.array(quantized_image)
        # GLCM特征计算
        from glcm_calculator import calculate_glcm_features, glcm_to_image
        max_gray_value = int(np.max(img_array))
        glcm_levels = gray_levels
        # 灰度值范围调整
        if max_gray_value >= gray_levels:
            img_array = np.floor(img_array / (max_gray_value + 1) * gray_levels).astype(np.uint8)
            max_gray_value = int(np.max(img_array))
        # 执行GLCM计算
        features, glcm = calculate_glcm_features(
            image=img_array,
            step_size=step_size,
            levels=glcm_levels,
            distances=distances,
            angles=angles,
            properties=english_features
        )
        # 显示计算结果
        result_text = '\n'.join([f'{k}: {v:.4f}' for k, v in features.items()])
        self.result_image.clear()
        self.result_image.setText(result_text)
        self.result_image.adjustSize()
        self.last_calculated_features = features
        # GLCM矩阵可视化
        try:
            glcm_img = glcm_to_image(glcm)
            self.update_image_display(self.glcm_image, glcm_img)
            self.last_glcm = glcm
        except Exception as e:
            QMessageBox.warning(self, "GLCM可视化错误", f"无法显示灰度共生矩阵: {str(e)}")
            self.glcm_image.clear()
    def toggle_all_features(self, state):
        checked = state == Qt.Checked
        for checkbox in self.feature_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
    def update_select_all_state(self):
        all_checked = all(checkbox.isChecked() for checkbox in self.feature_checkboxes.values())
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(all_checked)
        self.select_all_checkbox.blockSignals(False)
    def reset_parameters(self):
        self.step_length.setText(self.default_params["step"])
        self.distance.setText(self.default_params["distance"])
        self.angle.setText(self.default_params["angle"])
        self.grayscale.setText(self.default_params["grayscale"])
        QMessageBox.information(self, "提示", "参数已恢复默认值")
    def reset_view(self):
        if self.current_image is not None and self.original_image_copy is not None:
            self.current_image = self.original_image_copy.copy()
            self.update_image_display(self.preview_image, self.current_image)
            self.update_image_display(self.original_image, self.current_image)
            self.result_image.setText("")
            QMessageBox.information(self, "重置成功", "已恢复到原始图像状态")
        else:
            QMessageBox.warning(self, "警告", "没有可重置的图像")
    def clear_all(self):
        self.current_image = None
        self.original_image_copy = None
        if hasattr(self, 'last_glcm'):
            self.last_glcm = None
        if hasattr(self, 'last_calculated_features'):
            self.last_calculated_features = None
        self.preview_image.clear()
        self.original_image.clear()
        self.glcm_image.clear()
        self.result_image.clear()
        self.reset_parameters()
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(False)
        QMessageBox.information(self, "清除成功", "已清空所有图像并重置参数")
    def show_usage_guide(self):
        guide_text = """
<h3>单张图像处理模块使用指南</h3>
<p><b>模块简介：</b></p>
<p>本模块基于灰度共生矩阵(GLCM)算法，提供专业的图像纹理特征分析功能，支持多种纹理参数设置和可视化输出。</p>
<p><b>核心功能：</b></p>
<ul>
  <li>支持BMP、PNG、JPG等常见图像格式处理</li>
  <li>可调节GLCM参数：步长(1-5)、距离(1-10)、角度(0°,45°,90°,135°)</li>
  <li>支持8/16/32/64/256级灰度量化</li>
  <li>提供14种纹理特征计算（对比度、能量、相关性等）</li>
  <li>矩阵数据与特征值批量导出（CSV/Excel格式）</li>
</ul>
<p><b>标准工作流程：</b></p>
<ol>
  <li>通过【文件→打开图像】或Ctrl+O导入待分析图像</li>
  <li>在参数设置面板调整GLCM参数：
    <ul>
      <li>步长：决定像素对采样间隔（默认1）</li>
      <li>距离：分析像素间距（建议1-5）</li>
      <li>角度：纹理分析方向（多角度分析建议多次处理）</li>
      <li>灰度级：量化级别（根据图像复杂度选择）</li>
    </ul>
  </li>
  <li>点击【分析→处理图像】或F5执行分析</li>
  <li>通过预览窗口检查GLCM矩阵和特征值</li>
  <li>使用导出功能保存结果：
    <ul>
      <li>Ctrl+S：保存处理后的图像</li>
      <li>Ctrl+M：导出GLCM矩阵数据</li>
      <li>Ctrl+E：导出纹理特征值表格</li>
      <li>Ctrl+A：一键导出所有结果</li>
    </ul>
  </li>
</ol>
<p><b>应用场景：</b></p>
<ul>
  <li>医学图像分析：组织病理切片纹理特征提取</li>
  <li>工业检测：材料表面缺陷纹理识别</li>
  <li>遥感监测：地表覆盖纹理分类</li>
  <li>科学研究：材料微观结构定量分析</li>
</ul>
<p><b>注意事项：</b></p>
<ul>
  <li>推荐使用512x512以上分辨率的图像</li>
  <li>大尺寸图像建议先进行重采样(Ctrl+R)</li>
  <li>多角度分析需分别设置不同角度参数</li>
  <li>导出Excel数据前请关闭相关文件</li>
</ul>
        """
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("使用指南")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(guide_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    def show_about(self):
        about_text = """
<h3>单张图像处理模块</h3>
<p>版本：1.0.0</p>
<p></p>
<p><b>研究背景：</b></p>
<p>草地生态系统作为全球陆地生态系统的重要组成部分，在调节气候、防风固沙、维持生态平衡等方面发挥着至关重要的作用。草地理化参数，如叶绿素含量、叶面积指数等是表征草地生态系统健康状态的重要指标。采用基于光谱分析的遥感技术对草地生态的监测已日益成熟。其中，光谱指数是光谱分析中常用的方法之一。随着遥感的发展，涌现出大量的光谱指数。</p>
<p><b>模块设计目的：</b></p>
<p>本模块利用灰度共生矩阵相关机器学习算法建立草地特征光谱模型，对各个光谱指数进行系统梳理，计算输出灰度共生矩阵的纹理特征值，评价其在草地叶片及冠层特征反演中的准确性。输出结果将有利于提高草地植被遥感的准确性。</p>
<p><b>主要功能：</b></p>
<ul>
  <li>草地光谱图像预处理（灰度化、重采样）</li>
  <li>基于灰度共生矩阵(GLCM)的纹理特征提取</li>
  <li>草地理化参数与光谱指数关联分析</li>
  <li>光谱指数准确性评价</li>
  <li>分析结果数据导出与可视化</li>
</ul>
        """
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("关于")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(about_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    def resizeEvent(self, a0):
        """窗口事件处理 - 处理窗口大小变化事件，重新调整所有图像显示"""
        super().resizeEvent(a0)
        from PyQt5.QtCore import QTimer
        if hasattr(self, '_resize_timer'):
            self._resize_timer.stop()
        else:
            self._resize_timer = QTimer()
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._update_all_images)
        self._resize_timer.start(250)
        self._is_resizing = True
    def _update_all_images(self):
        """批量更新函数 - 更新所有图像标签的显示"""
        image_labels = [self.preview_image, self.original_image, self.glcm_image]
        for label in image_labels:
            if not hasattr(label, 'original_image'):
                setattr(label, 'original_image', None)
                continue
            original_image = getattr(label, 'original_image', None)
            if original_image is not None:
                label_size = label.size()
                if isinstance(original_image, (Image.Image, np.ndarray)):
                    self.update_image_display(label, original_image)
        self._is_resizing = False
    def update_image_display(self, label, image):
        """核心图像显示函数 - 支持灰度/RGB图像的无损显示，使用缓存提高性能"""
        if not hasattr(label, 'original_image'):
            setattr(label, 'original_image', None)
        label.original_image = image
        if image is None:
            label.clear()
            return
        cache_key = id(label)
        label_size = label.size()
        need_new_qimage = True
        if cache_key in self._pixmap_cache:
            cached_image_id, qimage = self._pixmap_cache[cache_key]
            if cached_image_id == id(image):
                need_new_qimage = False
        if need_new_qimage:
            qimage = None
            if isinstance(image, Image.Image):
                # 确保图像模式正确
                if image.mode not in ['L', 'RGB']:
                    image = image.convert('L')  
                if image.mode == 'L':
                    qimage = QImage(
                        image.tobytes(),
                        image.width,
                        image.height,
                        image.width,  
                        QImage.Format_Grayscale8
                    )
                else:
                    qimage = QImage(
                        image.tobytes(),
                        image.width,
                        image.height,
                        image.width * 3,  
                        QImage.Format_RGB888
                    )
            elif isinstance(image, np.ndarray):
                if image.ndim == 2:
                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8) 
                    qimage = QImage(
                        image.tobytes(),
                        image.shape[1],
                        image.shape[0],
                        image.strides[0],
                        QImage.Format_Grayscale8
                    )
                elif image.ndim == 3:
                    if image.shape[2] != 3:
                        raise ValueError("RGB图像必须为3通道")
                    qimage = QImage(
                        image.tobytes(),
                        image.shape[1],
                        image.shape[0],
                        image.strides[0],
                        QImage.Format_RGB888
                    )
                else:
                    raise ValueError("不支持的图像维度")
            else:
                raise ValueError("不支持的图像类型")
            if qimage is None:
                return
            self._pixmap_cache[cache_key] = (id(image), qimage)
        else:
            _, qimage = self._pixmap_cache[cache_key]
            if qimage is None:
                return
        pixmap = QPixmap.fromImage(qimage)
        transform_mode = Qt.FastTransformation if hasattr(self, '_is_resizing') and self._is_resizing else Qt.SmoothTransformation
        pixmap_scaled = pixmap.scaled(
            label_size,
            Qt.KeepAspectRatio,
            transform_mode
        )
        label.setPixmap(pixmap_scaled)
    def open_model_builder(self):
        self.model_builder = ModelBuilderWindow()
        self.model_builder.show()
    def add_to_recent_files(self, file_path):
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
        self.update_recent_files_menu()
    def update_recent_files_menu(self):
        self.recent_files_menu.clear()
        if not self.recent_files:
            no_files_action = QAction("无最近文件", self)
            no_files_action.setEnabled(False)
            self.recent_files_menu.addAction(no_files_action)
            return
        for i, file_path in enumerate(self.recent_files):
            import os
            file_name = os.path.basename(file_path)
            action = QAction(f"{i+1}. {file_name}", self)
            action.setData(file_path)
            action.triggered.connect(lambda checked, path=file_path: self.open_recent_file(path))
            self.recent_files_menu.addAction(action)
        self.recent_files_menu.addSeparator()
        clear_action = QAction("清除最近文件列表", self)
        clear_action.triggered.connect(self.clear_recent_files)
        self.recent_files_menu.addAction(clear_action)
    def open_recent_file(self, file_path):
        try:
            self.current_image = Image.open(file_path).convert("RGB")
            self.original_image_copy = self.current_image.copy()
            self.update_image_display(self.original_image, self.current_image)
            self.update_image_display(self.preview_image, self.current_image)
            self.add_to_recent_files(file_path)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法打开文件: {str(e)}")
            if file_path in self.recent_files:
                self.recent_files.remove(file_path)
                self.update_recent_files_menu()
    def clear_recent_files(self):
        self.recent_files = []
        self.update_recent_files_menu()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageTextureAnalyzer()
    window.show()
    sys.exit(app.exec_())