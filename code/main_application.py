# -*- coding: utf-8 -*-
#主界面
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, 
                            QMessageBox, QMenu, QWidget, QVBoxLayout, QLabel, QListWidget, QHBoxLayout, QAbstractItemView, QSplitter, QFileDialog)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QSize  
from main_window import ImageTextureAnalyzer
from batch_processing import BatchAnalysisWindow
from resources import QSS_STYLE, get_icon_from_base64  
from model_inference import ModelInferenceWidget  
from PyQt5.QtWidgets import QStackedWidget
from model_inference import ModelBuilderWindow
from spectral_preprocessor import SpectralPreprocessorWidget
class MainApplicationWindow(QMainWindow):
    """模型构建主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("谱纹智瞰——纹理增强的高光谱叶面积指数反演软件")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(QSS_STYLE)
        self.setWindowIcon(get_icon_from_base64())
        self.last_selected_folder = None  
        self.workspace = QStackedWidget()
        # 侧边栏容器和布局
        sidebar_container = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        # 工具栏标题标签
        sidebar_label = QLabel("工具栏")
        sidebar_label.setAlignment(Qt.AlignCenter)
        sidebar_label.setStyleSheet("font-weight: bold; padding: 8px; background-color: #f0f0f0; border-bottom: 1px solid #cccccc;")
        sidebar_layout.addWidget(sidebar_label)
        # 侧边栏列表
        self.sidebar = QListWidget()
        self.sidebar.addItems(['单张图像分析', '批量图像分析', '光谱处理', '模型构建'])
        self.sidebar.currentRowChanged.connect(self.on_sidebar_row_changed)
        sidebar_layout.addWidget(self.sidebar)
        # 侧边栏容器固定宽度
        sidebar_container.setFixedWidth(150)
        self.init_tools()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar_container)
        splitter.addWidget(self.workspace)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)
        self.create_main_menu()
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('就绪')
    # 初始化四个核心工具模块
    def init_tools(self):
        self.single_image_tool = ImageTextureAnalyzer()
        self.batch_image_tool = BatchAnalysisWindow()
        self.spectral_tool = SpectralPreprocessorWidget()
        self.model_builder_window = ModelBuilderWindow()
        self.workspace.addWidget(self.single_image_tool)
        self.workspace.addWidget(self.batch_image_tool)
        self.workspace.addWidget(self.spectral_tool)
        self.workspace.addWidget(self.model_builder_window)
        self.texture_analyzer = None
    # 创建主菜单栏和工具菜单
    def create_main_menu(self):
        """创建顶部菜单栏"""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件管理")
        import_action = QAction("选择文件夹", self)
        import_action.triggered.connect(self.import_images)
        file_menu.addAction(import_action)
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        tools_menu = menubar.addMenu("工具")
        # 单张图像分析工具
        single_image_action = QAction("单张图像分析", self)
        single_image_action.triggered.connect(self.show_single_analysis)
        tools_menu.addAction(single_image_action)
        # 批量图像分析工具
        batch_image_action = QAction("批量图像分析", self)
        batch_image_action.triggered.connect(self.show_batch_analysis)
        tools_menu.addAction(batch_image_action)
        # 光谱处理工具
        spectral_action = QAction("光谱处理", self)
        spectral_action.triggered.connect(self.show_spectral_preprocessor)
        tools_menu.addAction(spectral_action)
        # 模型构建工具
        model_builder_action = QAction("模型构建", self)
        model_builder_action.triggered.connect(self.show_model_builder)
        tools_menu.addAction(model_builder_action)
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        usage_action = QAction("使用说明", self)
        usage_action.triggered.connect(self.show_usage_instructions)
        help_menu.addAction(usage_action)
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    # 处理侧边栏工具切换事件
    def on_sidebar_row_changed(self, index):
        """处理侧边栏切换事件"""
        if index >= 0 and index < self.workspace.count():
            self.workspace.setCurrentIndex(index)
    def show_model_builder(self):
        """显示模型构建工具窗口"""
        if not hasattr(self, 'model_inference_window') or not self.model_inference_window:
            self.model_inference_window = ModelBuilderWindow()
        self.model_inference_window.show()
    def show_spectral_preprocessor(self):
        """显示光谱预处理独立窗口"""
        if not hasattr(self, 'spectral_window') or not self.spectral_window:
            self.spectral_window = SpectralPreprocessorWidget()
        self.spectral_window.show()
    def show_single_analysis(self):
        """显示单张图像分析界面"""
        if not self.texture_analyzer:
            self.texture_analyzer = ImageTextureAnalyzer()
        self.texture_analyzer.show()
    def show_batch_analysis(self):
        """显示批量分析界面"""
        if not hasattr(self, 'batch_window') or not self.batch_window:
            self.batch_window = BatchAnalysisWindow()
        self.batch_window.show()
    def show_selected_image(self, item):
        file_path = item.data(Qt.UserRole)
        if file_path and os.path.exists(file_path):
            self.status_bar.showMessage(f'正在加载文件：{file_path}')
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                pixmap = QPixmap(file_path)
                if not pixmap.isNull(): 
                    scaled_pixmap = pixmap.scaled(
                        self.image_viewer.width(), 
                        self.image_viewer.height(), 
                        Qt.KeepAspectRatio
                    )
                    self.image_viewer.setPixmap(scaled_pixmap)
                else:
                    QMessageBox.warning(self, '错误', '无法加载图像文件')
            else:
                try:
                    os.startfile(file_path)
                except Exception as e:
                    QMessageBox.warning(self, '错误', f'无法打开文件: {str(e)}')
            self.status_bar.showMessage(f'文件已处理：{file_path}')
        self.status_bar.showMessage(f'文件已加载：{file_path}')
    def import_images(self):
        """通过文件对话框导入文件夹并添加其中的图像到当前选中工具的列表"""
        from PyQt5.QtWidgets import QFileDialog
        initial_dir = self.last_selected_folder if self.last_selected_folder else os.getcwd()
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择图像文件夹", initial_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if not folder_path:
            return
    def export_current_image(self):
        """导出当前显示的图像"""
        current_pixmap = self.image_viewer.pixmap()
        if current_pixmap:
            file_path, _ = QFileDialog.getSaveFileName(self, '导出图像', '', '图像文件 (*.png *.jpg)')
            if file_path:
                current_pixmap.save(file_path)
    def adjust_zoom(self, factor: float):
        current_pixmap = self.image_viewer.pixmap()
        if current_pixmap:
            new_width = current_pixmap.width() * factor
            new_height = current_pixmap.height() * factor
            scaled_pixmap = current_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio)
            self.image_viewer.setPixmap(scaled_pixmap)
    def clear_imported_files(self):
        """清除当前工具已导入的文件列表及显示内容"""
        current_tool = self.workspace.currentWidget()
        if not current_tool or not hasattr(current_tool, 'image_list'):
            QMessageBox.warning(self, '错误', '当前工具界面不支持文件清除')
            return
        current_tool.image_list.clear()
        if hasattr(current_tool, 'image_viewer'):
            current_tool.image_viewer.clear()
        self.status_bar.showMessage('已成功清除当前工具的所有已导入文件')
    # 显示软件使用指南
    def show_usage_instructions(self):
        """显示软件使用说明"""
        usage_text = """
<h3>纹理增强的高光谱叶面积指数反演软件使用指南</h3>
<p><b>软件简介：</b></p>
<p>本软件集成图像纹理分析、批量处理、光谱预处理和模型构建四大功能模块，支持从图像特征提取到预测模型构建的完整工作流程。</p>
<p><b>核心功能：</b></p>
<ul>
  <li><b>单张图像分析</b>：基于灰度共生矩阵(GLCM)提取纹理特征，支持参数自定义和可视化结果展示</li>
  <li><b>批量图像分析</b>：批量处理多幅图像，统一参数设置，结果汇总导出</li>
  <li><b>光谱处理</b>：光谱数据预处理与特征提取，支持多种滤波和变换算法</li>
  <li><b>模型构建</b>：基于提取的特征训练预测模型，支持多种机器学习算法</li>
</ul>
<p><b>基本操作流程：</b></p>
<ol>
  <li>通过顶部菜单栏【文件→选择文件】或工具栏导入数据
    <ul>
      <li>图像数据支持JPG、PNG、BMP等格式</li>
      <li>光谱数据支持CSV格式</li>
    </ul>
  </li>
  <li>通过左侧边栏或顶部【工具】菜单选择功能模块</li>
  <li>根据模块特点设置相应参数并执行分析</li>
  <li>通过【文件】菜单导出结果数据或图像</li>
</ol>
<p><b>快捷键说明：</b></p>
<ul>
  <li>Ctrl+O：导入文件</li>
  <li>Ctrl+S：保存当前结果</li>
  <li>Ctrl+E：导出数据</li>
  <li>F5：执行分析</li>
  <li>Alt+F4：退出程序</li>
</ul>
<p><b>注意事项：</b></p>
<ul>
  <li>处理大尺寸图像时建议先进行重采样</li>
  <li>光谱数据需保持统一格式和采样率</li>
  <li>模型构建前请确保已准备好特征数据</li>
  <li>导出Excel数据前请关闭相关文件</li>
</ul>
        """
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("使用指南")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(usage_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    # 显示软件关于信息
    def show_about(self):
        """显示软件关于信息"""
        about_text = """
<h3>纹理增强的高光谱叶面积指数反演软件 v1.0</h3>
<p><b>软件背景：</b></p>
<p>本工具旨在为草地生态系统研究提供从图像与光谱数据到模型构建的完整解决方案，集成了特征提取、数据预处理和模型训练等功能，帮助研究人员高效分析生态系统特征。</p>
<p><b>主要功能模块：</b></p>
<ul>
  <li><b>图像纹理分析</b>：基于灰度共生矩阵(GLCM)算法提取14种纹理特征</li>
  <li><b>批量处理</b>：支持多图像自动化分析与结果汇总</li>
  <li><b>光谱预处理</b>：提供平滑、去噪、归一化等光谱数据预处理功能</li>
  <li><b>模型构建</b>：集成多种机器学习算法，支持模型训练与评估</li>
</ul>
<p><b>应用领域：</b></p>
<ul>
  <li>草地生态系统特征反演</li>
  <li>遥感图像纹理分析</li>
  <li>植被光谱特征提取</li>
  <li>生态模型构建与预测</li>
</ul>
        """
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("关于软件")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(about_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "qt.qpa=off"  
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS_STYLE)  
    app.setWindowIcon(get_icon_from_base64())  
    window = MainApplicationWindow()
    window.show()
    sys.exit(app.exec_())