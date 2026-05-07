from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QFileDialog, QListWidget, QLabel, QLineEdit, QTableWidget,
                            QHeaderView, QComboBox, QMessageBox, QApplication,
                            QMenuBar, QMenu, QAction)  # 新增菜单相关组件
import sys 
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QIntValidator
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import pandas as pd
import pywt
from PyQt5.QtWidgets import (QTableWidgetItem)  # 添加正确类名导入
from PyQt5.QtWidgets import QMessageBox
from resources import QSS_STYLE  # 新增导入样式表
from resources import QSS_STYLE, get_icon_from_base64  # 新增图标导入
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtWidgets import QDialog  # 新增导入
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from copy import deepcopy

class ParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置窗口参数")
        layout = QVBoxLayout()
        
        self.window_size = QLineEdit("21")
        self.window_size.setValidator(QIntValidator(3, 101))
        layout.addWidget(QLabel("窗口大小:"))
        layout.addWidget(self.window_size)
        
        btn_confirm = QPushButton("确认")
        btn_confirm.clicked.connect(self.accept)
        layout.addWidget(btn_confirm)
        
        self.setLayout(layout)

class SpectralPreprocessorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("光谱预处理")
        self.spectral_data = None
        self.processed_data = None
        self.setStyleSheet(QSS_STYLE)  # 应用全局样式表
        self.setWindowIcon(get_icon_from_base64())  # 新增窗口图标设置
        self.init_menu()
        self.init_ui()
        self.resize(800, 600)

    def init_menu(self):
        menu_bar = QMenuBar(self)
        
        # 文件菜单
        file_menu = menu_bar.addMenu("文件")
        file_menu.addAction("导入", self.load_spectral_data)
        file_menu.addAction("导出", self.export_data)

        # 视图菜单
        view_menu = menu_bar.addMenu("视图")
        view_menu.addAction("清除数据", self.clear_data)
        view_menu.addAction("显示预处理后曲线图", self.show_processed_plot) 
        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助")
        help_menu.addAction("使用文档", self.show_help)

    def show_processed_plot(self):
        # 添加中文字体配置
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        if self.processed_data is None:
            QMessageBox.warning(self, "警告", "请先执行预处理操作")
            return

        # 转换为二维数组（处理单样本情况）
        plot_data = np.atleast_2d(self.processed_data)
        
        # 创建绘图窗口
        plot_dialog = QDialog(self)
        plot_dialog.setWindowTitle("预处理结果曲线图")
        plot_dialog.resize(800, 600)
        
        # 创建画布
        figure = Figure()
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)
        
        # 设置全局字体
        for text_obj in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            text_obj.set_fontname('SimHei')
        
        # 绘制所有样本的光谱曲线
        for i in range(plot_data.shape[0]):
            ax.plot(self.spectral_data.columns.astype(float), 
                   plot_data[i], 
                   linewidth=0.5)
        
        # 设置坐标轴标签
        ax.set_xlabel('波长(nm)', fontname='SimHei')
        ax.set_ylabel('反射率', fontname='SimHei')
        method = self.cmb_method.currentText()  # 新增获取当前方法
        
        # 修改标题设置
        ax.set_title(f'{method}处理结果', fontname='SimHei')  # 动态显示方法名称
        
        # 添加到布局
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        plot_dialog.setLayout(layout)
        plot_dialog.exec_()

    def clear_data(self):
        """
        清除所有加载和处理过的数据
        重置界面显示状态
        """
        self.spectral_data = None
        self.processed_data = None
        self.lbl_file.setText("未选择文件")
        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(0)
        QMessageBox.information(self, "提示", "已清除所有数据")

    def init_ui(self):
        # 创建菜单栏
        menu_bar = QMenuBar()
        
        # 文件菜单
        file_menu = menu_bar.addMenu("文件")
        file_menu.addAction("导入文件", self.load_spectral_data)
        file_menu.addAction("导出文件", self.export_data)

        # 视图菜单
        view_menu = menu_bar.addMenu("视图")
        view_menu.addAction("清除数据", self.clear_data)
        view_menu.addAction("显示预处理后曲线图", self.show_processed_plot)

        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助")
        help_menu.addAction("使用文档", self.show_help)

        # 将菜单栏添加到布局
        layout = QVBoxLayout()
        layout.setMenuBar(menu_bar)
        
        # 文件操作区域
        file_layout = QHBoxLayout()
        self.btn_load = QPushButton("加载光谱数据")
        self.btn_load.clicked.connect(self.load_spectral_data)
        self.lbl_file = QLabel("未选择文件")
        file_layout.addWidget(self.btn_load)
        file_layout.addWidget(self.lbl_file)
        layout.addLayout(file_layout)

        # 预处理方法选择控件
        param_layout = QHBoxLayout()
        self.cmb_method = QComboBox()  # 确保创建组合框实例
        self.cmb_method.addItems([
            "多元散射校正(MSC)", "标准正态变换(SNV)", "移动平均平滑(MA)",
            "Savitzky-Golay平滑滤波(SG)", "趋势校正(DT)", "小波变换(WT)",
            "均值中心化(CT)", "最大最小值归一化(MMS)", "稳健标准化(RS)",
            "最大绝对值归一化(MAS)"
        ])
        param_layout.addWidget(QLabel("预处理方法:"))
        param_layout.addWidget(self.cmb_method)  # 确保添加到布局
        layout.addLayout(param_layout)

        # 操作按钮
        btn_process = QPushButton("执行预处理")
        btn_process.clicked.connect(self.process_data)
        layout.addWidget(btn_process) 
        layout.addWidget(btn_process)

        # 结果展示表格
        self.result_table = QTableWidget()
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.result_table)

        self.setLayout(layout)
        
    @staticmethod
    def MMS(data):
        return MinMaxScaler().fit_transform(data)

    @staticmethod
    def SS(data):
        return StandardScaler().fit_transform(data)

    @staticmethod 
    def RS(data):
        return RobustScaler().fit_transform(data)

#最大绝对值归一化
    @staticmethod
    def MAS(data):
        return MaxAbsScaler().fit_transform(data)

# 均值中心化
    @staticmethod
    def CT(data):
        n, p = data.shape
        data_CT = np.ones((n, p))
        for i in range(data.shape[0]):
            MEAN = np.mean(data[i])
            data_CT[i] = data[i] - MEAN
        return data_CT

# 标准正态变换
    # 修改SNV方法返回类型
    @staticmethod
    def SNV(data):
        m = data.shape[0]
        n = data.shape[1]
        data_std = np.std(data, axis=1)
        data_average = np.mean(data, axis=1)
        data_SNV = np.array([(data[i] - data_average[i]) / data_std[i] for i in range(m)])
        return data_SNV

    @staticmethod
    def MA(data, w):
        n, p = data.shape
        data_MA = np.ones((n, p))
        for i in range(data.shape[0]):
            out0 = np.convolve(data[i], np.ones(w, dtype=int), 'valid') / w
            r = np.arange(1, w - 1, 2)
            start = np.cumsum(data[i, :w - 1])[::2] / r
            stop = (np.cumsum(data[i, :-w:-1])[::2] / r)[::-1]
            data_MA[i] = np.concatenate((start, out0, stop))
        return data_MA

    @staticmethod
    def SG(data, w, p):
        return signal.savgol_filter(data, w, p)

    @staticmethod
    def DT(data):
        n, p = data.shape
        x = np.arange(p).reshape(-1, 1)
        out = np.array(data)
        l = LinearRegression()
        for i in range(out.shape[0]):
            l.fit(x, out[i].reshape(-1, 1))
            k = l.coef_
            b = l.intercept_
            for j in range(out.shape[1]):
                out[i][j] = out[i][j] - (j * k + b)
        return out

    @staticmethod
    def WT(data):
        data = deepcopy(data)
        if isinstance(data, pd.DataFrame):
            data = data.values

        def wave_(data):
            w = pywt.Wavelet('db8')  # 选用Daubechies8小波
            maxlev = pywt.dwt_max_level(len(data), w.dec_len)
            coeffs = pywt.wavedec(data, 'db8', level=maxlev)
            threshold = 256
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
            datarec = pywt.waverec(coeffs, 'db8')
            return datarec

        tmp = None
        for i in range(data.shape[0]):
            if (i == 0):
                tmp = wave_(data[i])
            else:
                tmp = np.vstack((tmp, wave_(data[i])))

        return tmp[:, :data.shape[1]]

    @staticmethod
    def MSC(data):
        n, p = data.shape
        mean = np.mean(data, axis=0)
        data_MSC = np.ones((n, p))

        for j in range(n):
            mean = np.mean(data, axis=0)

        # 线性拟合
        for i in range(n):
            y = data[i, :]
            l = LinearRegression()
            l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
            k = l.coef_
            b = l.intercept_
            data_MSC[i, :] = (y - b) / k
        return data_MSC


    def process_data(self):
        if self.spectral_data is None:
            QMessageBox.warning(self, "警告", "请先加载光谱数据")
            return
        
        try:
            method = self.cmb_method.currentText()
            data = self.spectral_data.values.astype(np.float64)
            window = 21

            if method in ["移动平均平滑(MA)", "Savitzky-Golay平滑滤波(SG)"]:
                dialog = ParameterDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    window = int(dialog.window_size.text())
                else:
                    return

            # 统一处理所有方法分支
            if method == "移动平均平滑(MA)":
                processed = self.MA(data, w=window)
            elif method == "Savitzky-Golay平滑滤波(SG)":
                processed = self.SG(data, w=window, p=3)
            elif method == "趋势校正(DT)":
                processed = self.DT(data)
            elif method == "小波变换(WT)":
                processed = self.WT(data)
            elif method == "均值中心化(CT)":
                processed = self.CT(data)
            elif method == "最大最小值归一化(MMS)":
                processed = self.MMS(data)
            elif method == "稳健标准化(RS)":
                processed = self.RS(data)
            elif method == "最大绝对值归一化(MAS)":
                processed = self.MAS(data)
            elif method == "多元散射校正(MSC)":
                processed = self.MSC(data)  # 注意大写
            elif method == "标准正态变换(SNV)":
                processed = self.SNV(data)
            else:
                raise ValueError("未知的预处理方法")

            # 后续处理保持不变
            processed_df = pd.DataFrame(processed, columns=self.spectral_data.columns)
            self.show_processed_data(processed_df)
            self.processed_data = processed
            QMessageBox.information(self, "完成", "预处理已完成")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败: {str(e)}")

    def show_processed_data(self, data):
        self.result_table.setRowCount(data.shape[0])
        self.result_table.setColumnCount(data.shape[1])
        # 新增：设置列标签为波段波长（原始数据的列名）
        self.result_table.setHorizontalHeaderLabels(data.columns.astype(str))
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.result_table.setItem(i, j, QTableWidgetItem(f"{data.iloc[i,j]:.4f}"))

        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.result_table.horizontalHeader().setMinimumSectionSize(120)


    def load_spectral_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择光谱文件", "", "CSV文件 (*.csv);;Excel文件 (*.xlsx)")
        if path:
            try:
                if path.endswith('.csv'):
                    self.spectral_data = pd.read_csv(path)
                else:
                    self.spectral_data = pd.read_excel(path)
                
                # 新增：验证列名是否为100-1000之间的波长数值
                columns = self.spectral_data.columns
                valid = True
                for col in columns:
                    try:
                        wavelength = float(col)
                        if not (100 <= wavelength <= 1000):
                            valid = False
                            break
                    except ValueError:
                        valid = False
                        break
                
                if not valid:
                    raise ValueError("导入的数据包含非光谱数据列。请确保第一行所有列名均为100-1000之间的波长数值。")
                
                self.lbl_file.setText(f"已加载: {path.split('/')[-1]}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"文件读取失败: {str(e)}")
                self.spectral_data = None  # 清除无效数据
                self.lbl_file.setText("未选择文件")

    def export_data(self):
        if not hasattr(self, 'processed_data'):
            QMessageBox.warning(self, "警告", "请先执行预处理操作")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存处理结果",
            "",
            "Excel文件 (*.xlsx);;CSV文件 (*.csv)"
        )
        
        if path:
            try:
                # 修改：使用原始列名作为导出文件的列标签
                df = pd.DataFrame(self.processed_data, columns=self.spectral_data.columns)
                if path.endswith('.xlsx'):
                    df.to_excel(path, index=False)
                else:
                    df.to_csv(path, index=False)
                QMessageBox.information(self, "成功", "文件已保存至：" + path)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def show_help(self):
        """显示完整的帮助文档"""
        help_content = """
        <h3>光谱预处理模块使用指南</h3>

        <p><b>1. 文件操作</b></p>
        <ul>
           <li><b>导入</b>: 支持CSV/Excel格式(列表示波长，行表示样本)</li>
           <li><b>导出</b>: 处理结果保存为相同格式，保留原始波长标签</li>
        </ul>

        <p><b>2. 预处理步骤</b></p>
        <ol>
           <li>通过「文件」→「导入」加载光谱数据</li>
           <li>从下拉菜单选择预处理方法</li>
           <li>对于移动平均(MA)和SG平滑，点击「执行预处理」后设置窗口大小(3-101之间的奇数)</li>
           <li>点击「执行预处理」按钮开始计算</li>
           <li>通过「视图」→「显示预处理后曲线图」查看结果</li>
        </ol>

        <p><b>3. 支持的预处理方法</b></p>
        <ul>
           <li>多元散射校正(MSC)</li>
           <li>标准正态变换(SNV)</li>
           <li>移动平均平滑(MA)</li>
           <li>Savitzky-Golay平滑滤波(SG)</li>
           <li>趋势校正(DT)</li>
           <li>小波变换(WT)</li>
           <li>均值中心化(CT)</li>
           <li>最大最小值归一化(MMS)</li>
           <li>稳健标准化(RS)</li>
           <li>最大绝对值归一化(MAS)</li>
        </ul>

        <p><b>4. 常见问题</b></p>
        <ul>
           <li>数据加载失败: 检查文件格式和编码，确保第一行是波长标签</li>
           <li>处理报错: 确认数据不含非数值内容，删除空行或文本注释</li>
           <li>曲线图显示异常: 确保数据维度一致，样本数不超过1000</li>
        </ul>
        """
        QMessageBox.information(self, "帮助文档", help_content.strip())
    

if __name__ == "__main__":
    app = QApplication(sys.argv)  
    window = SpectralPreprocessorWidget()  
    window.show() 
    sys.exit(app.exec_())
