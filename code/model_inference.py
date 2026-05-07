#模型构建工具
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, 
                             QTableWidget, QTableWidgetItem, QComboBox, QCheckBox, QLabel, 
                             QMessageBox, QGroupBox, QHeaderView,QGridLayout,QScrollArea, QCheckBox)  
from PyQt5.QtCore import Qt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.pipeline import make_pipeline  
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import Ridge
from PyQt5.QtWidgets import QScrollArea, QCheckBox, QLineEdit
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT  
from PyQt5.QtWidgets import QMainWindow  
from resources import QSS_STYLE
# 机器学习模型推理和构建组件
class ModelInferenceWidget(QWidget):
    """模型推理主窗口 - 集成多种机器学习算法进行LAI预测"""
    def __init__(self):
        super().__init__()
        self.data = None  
        self.feature_checkboxes = {}  
        self.model_checkboxes = {}   
        self.train_test_ratio = 0.8  
        self.model_results = {}  
        self.spectral_features = []  
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self._create_menu()
        self.models = {
            'Linear Regression': LinearRegression(),
            'Polynomial Regression (degree=2)': make_pipeline(PolynomialFeatures(2), LinearRegression()),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0, random_state=42),
            'PLSR': PLSRegression(),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=False),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'SVR': make_pipeline(StandardScaler(), SVR()),
            'Neural_Network': make_pipeline(StandardScaler(), MLPRegressor(random_state=42, max_iter=1000))
        }
        self.init_ui()  
        self.data = None  
        self.features = []  
        self.selected_models = []  
        self.scaler = StandardScaler()  
        self.init_models() 
    # 初始化用户界面布局
    def init_ui(self):
        # 清除现有布局内容
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        feature_model_layout = QHBoxLayout()  
        feature_group = QGroupBox("特征选择")
        feature_layout = QVBoxLayout()  
        wavelength_layout = QHBoxLayout()
        wavelength_layout.addWidget(QLabel("光谱波长范围:"))
        self.min_wavelength = QLineEdit()
        self.min_wavelength.setPlaceholderText("最小波长")
        self.min_wavelength.setFixedWidth(80)
        wavelength_layout.addWidget(self.min_wavelength)
        wavelength_layout.addWidget(QLabel("-"))
        self.max_wavelength = QLineEdit()
        self.max_wavelength.setPlaceholderText("最大波长")
        self.max_wavelength.setFixedWidth(80)
        wavelength_layout.addWidget(self.max_wavelength)
        self.apply_wavelength_btn = QPushButton("应用")
        self.apply_wavelength_btn.clicked.connect(self.apply_wavelength_range)
        wavelength_layout.addWidget(self.apply_wavelength_btn)
        wavelength_layout.addSpacing(20) 
        wavelength_layout.addSpacing(20)  
        wavelength_layout.addWidget(QLabel("多波段选择:"))
        self.multi_bands = QLineEdit()
        self.multi_bands.setPlaceholderText("例如: 400,500-600")
        self.multi_bands.setFixedWidth(150)
        wavelength_layout.addWidget(self.multi_bands)
        self.apply_multi_bands_btn = QPushButton("应用多波段")
        self.apply_multi_bands_btn.clicked.connect(self.apply_multi_bands)
        wavelength_layout.addWidget(self.apply_multi_bands_btn)
        wavelength_layout.addStretch()  
        feature_layout.addLayout(wavelength_layout)
        features_container_layout = QHBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.feature_checkboxes = {}
        self.feature_scroll_layout = QGridLayout(scroll_content)
        self.feature_scroll_layout.setColumnStretch(0, 1)
        self.feature_scroll_layout.setColumnStretch(1, 1)
        self.feature_scroll_layout.setColumnStretch(2, 1)
        self.feature_scroll_layout.setHorizontalSpacing(10)
        self.feature_scroll_layout.setVerticalSpacing(5)
        features_container_layout.addLayout(self.feature_scroll_layout, stretch=1)
        features = self.data.columns.tolist() if self.data is not None else []
        for index, feature_name in enumerate(features):
            row = index // 3
            col = index % 3
            cb = QCheckBox(feature_name)
            self.feature_checkboxes[feature_name] = cb
            self.feature_scroll_layout.addWidget(cb, row, col, Qt.AlignLeft)
        scroll.setWidget(scroll_content)
        features_container_layout.addWidget(scroll, stretch=1)  
        feature_layout.addLayout(features_container_layout)
        feature_group.setLayout(feature_layout)
        feature_model_layout.addWidget(feature_group, stretch=1)
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()  
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.model_checkboxes = {}  
        scroll_layout = QGridLayout(scroll_content)
        scroll_layout.setColumnStretch(0, 1)
        scroll_layout.setColumnStretch(1, 1)
        scroll_layout.setColumnStretch(2, 1)
        scroll_layout.setColumnStretch(3, 1)  # 新增第四列的拉伸设置
        scroll_layout.setHorizontalSpacing(10)
        scroll_layout.setVerticalSpacing(5)
        model_names = list(self.models.keys())
        for index, model_name in enumerate(model_names):
            row = index // 4  # 计算行号（每4个一行）
            col = index % 4   # 计算列号（0-3列）
            cb = QCheckBox(model_name)
            self.model_checkboxes[model_name] = cb
            scroll_layout.addWidget(cb, row, col, Qt.AlignLeft)
        scroll.setWidget(scroll_content)
        model_layout.addWidget(scroll)
        model_group.setLayout(model_layout)
        feature_model_layout.addWidget(model_group, stretch=1)  
        self.main_layout.addLayout(feature_model_layout)
        button_layout = QHBoxLayout()
        self.correlation_btn = QPushButton("correlation analysis")
        self.correlation_btn.clicked.connect(self.show_correlation_heatmap)  
        button_layout.addWidget(self.correlation_btn)
        self.run_btn = QPushButton("model building")
        self.run_btn.clicked.connect(self.run_inference)
        button_layout.addWidget(self.run_btn)
        self.main_layout.addLayout(button_layout)  
        results_layout = QHBoxLayout()
        self.main_layout.addLayout(results_layout)
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels(["模型", "特征组合", "训练RMSE", "测试RMSE", "训练R²", "测试R²"])
        header = self.result_table.horizontalHeader()
        for col in range(6):
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        results_layout.addWidget(self.result_table, stretch=1)
        try:
            self.result_table.itemClicked.disconnect()  
        except TypeError:
            pass  
        self.result_table.itemClicked.connect(self.on_table_row_clicked)
        # 2. 图表区域初始化
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.canvas = FigureCanvas(self.fig)
        # 创建导航工具栏
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # 使用垂直布局容纳工具栏+画布
        chart_layout = QVBoxLayout()
        chart_layout.addWidget(self.toolbar) 
        chart_layout.addWidget(self.canvas)  
        results_layout.addLayout(chart_layout, stretch=1)  
        export_layout = QHBoxLayout()
        export_table_btn = QPushButton("导出表格")
        export_table_btn.clicked.connect(self.export_table)
        export_layout.addWidget(export_table_btn)
        export_scatter_btn = QPushButton("导出散点图")
        export_scatter_btn.clicked.connect(self.export_scatter)
        export_layout.addWidget(export_scatter_btn)
        self.main_layout.addLayout(export_layout)  
    # 评估模型性能指标
    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        return train_rmse, test_rmse, train_r2, test_r2, y_test_pred
    def _get_model(self, model_name):
        if model_name == 'Linear Regression':
            return LinearRegression()
        elif model_name == 'Polynomial Regression (degree=2)':
            return make_pipeline(PolynomialFeatures(2), LinearRegression())
        elif model_name == 'Random Forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == 'XGBoost':
            return xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0, random_state=42)
        elif model_name == 'PLSR':
            return PLSRegression()
        elif model_name == 'CatBoost':
            return CatBoostRegressor(random_state=42, verbose=False)
        elif model_name == 'Extra Trees':
            return ExtraTreesRegressor(n_estimators=100, random_state=42)
        elif model_name == 'Gradient Boosting':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_name == 'AdaBoost':
            return AdaBoostRegressor(n_estimators=100, random_state=42)
        elif model_name == 'Ridge Regression':
            return Ridge(alpha=1.0, random_state=42)
        elif model_name == 'SVR':
            return make_pipeline(StandardScaler(), SVR())
        elif model_name == 'Neural_Network':
            return make_pipeline(StandardScaler(), MLPRegressor(random_state=42, max_iter=1000))
        return None
    def load_data(self):
        """数据加载函数 - 处理CSV文件导入和特征识别"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV文件 (*.csv)")
        if not file_path:
            return
        try:
            self.data = pd.read_csv(file_path)
            if 'LAI' not in self.data.columns:
                QMessageBox.critical(self, "错误", "文件缺少LAI数据，无法导入！")
                self.data = None
                return
            numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
            self.spectral_features = []
            self.non_spectral_features = []
            for col in numeric_cols:
                if col == 'LAI':
                    continue
                try:
                    wavelength = float(col)
                    if 100 <= wavelength <= 1000:
                        self.spectral_features.append(col)
                    else:
                        self.non_spectral_features.append(col)
                except ValueError:
                    self.non_spectral_features.append(col)
            # 使用非光谱特征作为普通特征
            self.features = self.non_spectral_features 
            for cb in self.feature_checkboxes.values():
                cb.deleteLater()
            self.feature_checkboxes.clear()
            for feature in self.features:
                cb = QCheckBox(feature)
                self.feature_checkboxes[feature] = cb
                self.feature_scroll_layout.addWidget(cb)  
            QMessageBox.information(self, "成功", f"数据加载完成，发现{len(self.spectral_features)}个光谱特征")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据失败: {str(e)}")
    def run_inference(self):
        """模型训练主函数 - 执行多模型批量训练和评估"""
        if self.data is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return
        selected_features = [name for name, cb in self.feature_checkboxes.items() if cb.isChecked()]
        if hasattr(self, 'selected_spectral_features'):
            selected_features.extend(self.selected_spectral_features)
        selected_models = [name for name, cb in self.model_checkboxes.items() if cb.isChecked()]
        if not selected_features:
            QMessageBox.warning(self, "警告", "请至少选择一个特征或设置光谱范围")
            return
        if not selected_models:
            QMessageBox.warning(self, "警告", "请至少选择一个模型")
            return
        X = self.data[selected_features]
        y = self.data['LAI']
        X_scaled = self.scaler.fit_transform(X)
        test_size = 1 - self.train_test_ratio  
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        for model_name in selected_models:
            model = self._get_model(model_name)
            if model is None:
                continue
            train_rmse, test_rmse, train_r2, test_r2, y_test_pred = self.evaluate_model(
                model, X_train, X_test, y_train, y_test)
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QTableWidgetItem(model_name))
            features_str = ",".join(selected_features)
            self.result_table.setItem(row, 1, QTableWidgetItem(features_str))
            self.result_table.setItem(row, 2, QTableWidgetItem(f"{train_rmse:.2f}"))
            self.result_table.setItem(row, 3, QTableWidgetItem(f"{test_rmse:.2f}"))
            self.result_table.setItem(row, 4, QTableWidgetItem(f"{train_r2:.2f}"))
            self.result_table.setItem(row, 5, QTableWidgetItem(f"{test_r2:.2f}"))
            y_train_pred = model.predict(X_train)
            result_key = f"{model_name}_{features_str}"
            self.model_results[result_key] = {
                'y_train': y_train,
                'y_train_pred': y_train_pred,
                'y_test': y_test,
                'y_test_pred': y_test_pred,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'model_name': model_name
            }
        if selected_models:
            last_model_name = selected_models[-1]
            last_key = f"{last_model_name}_{','.join(selected_features)}"
            if last_key in self.model_results:
                self.update_scatter_plot(self.model_results[last_key])
    def init_models(self):
        """初始化模型列表（移除冗余的model_combo操作）"""
        self.result_table.setSortingEnabled(True) 
        self.result_table.setAlternatingRowColors(True)  
        self.result_table.horizontalHeader().setStretchLastSection(True)  
    def export_table(self):
        """导出表格数据到CSV"""
        if self.result_table.rowCount() == 0:
            QMessageBox.warning(self, "警告", "没有可导出的数据")
            return
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "保存表格", "", "CSV文件 (*.csv)", options=options)
        if file_name:
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    
                    headers = [self.result_table.horizontalHeaderItem(i).text() for i in range(self.result_table.columnCount())]
                    f.write(','.join(headers) + '\n')
                    for row in range(self.result_table.rowCount()):
                        row_data = []
                        for col in range(self.result_table.columnCount()):
                            item = self.result_table.item(row, col)
                            row_data.append(item.text() if item else '')
                        f.write(','.join(row_data) + '\n')
                QMessageBox.information(self, "成功", "表格导出成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    def export_scatter(self):
        """导出散点图为图片"""
        if not hasattr(self, 'ax') or len(self.ax.collections) == 0:
            QMessageBox.warning(self, "警告", "没有可导出的散点图")
            return
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "保存散点图", "", "PNG文件 (*.png);;JPEG文件 (*.jpg)", options=options)
        if file_name:
            try:
                self.fig.savefig(file_name, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "成功", "散点图导出成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    def export_merged_data(self):
        """导出合并后的数据"""
        if self.data is None:
            QMessageBox.warning(self, "警告", "没有可导出的合并数据")
            return
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "保存合并数据", "", "CSV文件 (*.csv)", options=options)
        if file_name:
            try:
                self.data.to_csv(file_name, index=False, encoding='utf-8')
                QMessageBox.information(self, "成功", "合并数据导出成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    def show_correlation_heatmap(self):
        """显示特征相关性热力图（关键新增方法）"""
        if self.data is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return
        selected_features = [name for name, cb in self.feature_checkboxes.items() if cb.isChecked()]
        if not selected_features:
            QMessageBox.warning(self, "警告", "请至少选择一个特征")
            return
        from correlation_analyzer import CorrelationAnalyzer
        CorrelationAnalyzer.show_heatmap(self.data, selected_features)
    def clear_data(self):
        """清空表格和散点图（修复图例移除异常）"""
        self.result_table.setRowCount(0)  
        if hasattr(self, 'ax'):
            self.ax.clear()
            self.ax.set_title("")  
            if self.ax.legend_ is not None:
                self.ax.legend_.remove()
            if hasattr(self, 'canvas') and self.canvas:
                self.canvas.draw()
    def on_table_row_clicked(self, item):
        row = item.row()
        model_name = self.result_table.item(row, 0).text()
        features_str = self.result_table.item(row, 1).text().replace(' ', '')
        result_key = f"{model_name}_{features_str}"
        if result_key in self.model_results:
            self.update_scatter_plot(self.model_results[result_key])
        else:
            QMessageBox.warning(self, "数据错误", f"未找到模型结果: {result_key}")
    def update_scatter_plot(self, result_data):
        """结果可视化函数 - 绘制实际值vs预测值散点图"""
        self.ax.clear()
        print(f"更新散点图: {result_data['model_name']}")
        self.ax.scatter(result_data['y_train'], result_data['y_train_pred'], c='green', alpha=0.3, label='Training set prediction')
        self.ax.scatter(result_data['y_test'], result_data['y_test_pred'], c='blue', alpha=0.5, label='Test set prediction')
        self.ax.plot([result_data['y_train'].min(), result_data['y_train'].max()], 
                    [result_data['y_train'].min(), result_data['y_train'].max()], 'r--', lw=2)
        self.ax.set_xlabel('Actual LAI value')
        self.ax.set_ylabel('Predicted LAI value')
        self.ax.set_title(f'{result_data["model_name"]} Train_R^2={result_data["train_r2"]:.2f} Test_R^2={result_data["test_r2"]:.2f}')
        self.ax.legend()
        self.canvas.draw_idle()  
    def _create_menu(self):
        """创建本地菜单栏"""
        from PyQt5.QtWidgets import QMenuBar, QMenu, QAction
        menubar = QMenuBar()
        self.main_layout.setMenuBar(menubar) 
        file_menu = menubar.addMenu("文件")
        import_feature_action = QAction("导入特征表格", self)
        import_feature_action.triggered.connect(self.load_data)
        file_menu.addAction(import_feature_action)
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        settings_menu = menubar.addMenu("参数设置")
        training_settings_action = QAction("训练设置", self)
        training_settings_action.triggered.connect(self.open_training_settings)
        settings_menu.addAction(training_settings_action)
        data_menu = menubar.addMenu("数据管理")
        clear_data_action = QAction("清空数据", self)
        clear_data_action.triggered.connect(self.clear_data)
        data_menu.addAction(clear_data_action)
        select_all_action = QAction("全选纹理特征", self)
        select_all_action.triggered.connect(self._select_all_features)
        data_menu.addAction(select_all_action)
        deselect_all_action = QAction("取消全选", self)
        deselect_all_action.triggered.connect(self._deselect_all_features)
        data_menu.addAction(deselect_all_action)
        merge_columns_action = QAction("列合并表格（横向拼接）", self)
        merge_columns_action.triggered.connect(self._merge_columns_tables)
        data_menu.addAction(merge_columns_action)
        export_merged_action = QAction("导出合并数据", self)
        export_merged_action.triggered.connect(self.export_merged_data)
        data_menu.addAction(export_merged_action)
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    # 添加所有缺失的菜单方法实现
    def open_training_settings(self):
        from PyQt5.QtWidgets import QInputDialog, QMessageBox
        ratio, ok = QInputDialog.getDouble(
            self, "训练集设置", "请输入训练集比例（0-1之间）:",
            value=self.train_test_ratio, min=0.1, max=0.9, decimals=2
        )
        if ok:
            self.train_test_ratio = ratio
            QMessageBox.information(self, "设置成功", f"训练集比例已设置为：{ratio:.0%}")
    def _select_all_features(self):
        for cb in self.feature_checkboxes.values():
            cb.setChecked(True)
    def _deselect_all_features(self):
        for cb in self.feature_checkboxes.values():
            cb.setChecked(False)
    def _merge_columns_tables(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import pandas as pd
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择要列合并的两个表格文件", "", "CSV文件 (*.csv)"
        )
        if len(files) != 2:
            QMessageBox.information(self, "提示", "请选择恰好两个CSV文件")
            return
        try:
            df1 = pd.read_csv(files[0])
            df2 = pd.read_csv(files[1])
            merged_df = pd.concat([df1, df2], axis=1)
            self.data = merged_df
            QMessageBox.information(self, "成功", f"已完成列合并，共{len(merged_df)}行数据")
        except Exception as e:
            QMessageBox.critical(self, "合并失败", f"错误原因：{str(e)}")
    def _show_about(self):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.about(
            self, 
            "关于模型构建工具",
            """<h3>模型构建工具</h3>
            <p>支持多模型预测与特征相关性分析的可视化工具</p>
            <p><b>核心功能：</b></p>
            <ul>
                <li>特征数据导入与管理（支持CSV格式）</li>
                <li>训练集比例灵活设置（0.1-0.9范围）</li>
                <li>特征选择与光谱波长范围筛选</li>
                <li>多表格合并（行合并/列合并）</li>
                <li>模型性能评估（RMSE、R²指标）</li>
                <li>特征相关性热力图分析</li>
                <li>预测结果可视化与导出</li>
            </ul>
            <p><b>数据管理功能：</b></p>
            <ul>
                <li>列合并表格：横向拼接特征数据</li>
                <li>特征全选/反选：快速选择输入特征</li>
                <li>波长范围筛选：根据波长值范围选择光谱特征</li>
            </ul>
            <p><b>结果解释：</b></p>
            <ul>
                <li>训练RMSE：训练集均方根误差，值越小越好</li>
                <li>测试RMSE：测试集均方根误差，值越小越好</li>
                <li>训练R²：训练集决定系数，越接近1越好</li>
                <li>测试R²：测试集决定系数，越接近1越好</li>
                <li>散点图：实际值与预测值的分布关系</li>
            </ul>
            """
        )
    def toggle_all_features(self, state):
        """全选/取消全选所有特征复选框"""
        all_checked = all(checkbox.isChecked() for checkbox in self.feature_checkboxes.values())
        checked = not all_checked  
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(checked)
    def update_select_all_state(self):
        """当特征复选框状态变化时，更新全选复选框状态"""
        if not self.feature_checkboxes:
            return
        all_checked = all(checkbox.isChecked() for checkbox in self.feature_checkboxes.values())
        any_checked = any(checkbox.isChecked() for checkbox in self.feature_checkboxes.values())
    def apply_wavelength_range(self):
        """应用波长范围选择光谱特征"""
        try:
            min_wl = float(self.min_wavelength.text()) if self.min_wavelength.text() else 100
            max_wl = float(self.max_wavelength.text()) if self.max_wavelength.text() else 1000
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的波长数值")
            return
        selected_spectral = []
        for col in self.spectral_features:
            wavelength = float(col)
            if min_wl <= wavelength <= max_wl:
                selected_spectral.append(col)
        if not selected_spectral:
            QMessageBox.information(self, "提示", f"未找到{min_wl}-{max_wl}nm范围内的光谱特征")
            return
        self.selected_spectral_features = selected_spectral
        QMessageBox.information(self, "成功", f"已选择{len(selected_spectral)}个光谱特征")
    def apply_multi_bands(self):
        """应用多波段选择（格式如: 400,500-600,700）"""
        input_str = self.multi_bands.text().strip()
        if not input_str:
            QMessageBox.warning(self, "输入错误", "请输入波段选择")
            return
        selected_bands = set()
        parts = input_str.split(',')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                try:
                    start, end = part.split('-')
                    start_wl = float(start.strip())
                    end_wl = float(end.strip())
                    if start_wl > end_wl:
                        start_wl, end_wl = end_wl, start_wl
                    for band in self.spectral_features:
                        wl = float(band)
                        if start_wl <= wl <= end_wl:
                            selected_bands.add(band)
                except ValueError:
                    QMessageBox.warning(self, "格式错误", f"无效的范围格式: {part}")
                    return
            else:
                try:
                    target_wl = float(part)
                    found = False
                    for band in self.spectral_features:
                        if abs(float(band) - target_wl) < 1e-6:  
                            selected_bands.add(band)
                            found = True
                            break
                    if not found:
                        QMessageBox.information(self, "未找到", f"未找到波段: {target_wl}")
                except ValueError:
                    QMessageBox.warning(self, "格式错误", f"无效的波段格式: {part}")
                    return
        if not selected_bands:
            QMessageBox.information(self, "提示", "未选择任何波段")
            return
        self.selected_spectral_features = list(selected_bands)
        QMessageBox.information(self, "成功", f"已选择 {len(selected_bands)} 个光谱特征")
    def on_select_all_clicked(self):
        """全选/取消全选所有特征复选框"""
        all_checked = all(cb.isChecked() for cb in self.feature_checkboxes.values())
        new_state = not all_checked
        for cb in self.feature_checkboxes.values():
            cb.setChecked(new_state)
        self.select_all_button.setText("取消全选" if new_state else "全选")
class ModelBuilderWindow(QMainWindow):
    """模型构建独立窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("模型构建工具")
        self.setGeometry(300, 300, 1200, 1000)
        self.setStyleSheet(QSS_STYLE)
        self.model_widget = ModelInferenceWidget()  
        self.setCentralWidget(self.model_widget)     
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication  
    app = QApplication(sys.argv)
    window = ModelBuilderWindow()
    window.show()
    sys.exit(app.exec_())