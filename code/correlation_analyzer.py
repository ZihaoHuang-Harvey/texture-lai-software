#相关性计算模块
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 特征相关性分析工具类
class CorrelationAnalyzer:
    @staticmethod
    def show_heatmap(data, features):
        # 显示特征间的相关性热力图
        df = data[features]
        corr_matrix = df.corr(method='pearson')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            annot_kws={'size': 12},
            cmap='coolwarm',
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('特征相关性热力图', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()