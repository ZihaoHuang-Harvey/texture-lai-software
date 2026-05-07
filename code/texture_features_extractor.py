import pandas as pd
# 纹理特征数据导出工具
def save_features_to_excel(features, path):
    df = pd.DataFrame([features])
    df.to_excel(path, index=False)
