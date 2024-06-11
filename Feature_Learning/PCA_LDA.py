import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import fetch_lfw_people

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载人脸数据集
faces = fetch_lfw_people(min_faces_per_person=100)

print(faces.target_names) # 数据集样本类别
print(faces.images.shape) # 图像大小
print(faces.data.shape) # 样本数据大小
print(faces.target.shape) # 标签数组

# 查看部分人脸图片
fig, ax = plt.subplots(2,5) # 生成2行5列的子图，查看10张图片
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone') # 显示人脸图片
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]]) # 显示姓名

plt.show()
X = faces.data
y = faces.target
target_names = faces.target_names

# LDA降维
lda = LDA(n_components=len(target_names) - 1).fit(X, y)
data_lda = lda.transform(X) # 降维转换
print('LDA:', data_lda.shape) # 查看数据维度

# PCA降维
pca = PCA(n_components=150).fit(X) # 利用PCA算法降维
data_pca = pca.transform(X) # 降维转换
print('PCA:', data_pca.shape) # 查看数据维度

data = {'orign':X, 'lda':data_lda, 'pca':data_pca}

for name, X in data.items():
    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 使用逻辑回归进行分类
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, y_train)

    # 预测测试集标签
    y_pred = log_reg.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name}逻辑回归准确率: {accuracy}')
    print(f'{name}分类报告:')
    print(classification_report(y_test, y_pred))

    if (name == 'pca'):
        # 将数据投影到二维空间
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_test)

        # 可视化二维投影
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_test, cmap='viridis', alpha=0.5)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('PCA 二维投影')
        plt.show()

    elif (name == 'lda'):
        # 将数据投影到二维空间
        lda_2d = LDA(n_components=2)
        X_lda_2d = lda_2d.fit_transform(X_test, y_test)

        # 可视化二维投影
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_lda_2d[:, 0], X_lda_2d[:, 1], c=y_test, cmap='viridis', alpha=0.5)
        plt.xlabel('LDA 1')
        plt.ylabel('LDA 2')
        plt.title('LDA 二维投影')
        plt.show()

