import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

# 使用高斯混合模型生成合成数据
gmm = GaussianMixture(n_components=3, random_state=42)
# 使用随机数据拟合 GMM
X_init = np.random.rand(300, 2)
gmm.fit(X_init)

X_gmm, y_gmm = gmm.sample(100)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 可视化生成的样本及其标签
plt.scatter(X_gmm[:, 0], X_gmm[:, 1], c=y_gmm, cmap='viridis', alpha=0.5)
plt.title('GMM 生成的样本及其标签')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.show()

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_gmm)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_gmm, test_size=0.3, random_state=42)

# 定义 KNN 模型
knn = KNeighborsClassifier(n_neighbors=5)

# 训练 KNN 模型
knn.fit(X_train, y_train)

# 预测测试集标签
y_pred = knn.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'KNN 准确率: {accuracy}')
print('分类报告:')
print(classification_report(y_test, y_pred))

# 可视化测试集的预测结果
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', alpha=0.5)
plt.title('KNN 分类结果')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.show()

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵 - KNN')
plt.show()
