import numpy as np  
from sklearn.mixture import GaussianMixture  
from sklearn.metrics import silhouette_score  
from sklearn.datasets import make_blobs  

# 使用高斯混合模型生成合成数据
gmm = GaussianMixture(n_components=3)
# 使用随机数据拟合 GMM
X_init = np.random.rand(1000, 2)
gmm.fit(X_init)

X, y = gmm.sample(100)

scores = []  
  
for k in range(2, 6):  
    gmm = GaussianMixture(n_components=k, random_state=0)  
    gmm.fit(X)  # 使用EM算法拟合模型  
    labels = gmm.predict(X)  # 预测样本的类别  
    score = silhouette_score(X, labels)  # 计算轮廓系数  
    scores.append(score)  
    print(f"K={k}, Silhouette Score: {score}")  
  
best_k = scores.index(max(scores)) + 2  # 因为range是从2开始的  
print(f"Best K value: {best_k}")