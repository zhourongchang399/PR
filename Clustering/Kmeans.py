import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 使用高斯混合模型生成合成数据
gmm = GaussianMixture(n_components=3)
# 使用随机数据拟合 GMM
X_init = np.random.rand(1000, 2)
gmm.fit(X_init)

X, _ = gmm.sample(100)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 可视化生成的样本及其标签
plt.scatter(X[:, 0], X[:, 1])
plt.title('GMM 生成的样本')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.show()

class CustomKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init='k-means++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids = None
        self.inertia_ = None
        self.labels_ = None
        self.history_centroids = []
        self.history_labels = []

    def initialize_centroids(self, X):
        if self.init == 'k-means++':
            centroids = []
            centroids.append(X[np.random.randint(X.shape[0])])
            for _ in range(1, self.n_clusters):
                distances = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
                probs = distances / np.sum(distances)
                cumulative_probs = np.cumsum(probs)
                r = np.random.rand()
                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        centroids.append(X[j])
                        break
            self.centroids = np.array(centroids)
        else:
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X[indices]

    def fit(self, X):
        self.initialize_centroids(X)
        for i in range(self.max_iter):
            # 记录每次迭代的质心和标签
            self.history_centroids.append(self.centroids.copy())
            
            # 计算每个点到质心的距离
            distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
            self.labels_ = np.argmin(distances, axis=0)
            
            # 记录每次迭代的标签
            self.history_labels.append(self.labels_.copy())

            new_centroids = np.array([X[self.labels_ == j].mean(axis=0) for j in range(self.n_clusters)])
            
            # 检查是否收敛
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.tol):
                break
            
            self.centroids = new_centroids

        self.inertia_ = np.sum([np.linalg.norm(X[self.labels_ == j] - centroid, axis=1).sum() for j, centroid in enumerate(self.centroids)])

    def plot_iterations(self, X):
        plt.figure(figsize=(15, 10))
        for i, (centroids, labels) in enumerate(zip(self.history_centroids, self.history_labels)):
            plt.subplot(3, 3, i + 1)
            plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis')
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
            plt.title(f'Iteration {i + 1}')
            if i + 1 == len(self.history_centroids):
                break
        plt.tight_layout()
        plt.show()

# 定义 K 值
k_values = [2, 3, 4]

# 初始化自定义 K-Means 模型
kmeans_models = [CustomKMeans(n_clusters=k, max_iter=20, tol=1e-4, init='k-means++') for k in k_values]

for i, kmeans in enumerate(kmeans_models):
    kmeans.fit(X)
    kmeans.plot_iterations(X)

plt.show()

# 定义不同的随机种子
seeds = [42, 100, 2023, 1314, 520]

# 对每个种子运行 K-means 并绘制结果
plt.figure(figsize=(15, 5))
for i, seed in enumerate(seeds):
    np.random.seed(seed)
    kmeans = CustomKMeans(n_clusters=3, max_iter=100, tol=1e-4, init='k-means++')
    kmeans.fit(X)
    plt.subplot(1, len(seeds), i+1)
    plt.title(f'随机种子 {seed}')
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=10, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')

plt.tight_layout()
plt.show()
