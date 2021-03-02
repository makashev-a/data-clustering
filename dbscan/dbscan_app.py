import matplotlib.pyplot as pl
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Выполнение кластеризации с помощью DBSCAN
iris = load_iris()
dbscan = DBSCAN(eps=0.5, min_samples=5)

dbscan.fit(iris.data)

# Выполнение печати меток принадлежности к кластерам
print(dbscan.labels_)

# Визуализация данных
pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

pl.legend([c1, c2, c3], ['Кластер 1', 'Кластер 2', 'Шум'])
pl.title('DBSCAN находит 2 кластера и шум')
pl.show()
