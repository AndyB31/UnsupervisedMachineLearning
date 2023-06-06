
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np

from random import uniform
from tqdm.notebook import tqdm

mnist = fetch_openml("mnist_784", version=1)

X_train = mnist['data'].to_numpy()

def euclidean(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))

class KMeans:
    def __init__(self, k: int = 5, max_iter: int = 200):
        self.k = k
        self.max_iter = max_iter

    def run(self, X):
        prev_cent = None
        min_, max_ = np.min(X, axis=0), np.max(X, axis=0)
        self.centroids = [uniform(min_, max_) for _ in range(self.k)]

        for it in tqdm(range(self.max_iter)):
            if not np.not_equal(self.centroids, prev_cent).any():
                break
            # Sort each data point, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.k)]
            for x in X:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            prev_cent = self.centroids
            self.centroids = [np.mean(cluster, axis=0)
                              for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                # Catch any np.nans, resulting from a centroid having no points
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_cent[i]
        return self.centroids

    def eval(self, X, cent = None):
        if cent:
            self.centroids = cent
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return np.array(centroids), centroid_idxs


# kmeans = KMeans(k=5, max_iter=300)

# kmeans.run(X_train)

# centers, classif = kmeans.eval(X_train)

# from sklearn.cluster import KMeans as km

# print(X_train.shape)

# km_sk = km(5, max_iter=200)

# km_sk.fit(X_train)

# pred = km_sk.cluster_centers_
# # print(np.array(kmeans.centroids).shape)
# print(pred.shape)
# # delta = (pred-np.array(kmeans.centroids))
# with np.printoptions(threshold=np.inf):
#     # print(delta)
#     print(km_sk.labels_)
#     print(X_train[0].shape)
#     print(X_train[0])


from skimage import io
import numpy as np
from tqdm.notebook import tqdm

class Compressor:
    def __init__(self, size = 64):
        self.size = size
        self.kmeans = None
        self.centers = None

    def Compress(self, imgarr = None, imgpath: str = None, outpath: str = None):
        #Read the image
        image = io.imread(imgpath) if imgpath else imgarr
        with np.printoptions(threshold=np.inf):
            print(image.shape)
            # print(image)
        # io.imshow(image)
        # io.show()

        #Dimension of the original image
        rows = image.shape[0]
        cols = image.shape[1]

        #Flatten the image
        image = image.reshape(rows*cols, 3)

        print("kmeans start")
        #Implement k-means clustering to form k clusters
        self.kmeans = KMeans(k=self.size, max_iter=300)
        self.centers = self.kmeans.run(image)
        print("kmeans eval")

        _, labels_ = self.kmeans.eval(image)

        print(labels_)
        labels_ = np.array(labels_).reshape(rows, cols)

        if outpath:
            io.imsave(outpath, labels_)

        return labels_, self.centers

    def Decompress(self, imgarr = None, imgpath: str = None, outpath: str = None, cent = None):
        #Read the image
        if cent:
            self.centers = cent
        image = io.imread(imgpath) if imgpath else imgarr
        with np.printoptions(threshold=np.inf):
            print(image.shape)
            # print(image)
        # io.imshow(image)
        # io.show()

        #Dimension of the original image
        rows = image.shape[0]
        cols = image.shape[1]

        new = np.zeros((rows, cols, 3), dtype=np.uint8)
        self.centers = np.array(self.centers)
        print(self.centers.shape)
        for i in tqdm(range(rows)):
            for j in range(cols):
                index = image[i, j]
                new[i, j, :] = self.centers[index, :]
        io.imshow(new)
        io.show()

        if outpath:
            io.imsave(outpath, new)

        return new


comp = Compressor(12)
lbl, cent = comp.Compress(imgpath="images.jpeg", outpath = 'images_12.jpeg')

comp = Compressor(8)
print(lbl)
comp.Decompress(imgarr=lbl, outpath = 'images_12_decomp.jpeg', cent=cent)
