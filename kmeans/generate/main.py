import unsupervisedDeepLearning
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    k = 11  # Number of clusters
    no_of_iterations = 20  # Number of iterations for K-means
    centroids = unsupervisedDeepLearning.kmeansgenerate(k, no_of_iterations) # Apply K-means clustering
    # Generate new digits from the cluster centroids
    generated_digits = centroids.reshape(k, 28, 28)


    # In[ ]:


    fig, axs = plt.subplots(1, k, figsize=(12, 2))
    for i in range(k):
        axs[i].imshow(generated_digits[i], cmap='gray')
        axs[i].axis('off')
    plt.show()
    
