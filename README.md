# Clustering Techniques

We will be talking about 4 categories of Clustering Techniques in this Repo:
- K-means
- Agglomerative clustering
- Density-based spatial clustering (DBSCAN)
- Gaussian Mixture Modelling (GMM)


## K-means
The K-means algorithm is an iterative process with three critical stages:
Pick initial cluster centroids
The algorithm starts by picking initial k cluster centers which are known as centroids. Determining the optimal number of clusters i.e k as well as proper selection of the initial clusters is extremely important for the performance of the model. The number of clusters should always be dependent on the nature of the dataset while poor selection of the initial cluster can lead to the problem of local convergence. Thankfully, we have solutions for both.
For selection of initial clusters, we can either run multiple iterations of the model with various initializations to pick the most stable one or use the “k-means++” algorithm which has the following steps:
1. Randomly select the first centroid from the dataset
2. Compute distance of all points in the dataset from the selected centroid
3. Pick a point as the new centroid that has maximum probability proportional to this distance
4. Repeat steps 2 and 3 until k centroids have been sampled
The algorithm initializes the centroids to be distant from each other leading to more stable results than random initialization.

For implementing the model in python we need to do specify the number of clusters first. I have used Elbow method for this.

## Agglomerative clustering
Agglomerative clustering is a general family of clustering algorithms that build nested clusters by merging data points successively. This hierarchy of clusters can be represented as a tree diagram known as dendrogram. The top of the tree is a single cluster with all data points while the bottom contains individual points. There are multiple options for linking data points in a successive manner:
1. Single linkage: It minimizes the distance between the closest observations of pairs of clusters
2. Complete or Maximum linkage: Tries to minimize the maximum distance between observations of pairs of clusters
3. Average linkage: It minimizes the average of the distances between all observations of pairs of clusters
4. Ward: Similar to the k-means as it minimizes the sum of squared differences within all clusters but with a hierarchical approach. We will be using this option in our exercise.

And similar to K-means, we will have to specify the number of clusters in this model and the dendrogram can help us do that.

## DBSCAN
DBSCAN groups together points that are closely packed together while marking others as outliers which lie alone in low-density regions. There are two key parameters in the model needed to define ‘density’: minimum number of points required to form a dense region min_samples and distance to define a neighborhood eps. Higher min_samples or lower eps demands greater density to form a cluster.
Based on these parameters, DBSCAN starts with an arbitrary point x and identifies points that are within neighbourhood of x based on eps and classifies x as one of the following:
1. Core point: If the number of points in the neighbourhood is at least equal to the min_samples parameter then it called a core point and a cluster is formed around x.
2. Border point: x is considered a border point if it is part of a cluster with a different core point but number of points in it’s neighbourhood is less than the min_samples parameter. Intuitively, these points are on the fringes of a cluster.
3. Outlier or noise: If x is not a core point and distance from any core sample is at least equal to or greater thaneps , it is considered an outlier or noise.

For tuning the parameters of the model, we first identify the optimal eps value by finding the distance among a point’s neighbors and plotting the minimum distance. This gives us the elbow curve to find density of the data points and optimal eps value can be found at the inflection point. We use the NearestNeighbours function to get the minimum distance and the KneeLocator function to identify the inflection point.

## Gaussian Mixture Modelling (GMM)
A Gaussian mixture model is a distance based probabilistic model that assumes all the data points are generated from a linear combination of multivariate Gaussian distributions with unknown parameters. Like K-means it takes into account centers of the latent Gaussian distributions but unlike K-means, the covariance structure of the distributions is also taken into account. The algorithm implements the expectation-maximization (EM) algorithm to iteratively find the distribution parameters that maximize a model quality measure called log likelihood. The key steps performed in this model are:
1. Initialize k gaussian distributions
2. Calculate probabilities of each point’s association with each of the distributions
3. Recalculate distribution parameters based on each point’s probabilities associated with the the distributions
4. Repeat process till log-likelihood is maximized

There are 4 options for calculating covariances in GMM:
1. Full: Each distribution has its own general covariance matrix
2. Tied: All distributions share general covariance matrix
3. Diag: Each distribution has its own diagonal covariance matrix
4. Spherical: Each distribution has its own single variance

Apart from selecting the covariance type, we need to select the optimal number of clusters in the model as well. I used Silhouette score

Reference: 
1. https://www.datasklr.com/segmentation-clustering/an-introduction-to-clustering-techniques
2. https://www.kaggle.com/datafan07/heart-disease-and-some-scikit-learn-magic
