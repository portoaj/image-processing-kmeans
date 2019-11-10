# Image Processing with K-Means
The KMeans class takes an image, clusters it using the K-Means algorithm and gives a variety of outputs.  It also includes an elbow method implementation for automatically choosing the optimal number of clusters(K).
# Usage
1.Run the jupyter notebook (imageprocessingkmeans.ipynb) and upload an image and the kmeans class
*OR*
1. Import KMeans
```python
from kmeans import KMeans
```
2. Initialize a KMeans class with 3 dimensions(R,G,B)
```python
km = KMeans(3)
```
3. Use elbow method to find optimal K value
```python
km.k, km.inertias = km.elbow(<maxK>, <maxIterations>, <filePath>)
#Or set km.k manually without using the elbow method
```
4.Train the model
```python
km.train(<maxIterations>)
```
5.Generate desired output
```python
#Use any of the following
km.graphInertias()
km.outputGraph()
km.outputImage()
km.outputImageAlternate()
```
