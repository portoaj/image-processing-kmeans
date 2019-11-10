from mpl_toolkits.mplot3d import Axes3D
import random
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class KMeans:
  def __init__(self, dimensions):
    self.p = np.zeros(0)
    self.height = 0
    self.width = 0
    self.dimensions = dimensions
    self.points = np.zeros(0)
    self.centroids = np.zeros(0)
    self.k = 2
    self.maxIterations = 0
    self.centroids = np.zeros(0)
    self.pointLabels = np.zeros(0)
    self.inertias = list()
  def initialize(self, inputPath):
    self.p = np.array(Image.open(inputPath))
    self.height = len(self.p)
    self.width = len(self.p[0])
    tempPoints = np.zeros((self.dimensions,(self.width * self.height)))
    for i in range(len(self.p)):
      for j in range(len(self.p[0])):
        for k in range(self.dimensions):
          tempPoints[k][i * self.width + j] = self.p[i][j][k]
    self.points = tempPoints.T
    self.pointLabels = [None] * self.width * self.height
    print('initialized')

  def updatePoint(self, point, centroids, index, pointLabels):
    pointLabels[index] = np.argmin(((point[:, None] - centroids.T) ** 2).sum(axis=0))

  def updateCentroid(self, centroids, points, index, pointLabels, dimensions):
    indices = list()
    resets = 0
    for i in range(len(pointLabels)):
      if pointLabels[i] == index:
        indices.append(i)
    if len(indices) == 0:
      resets = resets + 1
      self.setRandomPos(centroids, index, dimensions)
      return np.zeros((dimensions)), True
    return np.divide(points[indices].sum(axis=0), len(indices)), False

  def getInertia(self, points, pointLabels, centroids):
    inertia = 0
    for i in range(len(points)):
      inertia += ((points[i] - centroids[pointLabels[i]]) ** 2).sum()
    print('inertia', inertia)
    return inertia

  def setRandomPos(self, centroids, index, dimensions):
    newPos = [0] * dimensions
    for i in range(dimensions):
      newPos[i] = random.random() * 255
    centroids[index] = newPos

  def finishCheck(self, centroids, oldCentroids):
    return np.array_equal(centroids, oldCentroids)

  def train(self, maxIterations):
    tempCentroids = np.zeros((self.dimensions, self.k))
    for i in range(self.dimensions):
      for j in range(self.k):
        tempCentroids[i][j] = random.random() * 255
    self.centroids = tempCentroids.T
    self.pointLabels = [None] * self.width * self.height
    iter = 0
    oldCentroids = np.zeros((self.k, self.dimensions))
    while iter < maxIterations:
      #Set labels relating each point to the closest centroid/ their cluster
      for i in range(len(self.points)):
        self.updatePoint(self.points[i], self.centroids, i, self.pointLabels)
      numberReset = 0
      for i in range(len(self.centroids)):
        potentialCentroid, reset = self.updateCentroid(self.centroids, self.points, i, self.pointLabels, self.dimensions)
        if not reset:
          self.centroids[i] = potentialCentroid
        else:
          numberReset = numberReset + 1
      if numberReset is not 0:
        print('reset', numberReset, 'centroids')
      if self.finishCheck(self.centroids, oldCentroids):
        print('finished')
        break
      oldCentroids = self.centroids.copy()
      iter += 1
      print('epoch',iter)
    return self.getInertia(self.points, self.pointLabels, self.centroids)
  def outputGraph(self):
    r = list()
    g = list()
    b = list()
    clusters = list()
    for i in range(len(self.points)):
      r.append(self.points[i][0])
      g.append(self.points[i][1])
      b.append(self.points[i][2])
      clusters.append(self.pointLabels[i])
    for i in range(len(self.centroids)):
      r.append(self.centroids[i][0])
      g.append(self.centroids[i][1])
      b.append(self.centroids[i][2])
      clusters.append(self.k + 1)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(r, g, b, c=clusters, cmap='Accent', marker=',')
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    print(self.centroids, self.k, self.pointLabels)
    plt.show()
  def outputImageAlternate(self):
    data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    for i in range(self.width * self.height):
      data[int(i / self.width)][int(i % self.width)] = self.centroids[self.pointLabels[i]]
    plt.imshow(data)
  def outputImage(self):
    data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    for i in range(self.width * self.height):
      data[int(i / self.width)][int(i % self.width)] = self.centroids[self.pointLabels[i]]
    img = Image.fromarray(data, 'RGB')
    img.show()

  def elbow(self, maxK, maxIterations, inputPath):
    inertias = np.zeros(maxK - 1)
    distances = np.zeros(maxK - 1)
    #find all inertias
    for i in range(len(inertias)):
      self.k = i + 2
      self.initialize(inputPath)
      print('testing k=' + str(i + 2))
      inertias[i] = self.train(maxIterations)
    print('found inertias')
    #use line-distance formula to find best cluster
    for i in range(len(inertias)):
      x0, y0 = i + 2, inertias[i]
      x1, y1 = 2, inertias[0]
      x2, y2 = maxK, inertias[len(inertias) - 1]

      numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
      denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
      distances[i] = numerator / denominator
    print(inertias)
    print('optimal k', distances.argmax() + 2)
    return distances.argmax() + 2, inertias
  def graphInertias(self):
    xs = np.arange(2, len(self.inertias) + 2)
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.plot(xs, self.inertias)
    plt.show()