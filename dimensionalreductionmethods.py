import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
#import scipy

class DimensionalReductionMethods(object):

    def __init__(self, dataMatrix):
        self.dataMatrix = dataMatrix

    def PCA(self):
        modifiedDataset = np.copy(self.meanSubtraction())
        covMatrix = self.calculateCovarianceMatrix(modifiedDataset)
        dataVectorBase = self.getLargerEigenvectors(covMatrix, 2)
        self.PCADataCoordinates = np.dot(modifiedDataset, dataVectorBase)
        #dataCoordTransposed = np.transpose(self.PCADataCoordinates)

        #area = 2  # 0 to 15 point radii
        #plt.scatter(dataCoordTransposed[0], dataCoordTransposed[1], area, 'C1', alpha=0.5)
        #plt.show()


    def MDS(self):
        distanceMatrix = self.calculateDistanceMatrix()
        P = np.power(distanceMatrix, 2)
        identity = np.identity(len(P))
        J = np.subtract(np.identity(len(P)), np.divide(np.ones( (len(P), len(P) ) ), len(P) ) )
        print(J)
        B = np.divide(np.dot(J, np.dot(P, J) ),-2)
        dataVectorBase = self.getLargerEigenvectors(B, 2)
        print(dataVectorBase)
        self.MDSDataCoordinates = np.dot(distanceMatrix, dataVectorBase)
        print(self.MDSDataCoordinates)

        #dataCoordTransposed = np.transpose(self.MDSDataCoordinates)
        #area = 2  # 0 to 15 point radii
        #plt.scatter(dataCoordTransposed[0], dataCoordTransposed[1], area, 'C1', alpha=0.5)
        #plt.show()

    def getLargerEigenvectors(self, matrix, d):
        size = len(matrix)
        lambdaValues, vectors = LA.eig(matrix)
        print(lambdaValues)
        #print(vectors)
        vectors = np.transpose(vectors)
        largestEigenvalues = []
        for i in range (0, d):
            largestEigenvalues.append({'lambda': lambdaValues[i], 'idx': i})
        largestEigenvalues = sorted(largestEigenvalues, key=lambda eigenval: eigenval['lambda'], reverse=True)
        for i in range (d, len(lambdaValues)):
            largestEigenvalues.append({'lambda': lambdaValues[i], 'idx': i})
            largestEigenvalues = sorted(largestEigenvalues, key=lambda eigenval: eigenval['lambda'], reverse=True)
            largestEigenvalues = largestEigenvalues[:-1]
        print(largestEigenvalues)
        selectedEigenvectors = []
        for i in range(0, d):
            selectedEigenvectors.append(vectors[largestEigenvalues[i]['idx']])
        #print("Eigenvectors selected:")
        #print(selectedEigenvectors)
        return np.transpose(selectedEigenvectors)

    def meanSubtraction(self):
        modifiedDataset = np.copy(self.dataMatrix['dataMatrix'])
        rollCallCount = len(modifiedDataset[0])
        deputyCount = len(modifiedDataset)
        for j in range(0, rollCallCount):
            sum = 0
            for i in range(0, deputyCount):
                sum = sum + self.dataMatrix['dataMatrix'][i][j]
            mean = sum / deputyCount
            for i in range(0, deputyCount):
                modifiedDataset[i][j] = modifiedDataset[i][j] - mean
        return modifiedDataset

    def calculateCovarianceMatrix(self, matrix):
        rollCallCount = len(matrix[0])
        covarianceMatrix = np.zeros((rollCallCount, rollCallCount))
        for i in range(0, rollCallCount):
            for j in range(i, rollCallCount):
                covarianceMatrix[i][j] = self.calculateCovariance(matrix, i, j, rollCallCount)
        
        for i in range(1, rollCallCount):
            for j in range(0, i):
                covarianceMatrix[i][j] = covarianceMatrix[j][i]
        return covarianceMatrix
    
    def calculateCovariance(self, matrix, i, j, covDim):
        cov = 0
        for k in range(0, covDim):
            cov = cov + matrix[k][i] * matrix[k][j]
        return cov / (covDim - 1)

    def calculateDistanceMatrix(self):
        deputiesCount = len(self.dataMatrix['dataMatrix'])
        distanceMatrix = np.zeros((deputiesCount, deputiesCount))
        for i in range(0, deputiesCount):
            for j in range(i+1, deputiesCount):
                v1 = self.dataMatrix['dataMatrix'][i]
                v2 = self.dataMatrix['dataMatrix'][j]
                distanceMatrix[i][j] = self.euclideanDistance(v1, v2);
        for i in range(1, deputiesCount):
            for j in range(0, i):
                distanceMatrix[i][j] = distanceMatrix[j][i]
        return distanceMatrix
    
    def euclideanDistance(self, v1, v2):
        sum = 0
        for i in range(0, len(v1)):
            sum = sum + math.pow(v1[i]-v2[i], 2)
        return math.sqrt(sum)
