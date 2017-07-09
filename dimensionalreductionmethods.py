import numpy as np
import scipy

class DimensionalReductionMethods:
    def __init__(self):

    def PCA(dataset):
        modifiedDataset = meanSubtraction(dataset)
        covMatrix = calculateCovarianceMatrix(modifiedDataset)
        

    def MDS(dataset):

    def meanSubtraction(dataset):
        modifiedDataset = np.copy(dataset)
        rollCallCount = len(modifiedDataset[0])
        deputyCount = len(modifiedDataset)
        for j in range(0, rollCallCount):
            sum = 0
            for i in range(0, deputyCount):
                sum = sum + dataset[i][j]
            mean = sum / deputyCount
            for i in range(0, deputyCount):
                modifiedDataset[i][j] = modifiedDataset[i][j] - mean
        return modifiedDataset

    def calculateCovarianceMatrix(matrix):
        rollCallCount = len(matrix[0])
        covarianceMatrix = np.zeros((rollCallCount, rollCallCount))
        for i in range(0, rollCallCount):
            for j in range(i, rollCallCount):
                covarianceMatrix[i][j] = calculateCovariance(matrix, i, j, rollCallCount)
        
        for i in range(1, rollCallCount):
            for j in range(0, i):
                covarianceMatrix[i][j] = covarianceMatrix[j][i]
        return covarianceMatrix
    
    def calculateCovariance(matrix, i, j, covDim):
        cov = 0
        for k in range(0, covDim):
            cov = cov + matrix[k][i] * matrix[k][j]
        return cov / (covDim - 1)