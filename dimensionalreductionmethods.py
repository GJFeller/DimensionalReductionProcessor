from __future__ import division
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import json
#import scipy

class DimensionalReductionMethods(object):

    def __init__(self, dataMatrix):
        self.dataMatrix = dataMatrix
        self.PCADataCoordinates = None
        self.MDSDataCoordinates = None
        self.SammonDataCoordinates = None

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
        distanceMatrix = self.calculateDistanceMatrix(self.dataMatrix['dataMatrix'])
        P = np.power(distanceMatrix, 2)
        identity = np.identity(len(P))
        J = np.subtract(np.identity(len(P)), np.divide(np.ones( (len(P), len(P) ) ), len(P) ) )
        #print(J)
        B = np.divide(np.dot(J, np.dot(P, J) ),-2)
        dataVectorBase = self.getLargerEigenvectors(B, 2)
        #print(dataVectorBase)
        self.MDSDataCoordinates = np.dot(distanceMatrix, dataVectorBase)
        #print(self.MDSDataCoordinates)

        #dataCoordTransposed = np.transpose(self.MDSDataCoordinates)
        #area = 2  # 0 to 15 point radii
        #plt.scatter(dataCoordTransposed[0], dataCoordTransposed[1], area, 'C1', alpha=0.5)
        #plt.show()
    
    def SammonMapping(self):
        maxIter = 50
        d = 2
        MF = 0.4
        originalDistanceMatrix = self.calculateDistanceMatrix(self.dataMatrix['dataMatrix'])
        #y = self.initializeSeedVectors(d)
        y = self.PCADataCoordinates
        sammonDistanceMatrix = self.calculateDistanceMatrix(y)
        error = self.errorFunction(originalDistanceMatrix, sammonDistanceMatrix)
        print("Sammon initial error %f" % error)
        errorPrevious = error
        for iter in range(0, maxIter):
            ynew = np.copy(y)
            for p in range(0, len(originalDistanceMatrix)):
                for q in range(0, d):
                    firstDerivate = 0
                    secondDerivate = 0
                    for j in range(0, len(originalDistanceMatrix)):
                        if p != j:
                            dpj_ = originalDistanceMatrix[p][j]
                            if dpj_ == 0:
                                dpj_ = 1e-10
                            dpj = sammonDistanceMatrix[p][j]
                            if dpj == 0:
                                dpj = 1e-10
                            firstDerivate = firstDerivate + ((dpj_ - dpj)/(dpj*dpj_))*(y[p][q]-y[j][q])
                            secondDerivate = secondDerivate + (1/dpj*dpj_)*((dpj_ - dpj) - (math.pow(y[p][q]-y[j][q], 2)/dpj)*(1 + ((dpj_-dpj)/dpj)))
                    c = self.getNormalizationFactor(originalDistanceMatrix)
                    firstDerivate = firstDerivate*(-2/c)
                    secondDerivate = secondDerivate*(-2/c)
                    ynew[p][q] = y[p][q] - MF*(firstDerivate/abs(secondDerivate))
            y = np.copy(ynew)
            sammonDistanceMatrix = self.calculateDistanceMatrix(y)
            error = self.errorFunction(originalDistanceMatrix, sammonDistanceMatrix)
            print("Sammon error in %d iteration with MF %f = %f" % (iter, MF, error))
            print(y)
        self.SammonDataCoordinates = y
            #if error > errorPrevious:
            #    MF = MF * 0.2
            #else:
            #    MF = MF * 1.5
            #    if MF > 0.4:
            #        MF = 0.4
        

    def errorFunction(self, originalDistanceMatrix, sammonDistanceMatrix):
        sumNormalizationFactor = self.getNormalizationFactor(originalDistanceMatrix)
        sumErrorFactor = 0
        for i in range(0, len(originalDistanceMatrix)-1):
            for j in range(i+1, len(originalDistanceMatrix)):
                divisor = originalDistanceMatrix[i][j]
                if originalDistanceMatrix[i][j] == 0:
                    divisor = 1e-10
                #print(divisor)
                sumErrorFactor = sumErrorFactor + (math.pow(originalDistanceMatrix[i][j] - sammonDistanceMatrix[i][j], 2) / divisor)
        return sumErrorFactor / sumNormalizationFactor
        
    def getNormalizationFactor(self, originalDistanceMatrix):
        sumNormalizationFactor = 0
        for i in range(0, len(originalDistanceMatrix)-1):
            for j in range(i+1, len(originalDistanceMatrix)):
                sumNormalizationFactor = sumNormalizationFactor + originalDistanceMatrix[i][j]
        return sumNormalizationFactor

    def initializeSeedVectors(self, d):
        deputiesCount = len(self.dataMatrix['dataMatrix'])
        return np.random.rand(deputiesCount,d)
        


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

    def calculateDistanceMatrix(self, dataMatrix):
        deputiesCount = len(dataMatrix)
        distanceMatrix = np.zeros((deputiesCount, deputiesCount))
        for i in range(0, deputiesCount):
            for j in range(i+1, deputiesCount):
                v1 = dataMatrix[i]
                v2 = dataMatrix[j]
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

    def exportCoordinatesToJSON(self):
        if self.PCADataCoordinates is not None:
            PCAScatterData = []
            for i, point in enumerate(self.PCADataCoordinates):
                PCAScatterData.append({'data': point.real.tolist(), 'party': self.dataMatrix['deputyList'][i]['party'], 'name': self.dataMatrix['deputyList'][i]['deputyId']})
            with open('data/semesters/PCA/'+self.dataMatrix['filename']+'.json', 'w') as fp:
                json.dump(PCAScatterData, fp)
        if self.MDSDataCoordinates is not None:
            MDSScatterData = []
            for i, point in enumerate(self.MDSDataCoordinates):
                MDSScatterData.append({'data': point.real.tolist(), 'party': self.dataMatrix['deputyList'][i]['party'], 'name': self.dataMatrix['deputyList'][i]['deputyId']})
            with open('data/semesters/MDS/'+self.dataMatrix['filename']+'.json', 'w') as fp:
                json.dump(MDSScatterData, fp)
        if self.SammonDataCoordinates is not None:
            SammonScatterData = []
            for i, point in enumerate(self.SammonDataCoordinates):
                SammonScatterData.append({'data': point.real.tolist(), 'party': self.dataMatrix['deputyList'][i]['party'], 'name': self.dataMatrix['deputyList'][i]['deputyId']})
            with open('data/semesters/Sammon/'+self.dataMatrix['filename']+'.json', 'w') as fp:
                json.dump(SammonScatterData, fp)
