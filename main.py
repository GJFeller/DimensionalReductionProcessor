import json
import os
import glob
import numpy as np
from dimensionalreductionmethods import DimensionalReductionMethods

def find_index(dicts, key, value):
    class Null: pass
    for i, d in enumerate(dicts):
        if d.get(key, Null) == value:
            return i
    else:
        return -1

data = []
list_of_files = glob.glob('./data/semesters/*.json')
for file_name in list_of_files:
    file = open(file_name, 'r')
    with file as data_file:
        if os.path.splitext(os.path.basename(file_name))[0] == "2013-1":
            data.append({'filename': os.path.splitext(os.path.basename(file_name))[0], 'data': json.load(data_file)})


dataMatrix = []
#print(data[0]["data"][0])
for dataRow in data:
    deputiesVotedList = []
    for rollCall in dataRow["data"]:
        for rollVote in rollCall['rollVotes']:
            if find_index(deputiesVotedList, 'deputyId', rollVote['deputyID']) == -1:
                deputiesVotedList.append({'deputyId': rollVote['deputyID'], 'party': rollVote['party']})
    deputiesVotedList = sorted(deputiesVotedList, key=lambda deputy: deputy['deputyId'])
    rollCallMatrix = np.zeros((len(deputiesVotedList), len(dataRow['data'])))
    for rollCallIdx, rollCall in enumerate(dataRow["data"]):
        for rollVote in rollCall['rollVotes']:
            deputyIdx = find_index(deputiesVotedList, 'deputyId', rollVote['deputyID'])
            rollCallMatrix[deputyIdx][rollCallIdx] = rollVote['vote']
    dataMatrix.append({'filename': dataRow["filename"], 'dataMatrix': rollCallMatrix, 'deputyList': deputiesVotedList})
    #if dataRow['filename'] == "2013-1":
    #    print(dataRow["filename"])
    #    print(len(deputiesVotedList))
    #    print(deputiesVotedList)
    #    print(rollCallMatrix.shape)
    #    print(rollCallMatrix)
#print(dataMatrix)
processedData = []
for dataRow in dataMatrix:
    #if dataRow["filename"] == "2013-1":
    semesterData = DimensionalReductionMethods(dataRow)
    semesterData.PCA()
    #semesterData.SammonMapping()
    semesterData.MDS()
    semesterData.exportCoordinatesToJSON()
    processedData.append(semesterData)
    #DimensionalReductionMethods.PCA(dataRow['dataMatrix'])
    
    
    
