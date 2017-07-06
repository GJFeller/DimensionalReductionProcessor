import json
import os
import glob

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
        data.append({'filename': os.path.splitext(os.path.basename(file_name))[0], 'data': json.load(data_file)})


dataMatrix = []
#print(data[0]["data"][0])
for dataRow in data:
    print(dataRow["filename"])
    deputiesVotedList = []
    for rollCall in dataRow["data"]:
        for rollVote in rollCall['rollVotes']:
            if find_index(deputiesVotedList, 'deputyId', rollVote['deputyID']) == -1:
                deputiesVotedList.append({'deputyId': rollVote['deputyID'], 'party': rollVote['party']})
    print(len(deputiesVotedList))
    
