import json
import os
import glob

data = []
list_of_files = glob.glob('./data/semesters/*.json')
for file_name in list_of_files:
    file = open(file_name, 'r')
    with file as data_file:
        data.append({'filename': file_name, 'data': json.load(data_file)})

print(data)
