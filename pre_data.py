import json
import numpy as np


data_file = 'data/log_data.json'

with open(data_file, encoding='utf8') as i_f:
    data = json.load(i_f)


exercise = dict()


for log in data:
    for i in log['logs']:
        ex_id = i['exer_id'] - 1
        knowledge_code = i['knowledge_code']
        for j in range(len(knowledge_code)):
            knowledge_code[j] -= 1
        if ex_id not in exercise.keys():
            exercise[ex_id] = knowledge_code

cnt = len(exercise)

graph = np.zeros((cnt,cnt))
np.save('exercise2concept.npy', exercise) 


for key1, value1 in exercise.items():
    for key2, value2 in exercise.items():
        for i in value1:
            for j in value2:
                if i == j:
                    graph[key1][key2] = 1.0

np.save('course_graph_assist09.npy', graph)