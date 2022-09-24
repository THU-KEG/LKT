import json
import numpy as np
stu_stats = []


def theta(a, b):
    if float(a) > float(b):
        return 1
    return 0

with open('result/student_stat.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip()
        line = line[1:-1].split(',')
        stu_stats.append(line)

# stu_stats = np.random.random((2093,101))

stu_num = len(stu_stats)

with open('data/new_lkt2/group_trainset_by_knowledge.json', encoding='utf8') as i_f:
    text = json.load(i_f)
    doasum = 0
    for i in text:
        kn_id = i['knowledge_code']
        z = [[0] * stu_num for j in range(stu_num)]
        z_sum = 0
        doak = 0
        for j in range(stu_num):
            for k in range(stu_num):
                z[j][k] = theta(stu_stats[j][kn_id], stu_stats[k][kn_id])
                z_sum += z[j][k]
        for ex in i['exercises']:
            for x in ex['logs']:
                for y in ex['logs']:
                    if theta(x['score'], y['score']) == 1:
                        doak += z[int(x['user_id'])][int(y['user_id'])]
        doak /= z_sum
        doasum += doak
        print(doak*1000)
    print(doasum/101)
                        