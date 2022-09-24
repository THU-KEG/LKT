lcd_an = []
lcde_an = []
ncd_an = []

lcd_tr = []
lcde_tr = []
ncd_tr = []


with open('result/model_stsa.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip()
        lcd_an.append(int(line))

with open('../NeuralCD/result/model_stsa.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip()
        ncd_an.append(int(line))

with open('result/model_stsa_nov.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip()
        lcde_an.append(int(line))

with open('result/model_stu_true.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip()
        lcd_tr.append(int(line))

with open('../NeuralCD/result/model_stu_true.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip()
        ncd_tr.append(int(line))

with open('result/model_stu_true_nov.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip()
        lcde_tr.append(int(line))
print(len(lcd_an))
for i in range(len(lcd_an)):
    if lcd_tr[i] > lcde_tr[i] and lcde_an[i] > ncd_an[i]:
        print(i, lcd_an[i], lcde_an[i], ncd_an[i])