import torch
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import ValTestDataLoader
from model import Net


# can be changed according to config.txt
exer_n = 519
knowledge_n = 101
student_n = 2093
video_n = 857
with open('data/new_lkt2/content_embedding.pth', 'rb') as f:
    emb = torch.load(f)
device = torch.device('cuda:7')
    
data_path = 'data/new_lkt2/'
with open('ran.pth', 'rb') as r:
    ran = torch.load(r)

hidden_dim = 128

def test(epoch):
    data_loader = ValTestDataLoader('test')
    net = Net(student_n, exer_n, knowledge_n, hidden_dim, video_n, data_path, ran, emb, device)
    print('testing model...')
    data_loader.reset()
    load_snapshot(net, 'model/model_new_epoch' + str(epoch))
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    stu = []
    stsa = []
    stu_true = []
    stuid = []


    while not data_loader.is_end():        
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, input_video_ids = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, input_video_ids = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device), input_video_ids.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, device, input_video_ids)
        output = output.view(-1)
        # compute accuracy
        stu_cnt = 0
        true_cnt = 0
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
                stu_cnt += 1

        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5):
                true_cnt += 1
        id = input_stu_ids.to(torch.device('cpu')).tolist()[0]
        if id == 1247:
            ex_ids = input_exer_ids.to(torch.device('cpu')).tolist()
            kn_ids = input_knowledge_embs.to(torch.device('cpu')).tolist()
            for i in range(len(labels)):
                if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                    stu.append(ex_ids[i])
        stsa.append(stu_cnt)
        stu_true.append(true_cnt)
        stuid.append(input_stu_ids.to(torch.device('cpu')).tolist()[0])
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)

    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch, accuracy, rmse, auc))
    with open('result/model_test.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch, accuracy, rmse, auc))
    with open('result/model_stu.txt', 'w', encoding='utf8') as f:
        f.write(str(stu))
    with open('result/model_stsa.txt', 'w', encoding='utf8') as f:
        for i in stsa:
            f.write(str(i) + '\n')
    with open('result/model_stu_true.txt', 'w', encoding='utf8') as f:
        for i in stu_true:
            f.write(str(i) + '\n')
    with open('result/model_stuid.txt', 'w', encoding='utf8') as f:
        for i in stuid:
            f.write(str(i) + '\n')
    



def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def get_status(epoch):
    '''
    An example of getting student's knowledge status
    :return:
    '''
    net = Net(student_n, exer_n, knowledge_n, hidden_dim, video_n, data_path, ran, emb, device)
    load_snapshot(net, 'model/model_new_epoch' + str(epoch))
    # load_snapshot(net, 'model/model_epoch12')       # load model
    net.eval()

    with open('result/student_stat.txt', 'w', encoding='utf8') as output_file:
        status = net.get_knowledge_status(torch.LongTensor([1247])).tolist()[0]
        output_file.write(str(status) + '\n')
        # for stu_id in range(student_n):
        #     # get knowledge status of student with stu_id (index)
        #     status = net.get_knowledge_status(torch.LongTensor([stu_id])).tolist()[0]
        #     output_file.write(str(status) + '\n')


def get_exer_params(epoch):
    '''
    An example of getting exercise's parameters (knowledge difficulty and exercise discrimination)
    :return:
    '''
    net = Net(student_n, exer_n, knowledge_n, hidden_dim, video_n, data_path, ran, emb, device)
    load_snapshot(net, 'model/model_new_epoch' + str(epoch))
    # load_snapshot(net, 'model/model_epoch12')       # load model
    net.eval()
    net = net.to(device)

    exer_params_dict = []
    k_difficulty = net.get_exer_params(torch.LongTensor([347]).to('cuda:7'))
    k_difficulty = torch.max(k_difficulty)
    # k_difficulty = k_difficulty.to(torch.device('cpu')).tolist()[0]
    
    # for exer_id in range(exer_n):
    #     # get knowledge difficulty and exercise discrimination of exercise with exer_id (index)
    #     k_difficulty, e_discrimination = net.get_exer_params(torch.LongTensor([exer_id]))
    #     exer_params_dict[exer_id + 1] = (k_difficulty.tolist()[0], e_difficulty.tolist()[0])
    with open('result/exer_params.txt', 'w', encoding='utf8') as o_f:
        o_f.write(str(k_difficulty))


if __name__ == '__main__':
    if (len(sys.argv) != 2) or (not sys.argv[1].isdigit()):
        print('command:\n\tpython predict.py {epoch}\nexample:\n\tpython predict.py 70')
        exit(1)

    # global student_n, exer_n, knowledge_n
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n, video_n = list(map(eval, i_f.readline().split(',')))

    # test(int(sys.argv[1]))
    # get_status(int(sys.argv[1]))
    get_exer_params(int(sys.argv[1]))
