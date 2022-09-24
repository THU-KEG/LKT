import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader_new2 import TrainDataLoader, ValTestDataLoader
from model import Net, My_loss
import os
import scipy.sparse as sp
import torch.nn.functional as F
from torch import autograd

# can be changed according to config.txt2093,519,101,854
exer_n = 519
knowledge_n = 101
student_n = 2093
video_n = 1319
zong = exer_n + knowledge_n + video_n
# can be changed according to command parameter
device = torch.device(('cuda:4') if torch.cuda.is_available() else 'cpu')
epoch_n = 20
hidden_dim = 300
# emb = torch.rand((zong, 32))

with open('data/new_lkt2/content_embedding.pth', 'rb') as f:
    emb = torch.load(f)
    
# ran = torch.rand(1,16)
# with open('ran.pth', 'wb') as r:
#     torch.save(ran, r)
with open('ran.pth', 'rb') as r:
    ran = torch.load(r)

def train(data_path):
    data_loader = TrainDataLoader()
    torch.autograd.set_detect_anomaly(True)

    net = Net(student_n, exer_n, knowledge_n, hidden_dim, video_n, data_path, ran, emb, device)
    n = student_n + exer_n + video_n
    # n = student_n + exer_n
    net = net.to(device)
    # optimizer = optim.Adadelta(net.parameters(), lr=1.0)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    # optimizer = optim.SGD(net.parameters(), lr=0.02)
    print('training model...')


    loss_function = nn.NLLLoss()
    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, input_video_ids, kn_ids = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, input_video_ids, kn_ids = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device), input_video_ids.to(device), kn_ids.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, device, input_video_ids, kn_ids)
            # output = output_1.squeeze(dim=-1)
            # loss = loss_function(output, labels)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)
            # loss = loss_function(output_1.view(-1), labels)
            # grad_penalty = 0
            # loss = loss_function(torch.log(output), labels.reshape(labels.size(0), 1))
            # loss = F.cross_entropy(output_1, labels)
            loss = loss_function(torch.log(output), labels)
            
            loss.backward()
            optimizer.step()
            # net.apply_clipper()
            # del input_stu_ids, input_exer_ids, input_knowledge_embs, labels
            # torch.cuda.empty_cache()
            
            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0
        # validate and save current model every epoch
        rmse, auc = validate(net, epoch, data_path)
        save_snapshot(net, 'model/model_new_epoch' + str(epoch + 1))
        # save_snapshot(net, 'model/model_epoch' + str(epoch + 1))


def validate(model, epoch, data_path):
    data_loader = ValTestDataLoader('test')
    net = Net(student_n, exer_n, knowledge_n, hidden_dim, video_n, data_path, ran, emb, device)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()
    
    
    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, input_video_ids, kn_ids = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, input_video_ids, kn_ids = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device), input_video_ids.to(device), kn_ids.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, device, input_video_ids, kn_ids)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1

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
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])
    data_path = 'data/new_lkt2/'
    # global student_n, exer_n, knowledge_n, device
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n, video_n = list(map(eval, i_f.readline().split(',')))

    train(data_path)
