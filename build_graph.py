import numpy as np
import argparse
import json
from collections import Counter
import math
import random
import tqdm
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA


def build_concept_graph():
    prefix = 'data/new_lkt2/'
    n = 2093 + 519 + 854 + 101
    # n = 2093 + 519    
    with open(prefix+'new_lkt_graph.txt', 'r', encoding='utf-8') as f:
        graph = np.eye(n)
        for line in f:
            line = line.strip()
            line = line.split()
            if int(line[0]) < 2093  and int(line[1]) < 2093 + 519:
                graph[int(line[0])][int(line[1])] = 1.0
                # graph[int(line[1])][int(line[0])] = 0.1
            elif int(line[0]) < 2093  and int(line[1]) >= 2093 + 519:
                graph[int(line[0])][int(line[1])] = 0.2
            else:
                graph[int(line[0])][int(line[1])] = 1.0

        np.save(prefix+'graph.npy', graph)

def build_course_graph():
    prefix = 'data/new_lkt2/'
    n = 545 + 1319 + 101
    # n = 2093 + 519    
    with open(prefix+'course_graph_near.txt', 'r', encoding='utf-8') as f:
        graph = np.eye(n)
        for line in f:
            line = line.strip()
            line = line.split()
            graph[int(line[0])][int(line[1])] = float(line[2])
            graph[int(line[1])][int(line[0])] = float(line[2])
            

        np.save(prefix+'course_graph_near.npy', graph)

def build_embedding():
    prefix = 'data/new_lkt2/'
    with open(prefix+'id2concept.txt', 'r') as f1:
        with open(prefix+'concept_embedding.pth', 'rb') as f2:
            concept_embedding = torch.load(f2)
            node_embedding = []
            for line in f1:
                line = line.strip().split()
                id = line[0]
                concepts = line[1:]
                concepts_embedding = [concept_embedding[int(i)] for i in concepts]
                embedding = torch.mean(torch.stack(concepts_embedding), 0)
                node_embedding.append(embedding.numpy())
            for i in range(519 + 854, 519 + 854+101):
                embedding = concept_embedding[i-519-854]
                node_embedding.append(embedding.numpy())
            node_embedding = torch.FloatTensor(node_embedding)

            with open('node_embedding.pth', 'wb') as f:
                torch.save(node_embedding, f)
            
def build_content_embedding():
    prefix = 'data/new_lkt2/'
    with open(prefix+'content_embedding.pth', 'rb') as f:
        content_embedding = torch.load(f)
    with open(prefix+'concept_embedding.pth', 'rb') as f:
        concept_embedding = torch.load(f)
    node_embedding = torch.cat((content_embedding, concept_embedding), 0)
    print(node_embedding.size())

    with open('node_content_embedding.pth', 'wb') as f:
        torch.save(node_embedding, f)

def build_graph_sim():
    with open('data/new_lkt2/content_embedding.pth', 'rb') as f:
        emb = torch.load(f)
    ran = torch.rand(1, 768)
    new_emb = torch.cat((emb, ran), 0)
    ex_num, vi_num = 519, 857
    n = ex_num + vi_num + 1
    sim = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            sim[i][j] = torch.cosine_similarity(new_emb[i], new_emb[j], dim=0)

    sim = torch.from_numpy(sim)
    with open('content_sim.pth', 'wb') as f:
        torch.save(sim, f)

def main():
    # build_concept_graph()
    # build_content_embedding()
    # build_course_graph()
    build_graph_sim()

if __name__ == '__main__':
    main()