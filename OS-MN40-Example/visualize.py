
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import shutil
import os, shutil

folder = './vis/target/'
if not os.path.exists(folder):
    os.makedirs(folder)
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

data_root = '../data/'
collec_root = '../data/OS-MN40/'
# dist_path = './cache/ckpts/OS-MN40_2022-02-17-18-41-59/cdist_cosine.txt'
# dist_path = '../data/cdist.txt'
dist_path = './cache/ckpts_source/OS-MN40_2022-02-22-17-29-40/cdist_cosine.txt'
# dist_path = "./cache/ckpts/OS-MN40_2022-02-07-16-53-38/cdist.txt"


#################
batch_size = 48
n_worker = 4
n_class = 8

n_views = 3 # number of views 
init_top = 0 # retrieve topK from [init_top: init_top + topK]
topK = 10 # retrieve topK 
# 1, 3 
query_id = 1

#################


def read_object_list(filename, pre_path):
    object_list = []
    with open(filename, 'r') as fp:
        for name in fp.readlines():
            if name.strip():
                object_list.append(pre_path + name.strip())
    return object_list



def main():
    # init train_loader and test loader
    print("Loader Initializing...\n")

    query_list = read_object_list(data_root + "query.txt", collec_root + "query/")
    target_list = read_object_list(data_root + "target.txt", collec_root + "target/")
    
    dist_mat = np.loadtxt(dist_path)
    dist_mat = dist_mat[query_id]

    priority_list_idx = dist_mat.argsort()[:init_top + topK][init_top:]

    f = plt.figure(figsize=(10,3))
    
    plt.rc('font', size=6) 

    for each in range(n_views):
        query_img = cv2.imread(query_list[query_id] + "/image/h_" + str(each+ 20) + ".png")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./vis/query.png", query_img)
        f.add_subplot(n_views, topK + 1, 1 + each*(topK + 1))
        plt.imshow(query_img)
        if each == 0: 
            plt.title("Query")
            print("Name of Query obj: ", query_list[query_id].split("/")[-1])
        plt.axis('off')
    
    for idx, target_id in enumerate(priority_list_idx):
        # Debug, plot figure
        for each in range(n_views):
            retrieve_img = cv2.imread(target_list[target_id] + "/image/h_" + str(each + 20) + ".png")
            retrieve_img = cv2.cvtColor(retrieve_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./vis/target/target_" + str(init_top + idx + 1) + "_" + str(target_list[target_id].split('/')[-1]) + ".png", retrieve_img)
            f.add_subplot(n_views, topK + 1, idx + 2 + each*(topK + 1))
            plt.imshow(retrieve_img)
            plt.axis('off')
            if each == 0: 
                plt.title("Target " + str(init_top + idx + 1))

    plt.savefig('vis/visualize.png')



if __name__ == '__main__':
    main()
