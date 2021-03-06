import pickle
from utils import nn_score, ndcg_score, map_score, anmrr_score
import numpy as np

with open('/home/pbdang/Contest/SHREC22/OpenSet/MVTN/models/OS-MN40-Miss1/query_label_idx.pickle', 'rb') as handle:
    query = pickle.load(handle)

with open('/home/pbdang/Contest/SHREC22/OpenSet/MVTN/models/OS-MN40-Miss1/target_label_idx.pickle', 'rb') as handle:
    target = pickle.load(handle)


dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/ckpts_source_b0/OS-MN40_2022-02-28-09-56-14/assemble9.txt"
# dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/ckpts_source_b0/OS-MN40_2022-02-27-17-29-38/assemble8.txt"
# dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/ckpts_target/OS-MN40_2022-02-25-07-12-22/assemble7.txt"
dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/ckpts_source_b1/OS-MN40_2022-02-28-13-14-02/assemble13.txt"
dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/ckpts_source_b1/OS-MN40_2022-02-28-13-14-02/cdist_cosine_b1.txt"
# dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/ckpts_source_b5/OS-MN40_2022-02-28-19-13-54/assemble15.txt"
# dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/ckpts_source_b2/OS-MN40_2022-03-01-05-39-48/cdist_cosine_b2.txt"
dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/ckpts_target_test/OS-MN40_2022-03-02-08-43-58/cdist_cosine_1.txt"
# dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/miss_ckpts_source/OS-MN40_2022-03-03-16-19-08/cdist_cosine.txt"
dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/miss_ckpts_source/OS-MN40_2022-03-03-20-00-01_1024/cdist_cosine.txt"
dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/miss_ckpts_source/OS-MN40_2022-03-04-03-05-00_1024/cdist_cosine_2.txt"
dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/miss_ckpts_source/OS-MN40_2022-03-04-03-05-00_1024/ensemble_miss_1.txt"
dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/miss_ckpts_source/OS-MN40_2022-03-04-05-45-23_resnet/cdist_cosine.txt"
dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/ckpts_source_b1/OS-MN40_2022-02-28-13-14-02/ensemble_15ero_.txt"
# dist_path = "/home/nero/SHREC2022/openset/SHREC2022-OpenSet/cdist_cosine (1).txt"
# dist_path = "/home/nero/SHREC2022/openset/SHREC2022-OpenSet/cache/ckpts_source_b2/OS-MN40_2022-03-04-02-06-55/cdist_cosine.txt"
dist_path = "/home/pbdang/Contest/SHREC22/OpenSet/OS-MN40-Example/cache/ckpts_source_b1_att/OS-MN40_2022-03-04-10-27-54_att_pre/cdist_cosine_1.txt"
dist_mat = np.loadtxt(dist_path)
# print(dist_mat.shape)
# import pdb; pdb.set_trace()


MAP_score = map_score(dist_mat, query, target)
print("Done MAP score!")
NN_score = nn_score(dist_mat, query, target)
print("Done NN score!")
NDCG_score = ndcg_score(dist_mat, query, target)
print("Done NDCG score!")
ANMRR_score = anmrr_score(dist_mat, query, target)
print("Done ANMRR score!")
print("----------------------------------------")

print("MAP_score: ", MAP_score)
print("NN_score: ", NN_score)
print("NDCG_score: ", NDCG_score)
print("ANMRR_score: ", ANMRR_score)



