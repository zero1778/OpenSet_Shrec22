import os,pickle 
from tqdm import tqdm
data_path = "/home/pbdang/Contest/SHREC22/OpenSet/data/OS-MN40/"

class_idx = {
    'night_stand': 0, 'range_hood': 1, 'plant': 2, 'chair': 3, 'tent': 4,
    'curtain': 5, 'piano': 6, 'dresser': 7, 'desk': 8, 'bed': 9,
    'sink': 10,  'laptop':11, 'flower_pot': 12, 'car': 13, 'stool': 14,
    'vase': 15, 'monitor': 16, 'airplane': 17, 'stairs': 18, 'glass_box': 19,
    'bottle': 20, 'guitar': 21, 'cone': 22,  'toilet': 23, 'bathtub': 24,
    'wardrobe': 25, 'radio': 26,  'person': 27, 'xbox': 28, 'bowl': 29,
    'cup': 30, 'door': 31,  'tv_stand': 32,  'mantel': 33, 'sofa': 34,
    'keyboard': 35, 'bookshelf': 36,  'bench': 37, 'table': 38, 'lamp': 39, 
}
cate_id = { 17: 0,
            12: 1, 
            19: 2, 
            35: 3, 
            16: 3, 
            0 : 5, 
            10: 6, 
            38: 7}

with open('/home/pbdang/Contest/SHREC22/OpenSet/MVTN/models/OS-MN40-Miss1/query_label.pickle', 'rb') as handle:
    query = pickle.load(handle)

with open('/home/pbdang/Contest/SHREC22/OpenSet/MVTN/models/OS-MN40-Miss1/target_label.pickle', 'rb') as handle:
    target = pickle.load(handle)

test_data = []

# import pdb; pdb.set_trace()
for typef in os.listdir(data_path):
    if typef == "query":
        for each in tqdm(os.listdir(data_path + typef + "/")):
            tmp = {}
            if query[each] in cate_id.keys():
                tmp["path"] = data_path + typef + "/" + each  
                tmp["label"] = cate_id[query[each]]
            else:
                tmp["path"] = data_path + typef + "/" + each  
                tmp["label"] = 8

            test_data.append(tmp)
            
    elif typef == "target":
        for each in tqdm(os.listdir(data_path + typef + "/")):
            tmp = {}
            if target[each] in cate_id.keys():
                tmp["path"] = data_path + typef + "/" + each  
                tmp["label"] = cate_id[target[each]]
            else:
                tmp["path"] = data_path + typef + "/" + each  
                tmp["label"] = 8
                
            test_data.append(tmp)

import pdb; pdb.set_trace()
with open(data_path + 'target_data.pickle', 'wb') as handle:
    pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

