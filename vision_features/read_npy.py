import json, torch

import numpy as np

if __name__ == '__main__':
    d = np.load("detr.npy")[0]
    print(d.shape) # (11208, 100, 256)

    # b = json.load(open("../data/instruct_captions.json"))['captions']
    # print(len(b))  # 11208
    # p = torch.load("../vision_features_test/detr.pth")
    # print(p.shape)
    # d = torch.tensor(d[0])
    print(d, d[1:] - d[:99])
