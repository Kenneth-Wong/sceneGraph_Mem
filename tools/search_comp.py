import _init_paths
import os.path as osp
from model.config import cfg
import numpy as np

OUTPUT_DIR = osp.join(cfg.ROOT_DIR, 'output/vgg16/visual_genome_test_rel/default')
MODEL = ['10_20_LFFbaseline_fr9k_rpn_vgg16_iter_80000.ckpt_update',
         '10_20_LFFbaseline_iscbox_fr9k_rpn_vgg16_iter_80000.ckpt_update',
         '6a12_16_MemLoss_iter3_VertEdgeGru_fr9k_rpn_vgg16_iter_80000.ckpt_update']

mode = ['pred_cls', 'sg_cls', 'sg_det']


def loadData(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    data = dict()
    for line in lines:
        item = line.strip().split()
        data[int(item[0])] = np.array([float(k) for k in item[1:]])
    return data


def comp(base, tar, mod):
    baseF = osp.join(OUTPUT_DIR, base, mod + '.txt')
    tarF = osp.join(OUTPUT_DIR, tar, mod + '.txt')
    baseD = loadData(baseF)
    tarD = loadData(tarF)
    ids = []
    for id in tarD:
        if id not in baseD:
            continue
        if tarD[id][0] - baseD[id][0] > 0.5 and tarD[id][1] > baseD[id][1] and tarD[id][2] > baseD[id][2]:
            ids.append(id)
    print(ids)
    print(len(ids))


if __name__ == '__main__':
    comp(MODEL[0], MODEL[2], 'sg_det')