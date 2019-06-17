import os
import json
import numpy as np


DATA_DIR = '/home/wangwenbin/dataset/visualgenome/relationships.json'
relationships_data = json.load(open(DATA_DIR))

total_rel = 0.
overlap = 0.

for rs in relationships_data:
    for r in rs['relationships']:
        sub = [r['subject']['x'], r['subject']['y'], r['subject']['x'] + r['subject']['w'], \
               r['subject']['y'] + r['subject']['h']]
        obj = [r['object']['x'], r['object']['y'], r['object']['x'] + r['object']['w'], \
               r['object']['y'] + r['object']['h']]
        total_rel += 1
        x11, y11, x12, y12 = sub
        x21, y21, x22, y22 = obj
        iw = np.minimum(x12, x22) - np.maximum(x11, x21) + 1
        ih = np.minimum(y12, y22) - np.maximum(y11, y21) + 1
        if iw > 1 and ih > 1:
            overlap += 1

print(overlap / total_rel)
