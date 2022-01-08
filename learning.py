"""
import torch
a = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
b = torch.tensor([1,2])
print(a)
print(b)
c = torch.einsum('nhw,n->hw', a, b)
print(c)
x = torch.tensor([[1,2],[3,4]])
y = torch.tensor([[1,2],[3,4]])
z = torch.einsum('ik,kj->ij', x, y)
print(z)
"""

import cv2
import numpy as np
from PIL import Image
"""
a = cv2.imread('coco_panoptic/train2017/000000000009.jpg')
a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
print(a.shape)
cv2.imshow("xxx", a)
cv2.waitKey()

masks1 = np.asarray(Image.open('coco_panoptic/train2017/000000000009.jpg'), dtype=np.uint8)
masks2 = np.asarray(Image.open('coco_panoptic/train2017/000000000009.jpg'), dtype=np.uint32)
print(masks1)
print(masks1.shape)
print(masks2)
print(masks2.shape)
"""
# "annotations": [{"segments_info": [{"id": 3226956, "category_id": 1, "iscrowd": 0, "bbox": [413, 158, 53, 138], "area": 2840}, {"id": 6979964, "category_id": 1, "iscrowd": 0, "bbox": [384, 172, 16, 36], "area": 439},
"""
from panopticapi.utils import rgb2id
ann_path = 'coco_panoptic/annotations/panoptic_train2017/000000000009.png'
masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
masks = rgb2id(masks)
print(masks)
print(masks.shape)
ids = np.array([ann['id'] for ann in ann_info['segments_info']])
print(ids)
masks = masks == ids[:, None, None]

import torch
a = torch.tensor([1,2])
b = torch.tensor([3,4])
c = torch.einsum("x,x -> x",a, b)
print(c)
"""
a = [1,2,3,4,5,6]
print(a[-1])
import torch
b = torch.tensor([1,2])
print(b)




