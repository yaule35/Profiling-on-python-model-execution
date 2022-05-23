# -*- coding: utf-8 -*-
import json
import sys
import json
import sys
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

PATH = sys.argv[1]

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
])

class_idx = json.load(open("./data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

# Adding a normalization layer for Resnet18.
norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Model configuration
model = nn.Sequential(
    norm_layer,
    models.resnet18(pretrained=True)
).to(device)

model = model.eval()

# Load image
from PIL import Image
img = Image.open(PATH)
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
out = model(batch_t)
_, indices = torch.max(out, 1)
result = [idx2label[i] for i in indices]

print(f'Result:{result[0]}')
sys.exit()