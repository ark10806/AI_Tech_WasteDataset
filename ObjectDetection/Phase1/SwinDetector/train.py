import torch
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

import Data
import Networks

batch_size = 1
n_epochs = 5
lr = 1e-2
loss = 5

model = Networks.SwinTransformerObjectDetection(n_classes=60)
dataloader = Data.load_data(isTrain=True, batch_size=batch_size)
trainloader = dataloader['train']
validloader = dataloader['val']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'[{torch.cuda.get_device_name()}]')

for epoch in range(n_epochs):
    model.to(device)

    for idx, (images, labels) in enumerate(tqdm(trainloader)):
        images = images.to(device)
        labels = labels.to(device)

        out_cls, out_bb = model(images)
        loss_cls = F.cross_entropy(out_cls, )
