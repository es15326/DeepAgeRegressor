import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

import data
from metrics import *
from nets import EfficientNet, ResNet, ResNext, SENet50, ViT


def eval(y_pred, y_true):
    print(f'ResNext: MAE: {np.abs(y_true - y_pred).mean()}')
    AAR, MAE, *_, sigmas, maes = aar(y_true, y_pred)
    print(f'ResNext: AAR on valdiation: {AAR}')
    maes_str = '\t'.join([f'{m:.03f}' for m in maes])
    sigmas_str = '\t'.join([f'{np.sqrt(m):.03f}' for m in sigmas])

    print("Summary: MAE\t" + "\t".join(["MAE" + str(i) for i in range(1, 9)]) + "\tAAR")
    print(f'Summary: {MAE:.03f}\t{maes_str}\t{AAR:.03f}')
    print(f'Summary: {0:.03f}\t{sigmas_str}\t{0:.03f}')

# ckpt = torch.load('weights/resnext50_32x4d-epoch=027-val_arr=7.96.ckpt', map_location=torch.device('cpu'))

# for ResNext
model = ResNext()
ckpt = torch.load('data/checkpoints/WAAR-Loss-ResNext-synthetic-data-epoch=047-val_aar=8.05.ckpt', map_location=torch.device('cpu'))
# ckpt = torch.load('data/checkpoints/MSE-Loss-ResNext-epoch=024-val_aar=8.01.ckpt', map_location=torch.device('cpu'))
# ckpt = torch.load('/cluster/VAST/civalab/results/guess-the-age/data/checkpoints/WAAR-Loss-ResNext-epoch=025-val_aar=8.01.ckpt', map_location=torch.device('cpu'))
# ckpt = torch.load('data/checkpoints/WMSE-Loss-ResNext-epoch=023-val_aar=7.94.ckpt', map_location=torch.device('cpu'))

# # For ResNet
#####model = ResNet()
####ckpt = torch.load('data/checkpoints/MSE-Loss-ResNet-epoch=017-val_aar=7.81.ckpt', map_location=torch.device('cpu'))
# ckpt = torch.load('data/checkpoints/AAR-Loss-ResNet-epoch=027-val_aar=8.07.ckpt', map_location=torch.device('cpu'))
# ckpt = torch.load('data/checkpoints/MSE-Loss-ResNet-epoch=025-val_aar=7.99.ckpt', map_location=torch.device('cpu'))

# For SE-Net50 (pretrained on vggface)
# model = SENet50()
# ckpt = torch.load('data/checkpoints/MSE-Loss-SENet-epoch=053-val_aar=8.13.ckpt', map_location=torch.device('cpu'))
# ckpt = torch.load('data/checkpoints/WAAR-Loss-SENet-epoch=052-val_aar=8.03.ckpt', map_location=torch.device('cpu'))

# For SE-Net50 (pretrained on vggface)
# model = EfficientNet()
# ckpt = torch.load('data/checkpoints/AAR-Loss-EfficientNet-epoch=018-val_aar=8.01.ckpt', map_location='cpu')

new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(new_dict)

# ViT
# model = ViT()
# model.load_from_checkpoint('data/checkpoints/Aligned-VIT-epoch=009-val_arr=7.56.ckpt')

model.eval()
model.cuda()

valset = data.GTADataset('data/test.csv', '/cluster/VAST/civalab/results/guess-the-age/training_caip_contest',
                            transform=data.EVAL_TRANSFORMS,
                            return_paths=True)
N = len(valset)
batch_size = 256
dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs = np.zeros((N, 1))
ages = np.zeros((N, 1))

pbar = tqdm.tqdm(dataloader)
for i, (imgs, age, paths) in enumerate(pbar):
    age = age.detach().cpu().numpy().reshape(-1, 1)
    
    idx = i * batch_size
    imgs = imgs.cuda()
    out = (model(imgs) * 81).detach().cpu().numpy()
    out = np.round(out).astype('int').clip(1, None)
    
    outs[idx:min(N, idx + batch_size),:] = out
    ages[idx:min(N, idx + batch_size),:] = age
    
    AAR, MAE, *_, sigmas, maes = aar(age, out)
    pbar.set_description(f'AAR: {AAR:.3f} - MAE: {MAE:.3f}')


eval(outs, ages)
