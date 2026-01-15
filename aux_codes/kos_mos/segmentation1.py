# By Oleksiy Grechnyev, train and test a semantic segmentation network

import sys

import glob
import tqdm

import numpy as np
import cv2 as cv

# Solve cv + matplotlib conflict
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torchvision


########################################################################################################################
def print_it(a, name: str = ''):
    m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    # m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def downsize(img):
    img = cv.resize(img, None, None, 0.5, 0.5)
    return img


def auto_downsize(img):
    while max(img.shape[0], img.shape[1]) > 1920:
        img = downsize(img)
    return img


########################################################################################################################
def img2tens(img):
    assert isinstance(img, np.ndarray)
    mean =[0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img / 255
    img = (img - mean) / std
    img = img.transpose(2, 0, 1).astype('float32')
    t = torch.from_numpy(img)
    return t


########################################################################################################################
def tens2img(t):
    assert isinstance(t, torch.Tensor)
    t = t.detach().cpu()
    if len(t.shape) == 4:
        t = t[0]
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    t = t.permute(1, 2, 0)
    t = (t * std + mean) * 255
    t = t.to(dtype=torch.uint8)
    # print_it(t, 't')
    return t.numpy()


########################################################################################################################
class DSet(torch.utils.data.Dataset):
    def __init__(self, dset_dir):
        self.flist_img = sorted(glob.glob(dset_dir + '/images/*.png'))
        self.flist_mask = sorted(glob.glob(dset_dir + '/masks/*.png'))
        assert len(self.flist_img) == len(self.flist_mask)
        
    def __len__(self):
        return len(self.flist_img)
    
    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        
        img_frame = cv.imread(self.flist_img[idx])
        img_mask = cv.imread(self.flist_mask[idx], cv.IMREAD_GRAYSCALE)
        assert img_frame is not None and img_mask is not None
        
        tens_frame = img2tens(img_frame)
        tens_mask = torch.tensor(img_mask, dtype=torch.long)
        return tens_frame, tens_mask
        

########################################################################################################################
def show_seg_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    img_vis = np.zeros((*mask.shape[:2], 3), dtype='uint8')
    colors = [
        (0, 0, 0),       # Black BG
        (0, 0xff, 0),    # Green hedge
        (0, 0, 0xff),    # Red obstacle
        (0xff, 0, 0),    # Blue POST
        (0, 0xff, 0xff), # Yellow tree trunk
    ]
    
    for i in range(mask.max() + 1):
        img_vis[mask == i, :] = colors[i]
        
    return img_vis


########################################################################################################################
def concat_img(img_list):
    pad_x, pad_y = 100, 100
    n = len(img_list)
    im_h, im_w = img_list[0].shape[:2]

    tot_w = n * im_w + (n + 1) * pad_x
    tot_h = im_h + 2 * pad_y

    img = np.ones((tot_h, tot_w, 3), dtype='uint8') * 255
    for i in range(n):
        x = pad_x + i * (im_w + pad_x)
        y = pad_y
        img[y:y + im_h, x:x + im_w, :] = img_list[i]

    return img


########################################################################################################################
class Trainer:
    def __init__(self):
        self.device = torch.device('cuda')
        self.batch_size = 1
        self.n_epochs = 1000
        self.class_names = ['background', 'hedge', 'obstacle', 'post', 'tree trunk']
        self.num_classes = len(self.class_names)
        self.losses_train = None
        self.losses_val = None
        
        # Model etc.
        self.model = self.create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.class_weights = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0], device=self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Datasets and loaders
        self.dset_train = DSet('/home/seymour/prog-fun/py2/fun_seg/data/dset_zero')
        self.dset_val = DSet('/home/seymour/prog-fun/py2/fun_seg/data/dset_zero')
        self.dloader_train = torch.utils.data.DataLoader(self.dset_train, batch_size=self.batch_size)
        self.dloader_val = torch.utils.data.DataLoader(self.dset_val, batch_size=self.batch_size)
        
    def create_model(self):
        # weights = torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT
        # Should iitialize the resnet backbone by default
        model = torchvision.models.segmentation.fcn_resnet101(num_classes=self.num_classes)
        model.to(device=self.device)
        return model
    
    def val_one(self):
        self.model.eval()
        losses = []
        with torch.inference_mode():
            for frame, mask in self.dloader_val:
                frame = frame.to(self.device)
                mask = mask.to(self.device)
                out = self.model(frame)['out']
                loss = self.criterion(out, mask)
                losses.append(loss.item())
        return np.mean(losses)

    def train_one(self):
        self.model.train()
        losses = []
        for frame, mask in tqdm.tqdm(self.dloader_train):
            frame = frame.to(self.device)
            mask = mask.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(frame)['out']
            loss = self.criterion(out, mask)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        return np.mean(losses)

    def training_loop(self):
        self.losses_train = []
        self.losses_val = []

        for i_epoch in range(self.n_epochs):
            loss_train = self.train_one()
            loss_val = self.val_one()
            print(f'{i_epoch} : loss_train={loss_train}, loss_val={loss_val}')
            self.losses_train.append(loss_train)
            self.losses_val.append(loss_val)

    def run_test(self, dset):
        self.model.eval()
        for idx, (frame, mask) in enumerate(dset):
            frame = frame.unsqueeze(0).to(self.device)
            out = self.model(frame)['out']
            pred = out[0].detach().argmax(axis=0)
            fname = dset.flist_img[idx].split('/')[-1]
            text = f'{idx}: {fname}'

            vis_mask_gt = show_seg_mask(mask)
            vis_mask_pred = show_seg_mask(pred)
            vis = concat_img([vis_mask_gt, vis_mask_pred])
            cv.putText(vis, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0xff, 0, 0), 2)
            cv.imshow('vis', vis)
            if 27 == cv.waitKey(0):
                break


########################################################################################################################
def main1():
    trainer = Trainer()
    print('INIT loss=', trainer.val_one())
    trainer.training_loop()
    trainer.run_test(trainer.dset_val)

    plt.plot(trainer.losses_train, label='loss_train')
    plt.plot(trainer.losses_val, label='loss_val')
    plt.legend()
    plt.tight_layout()
    plt.show()


########################################################################################################################
if __name__ == '__main__':
    main1()
