import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import json
import math
from scipy.ndimage.interpolation import rotate
import random
import cv2
from model import CSRNet
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as tff
from grad_cam import CAM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




data_root = './data/attack_shanghai/'
model_path = './pre_trained/PartA_model.pth.tar'
save_path = './results'

PATCH_SIZE = 0.011
EPOCH = 2
Lambda = 0.01
STEP_SIZE = 0.01
ATTACK_ITERS = 25


part_A_train = os.path.join(data_root,'part_A_final/train_data','images')
part_A_test = os.path.join(data_root,'part_A_final/test_data','images')
part_B_train = os.path.join(data_root,'part_B_final/train_data','images')
part_B_test = os.path.join(data_root,'part_B_final/test_data','images')
train_path_sets = [part_A_train]
test_path_sets = [part_A_test]

train_img_paths = []
for path in train_path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        train_img_paths.append(img_path)

test_img_paths = []
for path in test_path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        test_img_paths.append(img_path)
        

model = CSRNet()
model = model.cuda()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

cam = CAM(model)



transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])



def batch_norm(batch, mean, std):
    
    assert batch.shape[1] == len(mean)
    assert len(mean) == len(std)

    batch_new = tff.normalize(batch, mean, std)
    return batch_new


def submatrix(arr):
    x, y = np.nonzero(arr)
    return arr[x.min():x.max()+1, y.min():y.max()+1]


def init_patch_square(image_size, patch_size, num_patch):
    # get mask
    #patches = []
    #image_size = image_size = image_size[0] * image_size[1]
    #noise_size = image_size*patch_size
    #noise_dim = int(noise_size**(0.5))
    #patch = np.random.rand(1,3,noise_dim,noise_dim)
    patch = np.random.rand(1, 3, 81, 81)
    for i in range(num_patch):
        patches.append(patch)
        
    return patches, patch.shape

def patch_transform(patch, data_shape, patch_shape, image_size, num_patch):
    # get dummy image
    x_list = []
    m_list = []
    for j in range(num_patch):
        x = np.zeros(data_shape)
    
        # get shape
        m_size = patch_shape[-1]
    
        for i in range(x.shape[0]):

            # random rotation
            
            rot = np.random.choice(4)
            for k in range(patch[j][i].shape[0]):
                patch[j][i][k] = np.rot90(patch[j][i][k], rot)
            
            
            # random location
            random_x = np.random.choice(image_size[0])
            if random_x + m_size > x.shape[-2]:
                while random_x + m_size > x.shape[-2]:
                    random_x = np.random.choice(image_size[0])
            random_y = np.random.choice(image_size[1])
            if random_y + m_size > x.shape[-1]:
                while random_y + m_size > x.shape[-1]:
                    random_y = np.random.choice(image_size[1])

            # apply patch to dummy image  
            x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[j][i][0]
            x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[j][i][1]
            x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[j][i][2]

        mask = np.copy(x)
        mask[mask != 0] = 1.0
        x_list.append(x)
        m_list.append(mask)
    
    
    return x_list, m_list


def train(epoch):
    model.eval()   
    for e in range(epoch):        
        for i in range(len(train_img_paths)):
            print("train:epoch%d 第%d张照片"%(e+1,i))
            images = transform(Image.open(train_img_paths[i]).convert('RGB')).cuda()
            gt_file = h5py.File(train_img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
            groundtruth = np.asarray(gt_file['density'])
            data_shape = images.unsqueeze(0).data.cpu().numpy().shape
            if i == 0 and e == 0:               
                patch, patch_shape = init_patch_square(image_size=[images.shape[1],images.shape[2]], patch_size=PATCH_SIZE, num_patch=1)
                #print(patch[0].shape)
            patch, mask = patch_transform(patch, data_shape, patch_shape, [images.shape[1],images.shape[2]], 1)
            
            patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
            patch, mask = patch.cuda(), mask.cuda()
            patch, mask = Variable(patch), Variable(mask)
            
            adv_x, mask, patch = attack(images.unsqueeze(0), groundtruth, patch, mask, train_img_paths[i].split('/')[-1], iters= ATTACK_ITERS)
         
            new_patchs = []
            for w in range(len(patch)):
                masked_patch = torch.mul(mask[w], patch[w])
                patch_ori = masked_patch.data.cpu().numpy()
                new_patch = np.zeros(patch_shape)
                for j in range(new_patch.shape[0]): 
                    for k in range(new_patch.shape[1]): 
                        new_patch[j][k] = submatrix(patch_ori[j][k])
                new_patchs.append(new_patch)
            patch = new_patchs
            
    return patch, patch_shape


def test(patch, patch_shape):
    model.eval()
        
    for i in range(len(test_img_paths)):

        print("test:第%d张照片"%i)
        images = transform(Image.open(test_img_paths[i]).convert('RGB')).cuda()
        gt_file = h5py.File(test_img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
        groundtruth = np.asarray(gt_file['density'])
        data_shape = images.unsqueeze(0).data.cpu().numpy().shape        
        patch, mask = patch_transform(patch, data_shape, patch_shape, [images.shape[1],images.shape[2]], 1)
            
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)

        adv_x = images.unsqueeze(0)
        for j in range(patch.shape[0]):
            adv_x = torch.mul((1-mask[j]),adv_x) + torch.mul(mask[j],patch[j])
        adv_x = torch.clamp(adv_x, 0, 1)


        for _, image in enumerate(adv_x):
            adv_image = transforms.ToPILImage()(image.cpu())
            f_name = test_img_paths[i].split('/')[-1]
            #print(adv_image)
            adv_image.save(os.path.join(save_path, f_name))
        
        new_patchs = []
        for w in range(len(patch)):
            masked_patch = torch.mul(mask[w], patch[w])
            patch_ori = masked_patch.data.cpu().numpy()
            new_patch = np.zeros(patch_shape)
            for j in range(new_patch.shape[0]): 
                for k in range(new_patch.shape[1]): 
                    new_patch[j][k] = submatrix(patch_ori[j][k])
            new_patchs.append(new_patch)
        patch = new_patchs

    
def attack(x, groundtruth, patch, mask, img_name, iters=25):
        
    model.eval()
    
    target = cv2.resize(groundtruth,(groundtruth.shape[1]//8,groundtruth.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
    target = torch.from_numpy(target)
    target = target.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()

    adv_x = x
    for i in range(patch.shape[0]):
        adv_x = torch.mul((1-mask[i]),adv_x) + torch.mul(mask[i],patch[i])
    adv_x = torch.clamp(adv_x, 0, 1)
       
    adv_x = Variable(adv_x.data, requires_grad=True)
    adv_x_norm = batch_norm(adv_x,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])    
    x_out = model(adv_x_norm)   
    s_map, _ = cam(adv_x_norm,'./')
    count = 0 

    
    while True:

        count += 1
        model.zero_grad()

        W = F.sigmoid(target - x_out)
        Loss = torch.sum(W.data * x_out) + Lambda * torch.sum(s_map)
        #print(Loss.data)
        Loss.backward()

        adv_grad = adv_x.grad.clone()

        for i in range(len(patch)):
            
            patch[i] = patch[i] + STEP_SIZE * adv_grad
            adv_x = torch.mul((1-mask[i]),adv_x) + torch.mul(mask[i],patch[i])
        adv_x = torch.clamp(adv_x, 0, 1)

        if count >= iters:
            break
        
        adv_x = Variable(adv_x.data, requires_grad=True)   
        adv_x_norm = batch_norm(adv_x,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  
        x_out = model(adv_x_norm)        
        s_map, _ = cam(adv_x_norm,'./')
    
    return adv_x, mask, patch
    
        
if __name__=="__main__":    
    patch, patch_shape = train(EPOCH)
    test(patch, patch_shape) 

