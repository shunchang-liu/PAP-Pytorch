import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import os
import sys
from model import CSRNet
from torchvision import datasets, transforms
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class GradCam():
    
    hook_a, hook_g = None, None
    
    hook_handles = []
    
    def __init__(self, model, conv_layer, use_cuda=False):
        
        self.model = model.eval()
        self.use_cuda=use_cuda
        if self.use_cuda:
            self.model.cuda()
        
        self.hook_handles.append(self.model._modules.get(conv_layer)[22].register_forward_hook(self._hook_a))
        
        
        self._relu = True
        self._score_uesd = True
        self.hook_handles.append(self.model._modules.get(conv_layer)[22].register_backward_hook(self._hook_g))
        
    
    def _hook_a(self, module, input, output):

        self.hook_a = output
        #print(self.hook_a.shape)

        
    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def _hook_g(self, module, grad_in, grad_out):

        self.hook_g = grad_out[0]
        #print(self.hook_g.shape)
    
    def _backprop(self, scores):
        
        loss = scores# .requires_grad_(True)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
    
    def _get_weights(self, scores):
        
        self._backprop(scores)
        
        return self.hook_g.squeeze(0).mean(axis=(1, 2))
    
    def __call__(self, input):

        out = self.model(input)
        scores = torch.sum(out)

        weights = self._get_weights(scores)

        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
        cam = F.relu(cam)

        cam_np = cam.data.cpu().numpy()
        
        cam_np = np.maximum(cam_np, 0)
        cam_np = cam_np - np.min(cam_np)
        cam_np = cam_np / np.max(cam_np)
    
        #print(loss_mul_rf)
        return cam, cam_np
        


class CAM:
    
    def __init__(self, model):
        self.grad_cam = GradCam(model=model, conv_layer='frontend', use_cuda=True)
        self.log_dir = "./"
        self.count = 0
        
    def __call__(self, img, log_dir):
        self.log_dir = log_dir
        #img = img / 255
        raw_img = img.data.cpu().numpy()[0].transpose((1, 2, 0))
        #input = self.preprocess_image(img)
        input = img
        ret, mask = self.grad_cam(input)
        #self.show_cam_on_image(raw_img, mask)
        return ret, mask
        
    def preprocess_image(self, img):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img
        for i in range(3):
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] - means[i]
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] / stds[i]
        input = preprocessed_img.requires_grad_(True)
        return input


    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]
        heatmap = cv2.resize(heatmap,(1024,709))
        heatmap = np.float32(heatmap) / 255
        cam = np.float32(img) + heatmap
        cam = cam / np.max(cam)
        Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'test_cam.jpg'))
        Image.fromarray(np.uint8(255 * heatmap)).save(os.path.join(self.log_dir, 'cam.jpg'))
        # cv2.imwrite("cam.jpg", np.uint8(255 * cam))

    