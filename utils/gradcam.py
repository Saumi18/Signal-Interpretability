import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:

    def __init__(self,model,layer):

        self.model = model
        self.layer = layer

        self.gradients = None
        self.activations = None

        layer.register_forward_hook(self.forward_hook)
        layer.register_backward_hook(self.backward_hook)


    def forward_hook(self,module,input,output):

        self.activations = output


    def backward_hook(self,module,grad_in,grad_out):

        self.gradients = grad_out[0]


    def generate(self,input_image,class_idx):

        output = self.model(input_image)[2]

        self.model.zero_grad()

        output[:,class_idx].backward()

        grads = self.gradients
        acts = self.activations

        weights = torch.mean(grads,dim=(2,3))

        cam = torch.zeros(acts.shape[2:]).to(input_image.device)

        for i,w in enumerate(weights[0]):

            cam += w*acts[0,i]

        cam = cam.cpu().detach().numpy()

        cam = np.maximum(cam,0)

        cam = cv2.resize(cam,(128,128))

        cam = cam / cam.max()

        return cam