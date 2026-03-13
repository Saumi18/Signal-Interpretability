import torch
import numpy as np
import cv2


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)


    def forward_hook(self, module, input, output):

        self.activations = output


    def backward_hook(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]


    def generate(self, input_tensor, class_idx):

        """
        input_tensor shape:
        (1,1,128,128)
        """

        features = self.model.backbone(input_tensor)

        output = self.model.mod_classifier(features)

        self.model.zero_grad()

        score = output[:, class_idx]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2,3))

        cam = torch.zeros(
            activations.shape[2:],
            device=input_tensor.device
        )

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i]

        cam = cam.detach().cpu().numpy()
        
        cam = np.maximum(cam, 0)
        
        cam = cv2.resize(cam, (128,128))
        cam = cam / (cam.max() + 1e-8)

        return cam
