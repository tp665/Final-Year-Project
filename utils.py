import torch
import torch.nn.functional as F
import numpy as np
import warnings

def ucb(x, gp, confidence=2.576):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, std = gp.predict(x, return_std=True)
    return mean + confidence * std

def verify_config(config):
    required_items = [
        'iterations',
        'group_size',
        'num_classes',
        'img_dims',
        'params'
    ]  
    missing_items = [i for i in required_items if i not in config]
    if(len(missing_items) > 0):
        for i in missing_items:
            print("Missing", i)
        return False
    return True

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def total_variation(images):
    result = 0
    for image in images:
        batch_size, channels, height, width = image.size()
        tv_height = torch.pow(image[:,:,1:,:]-image[:,:,:-1,:], 2).sum()
        tv_width = torch.pow(image[:,:,:,1:]-image[:,:,:,:-1], 2).sum()
        result += torch.div(torch.add(tv_height, tv_width),torch.mul(torch.mul(batch_size,channels), torch.mul(height,width)))
    return result

def group_regularisation(image, group_images):
    result = torch.mean(torch.stack(group_images), dim=0)
    return torch.norm(torch.sub(image, result), p=2)

class BatchNormHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.mean_var_hook)

    def mean_var_hook(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        mean_var = [mean, var]
        self.mean_var = mean_var

    def close(self):
        self.hook.remove()
