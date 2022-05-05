from optimiser import Optimiser
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import numpy as np
import gc
from reconstructor import Reconstructor

# set pytorch model directories
os.environ['TORCH_HOME'] = 'models'
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# start optimisation with 10 init evaluations and 10 optim evaluations
optim = Optimiser()
optim.optimise(10, 10)

########
#  or
########

# reconstruct image manually, will result save to working directory if TRUE supplied to reconstruct
dataset = 'dataset'
test_directory = os.path.join(dataset, 'test')

img_dims = 224
batch_size = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load target data and labels
image_transforms = { 
    'test': transforms.Compose([
        transforms.Resize(size=img_dims),
        transforms.CenterCrop(size=img_dims),
        transforms.ToTensor(),
    ])
}
training_batch_size = 256
data = {
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}
test_data_loader = DataLoader(data['test'], batch_size=training_batch_size, shuffle=False)
target_data = []
target_labels = []
chosen_images = [394]
# transform img to tensor
all_inputs = []
all_labels = []
for j, (inputs, labels) in enumerate(test_data_loader):
    all_inputs += inputs
    all_labels += labels
target_data = [all_inputs[i][None,:].to(device) for i in chosen_images][:batch_size]
target_labels = [torch.tensor([all_labels[i].item()]).to(device) for i in chosen_images][:batch_size]
del(all_inputs)
del(all_labels)
gc.collect()

config = dict(
    iterations=20000,
    group_size=4,
    num_classes=10,
    optimiser='AdamW',
    metric='emd',
    img_dims=img_dims,
    model='ResNet-18'
)

params = np.random.uniform([0,0,0,0,0,0], [1,1,1,0.1,1,1])
config['params'] = params
reconstructor = Reconstructor(config)
reconstructor.reconstruct(target_data, target_labels, True)