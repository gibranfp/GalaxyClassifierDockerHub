# %% [markdown]
# # FINAL TRAINIG DRAFT

# %%
import os
os.chdir('../')
os.getcwd()

from numpy import ceil
# PyTorch stuff
import torch
from torch.nn import CrossEntropyLoss 
from timm.loss import LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from torchsummary import summary
from torch.cuda import is_available as gpu_available
# Custom made funs
from torch_galaxy.networks import *
from torch_galaxy.data_prep import galaxy_dataset_train, galaxy_dataset_eval
from torch_galaxy.random_center import dist_info_base
from torch_galaxy.training_new import train_network, CosineDecaySchedulerLR

# %% [markdown]
# ## 1. DATA

# %%
# Distribution info
dist_info = dist_info_base.copy()
dist_info['num_samples'] = 30
dist_info['dist'] = 'beta'
dist_info['alpha_param'] = 1.487
dist_info['beta_param'] = 2.136333
dist_info

# %%
# Datasets
d_train = galaxy_dataset_train(data_path = '../data/Nair_MaNGA_sample.csv',
                               dist_info = dist_info)

d_valid = galaxy_dataset_eval(data_path = '../data/Nair_MaNGA_sample.csv',
                              dist_info = dist_info)
# Dataloaders
train_dataloader = DataLoader(d_train, batch_size = 500, shuffle = True, num_workers = 5)
valid_dataloader = DataLoader(d_valid, batch_size = 500, shuffle = False, num_workers = 5)

# %% [markdown]
# ## 2. NETWORK

# %%
network = get_network(num_classes = 6)
summary(network, (3, 50, 50), device = 'cpu')

# %% [markdown]
# ## 3. OPT AND LOS_FN

# %%
#loss_fn = CrossEntropyLoss(label_smoothing = 0.1)
loss_fn = LabelSmoothingCrossEntropy(smoothing = 0.1)
optimizer = AdamW(params       = network.parameters(),
                  lr           = 0.004,
                  weight_decay = 0.05,
                  betas        = (0.9, 0.999))

# %% [markdown]
# # 4. LR SCHEUDLER

# %%
epochs = 5
niter_epoch = int(ceil(len(train_dataloader.dataset) / train_dataloader.batch_size))

lr_cosine_scheduler = CosineDecaySchedulerLR(optimizer = optimizer, 
                                             base_lr = 0.004, 
                                             epochs = epochs, 
                                             niter_epoch = niter_epoch,  
                                             warmup_epochs = 2, 
                                             constant_epochs = 2)

# %% [markdown]
# ## 5. Get GPU

# %%
if gpu_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %%
print('Using ' + str(device) + ' device!')

# %% [markdown]
# ## 6. TRAIN NETWORK

# %%
import time
start = time.time()

# %%
train_network(epochs           = epochs, 
              network          = network, 
              optimizer        = optimizer, 
              loss_fn          = loss_fn, 
              train_dataloader = train_dataloader, 
              valid_dataloader = valid_dataloader,
              device           = device,
              #early_stopping  = 3,
              lr_scheduler     = lr_cosine_scheduler)

# %%
end = time.time()
print((end - start)/60)


