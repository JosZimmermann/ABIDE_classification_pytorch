
import torch
import torchvision
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary


import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn.model_selection import train_test_split
from skimage import io
from sklearn import preprocessing
from scipy import stats
from sklearn.preprocessing import RobustScaler

from tqdm.auto import tqdm

import random
import os

import copy





# SETUP FUNCTIONS
#--------------------------

def set_seed(seed=None, seed_torch=True):
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')

# In case that `DataLoader` is used
def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("WARNING: For this notebook to perform best, "
        "if possible, in the menu under `Runtime` -> "
        "`Change runtime type.`  select `GPU` ")
  else:
    print("GPU is enabled in this notebook.")

  return device

#--------------------------------------------------



# SETUP CUDA AND SEED
#--------------------------------------------------

SEED = 222
set_seed(seed=SEED)
DEVICE = set_device()

#--------------------------------------------------




# DEFINE L1 AND L2 LOSSS FUNCTION
#--------------------------------------------------

def get_L1_L2_norm(model):
    L1_norm = sum(p.abs().sum() for p in model.parameters())
    L2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

    return L1_norm, L2_norm

#--------------------------------------------------



# TRAING FUNCTION INCLUDING EARLY STOPPING
# - L1 and L2 penalty can be introduced by setting lambda_L1 or lambda_L2 > 0
#--------------------------------------------------

def train(model, num_epochs, train_batch, val_batch, device, earlystoping_patience=20, lambda_L1=0.0, lambda_L2=0.0):

  
  metrics = {'train_loss':[],
             'train_acc':[],
             'val_loss':[],
             'val_acc':[]}
             
  best_acc = 0.0
  wait = 0
  best_epoch = 0
  
  for epoch in tqdm(range(num_epochs)):
  
    acc_tmp = []
    loss_tmp = []
    
    model.train()
    for batch_idx, batch in enumerate(train_batch):
      data, target = batch["features"].to(DEVICE), batch["label"].to(DEVICE)
      
      # reset the parameter gradients
      optimizer.zero_grad()
      
      # forward pass + backward pass + optimize
      prediction = model(data)
      loss = criterion(prediction, target)

      L1_norm, L2_norm = get_L1_L2_norm(model)
      loss = loss + lambda_L1 * L1_norm + lambda_L2 * L2_norm      

      loss.backward()
      optimizer.step()

      acc_tmp.append(torch.mean(1.0 * (prediction.argmax(dim=1) == target)).cpu().item())
      loss_tmp.append(loss.cpu().item())

    metrics['train_loss'].append(np.mean(loss_tmp))
    metrics['train_acc'].append(np.mean(acc_tmp))
    

    val_loss_tmp = [] 
    val_acc_tmp = []
    
    model.eval()
    with torch.no_grad():
        for batch_id, batch_val in enumerate(val_batch):
          data, target = batch_val["features"].to(DEVICE), batch_val["label"].to(DEVICE)
          
          # forward pass only
          prediction = model(data)
          
          L1_norm, L2_norm = get_L1_L2_norm(model)
          total_loss_tmp = criterion(prediction, target).cpu().item() + lambda_L1 * L1_norm + lambda_L2 * L2_norm
          
          val_loss_tmp.append(total_loss_tmp.cpu())
          val_acc_tmp.append(torch.mean(1.0 * (prediction.argmax(dim=1) == target)).cpu().item())
        
        val_acc = np.mean(val_acc_tmp)
        val_loss = np.mean(val_loss_tmp)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)        
        
        
    # earlystopping
    if (val_acc > best_acc):
      best_acc = val_acc
      best_epoch = epoch
      #best_model = copy.deepcopy(model)
      wait = 0
    else:
      wait += 1

    if (wait > earlystoping_patience):
      print(f'Early stopped on epoch: {epoch}')
      break        
            

  return metrics, best_epoch #, best_model

#--------------------------------------------------



# TEST FUNCTION
#--------------------------------------------------

# define generic test function
def test(model, test_batch, device):

  model.eval()
  
  acc_test = []       
     
  for batch_idx, batch in enumerate(test_batch):
    data, target = batch["features"].to(DEVICE), batch["label"].to(DEVICE)
                
    prediction = model(data)

    acc = torch.mean(1.0 * (prediction.argmax(dim=1) == target))
    acc_test.append(acc.cpu().item())
    
  return np.mean(acc_test)
#--------------------------------------------------




# MLP CLASS
#--------------------------------------------------

class MLP(nn.Module):
  """
  This class implements MLPs in Pytorch of an arbitrary number of hidden
  layers of potentially different sizes.
  """

  def __init__(self, in_dim, out_dim, hidden_dims=[], use_bias=True, add_dropout=False, dropout_p=0.5):
    """
    Constructs a MultiLayerPerceptron

    Args:
      in_dim: Integer
        dimensionality of input data
      out_dim: Integer
        number of classes
      hidden_dims: List
        containing the dimensions of the hidden layers,
        empty list corresponds to a linear model (in_dim, out_dim)

    Returns:
      Nothing
    """

    super(MLP, self).__init__()

    self.in_dim = in_dim
    self.out_dim = out_dim

    # If we have no hidden layer, just initialize a linear model (e.g. in logistic regression)
    if len(hidden_dims) == 0:
      layers = [nn.Linear(in_dim, out_dim, bias=use_bias)]
    else:
      # 'Actual' MLP with dimensions in_dim - num_hidden_layers*[hidden_dim] - out_dim
      layers = [nn.Linear(in_dim, hidden_dims[0], bias=use_bias), nn.ReLU()]
      if add_dropout: layers += [nn.Dropout(dropout_p)]

      # Loop until before the last layer
      for i, hidden_dim in enumerate(hidden_dims[:-1]):
        layers += [nn.Linear(hidden_dim, hidden_dims[i + 1], bias=use_bias),
                   nn.ReLU()]
        if add_dropout: layers += [nn.Dropout(dropout_p)]

      # Add final layer to the number of classes
      layers += [nn.Linear(hidden_dims[-1], out_dim, bias=use_bias)]

    self.main = nn.Sequential(*layers)

  def forward(self, x):
    # x = x.view(-1, self.in_dim) # if you need to flatten input
    logits = self.main(x)
    #output = F.log_softmax(hidden_output, dim=1) # if you add this, you should use nll_loss instead of cross_entropy !!!
    
    return logits
    
#--------------------------------------------------




# LOAD DATASET
#--------------------------------------------------

if not os.path.isfile('connmats_allsubj.npy'):
    dataset_nilearn = datasets_nilearn.fetch_abide_pcp(derivatives="rois_cc200", legacy_format=True)
    data_raw = dataset_nilearn.rois_cc200
    data_list_np = []
    for i in range(len(data_raw)):
        data_list_np.append(np.array(data_raw[i]))

    cor_maker = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
    connmats_allsubj = cor_maker.fit_transform(data_list_np)
    label_allsubj = np.array(dataset_nilearn.phenotypic.DX_GROUP - 1)

    np.save('connmats_allsubj.npy', connmats_allsubj)
    np.save('labels_allsubj.npy', label_allsubj)

#--------------------------------------------------



# DATASET CLASS
#--------------------------------------------------

class ABIDE_dataset(Dataset):

    def __init__(self, data_file, label_file, transform=None):

        self.connmats_allsubj = np.load(data_file)
        self.label_allsubj = np.load(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.connmats_allsubj)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        connmat_flat = self.connmats_allsubj[idx,:]
        label = np.array(self.label_allsubj[idx])

        sample = {'features': connmat_flat, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


      
# custom normalize function    
class Normalise(object):

    def __call__(self, sample):
        features, label = sample['features'], sample['label']

        features_scaled = stats.zscore(features) # zscoring
      
        return {'features': features_scaled,
                'label': label}



# custom toTensor function     
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, label = sample['features'], sample['label']

        return {'features': torch.from_numpy(features).to(torch.float32),
                'label': torch.from_numpy(label).to(torch.long)}
                

#--------------------------------------------------





# DATALOADER
#--------------------------------------------------

transform = transforms.Compose([Normalise(), ToTensor()])

ABIDE_data = ABIDE_dataset('connmats_allsubj.npy', 'labels_allsubj.npy', transform)



train_dataset, val_dataset, test_dataset =  random_split(ABIDE_data, [0.7, 0.15, 0.15])
#train_dataset, val_dataset =  random_split(ABIDE_data, [0.8, 0.2])


batch_size = math.floor(len(train_dataset)/2)
num_workers = 4              
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)


val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset),
                        shuffle=True, num_workers=num_workers)

         
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset),
                        shuffle=True, num_workers=num_workers)                        



# print dimensions of data
batch = next(iter(train_dataloader))
features_size = batch['features'].size()

print("*********************************")
print("N whole dataset: %d" % len(ABIDE_data))
print("N training sample: %d" % len(train_dataset))
print("N validation sample: %d" % len(val_dataset))
print("N test sample: %d" % len(test_dataset))
print("The input size is: [%d,%d] (first dim is batchsize)" % features_size)
print("*********************************")

#--------------------------------------------------




        

# HYPERPARAMETERS TO TUNE
#--------------------------------------------------

n_epochs = 200
lr = 0.001

hidden_dims = [1024, 512, 256, 128]
add_dropout = True
dropout_p = 0.5

earlystopping_patience = 50 # after how many epochs training stops when accuracy is not higher than previous (best accuracy) epoch

lambda_L1 = 0.00005 # lambda = 0 -> no regularization
lambda_L2 = 0.0005

#--------------------------------------------------



# TRAINING THE NEURAL NET
#--------------------------------------------------

# define model
model = MLP(in_dim = features_size[1], out_dim = 2, hidden_dims = hidden_dims, add_dropout = add_dropout, dropout_p = dropout_p).to(DEVICE)
print(model)

criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

train_metrics,  best_epoch = train(model,
                                   n_epochs,
                                   train_dataloader, val_dataloader,
                                   DEVICE,
                                   earlystopping_patience,
                                  lambda_L1 = lambda_L1, lambda_L2 = lambda_L2)
 
#--------------------------------------------------




# PLOT TRAINING PROGRESS
#-------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(range(len(train_metrics['train_loss'])), train_metrics['train_loss'],
           alpha=0.8, label='Training')
ax[0].plot(range(len(train_metrics['val_loss'])), train_metrics['val_loss'], label='Validation')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(range(len(train_metrics['train_acc'])), train_metrics['train_acc'],
           alpha=0.8, label='Training')
ax[1].plot(range(len(train_metrics['val_acc'])), train_metrics['val_acc'], label='Validation')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
plt.tight_layout()

plt.savefig('ABIDE_progress_training_batchsize' + str(batch_size) + "_lr" + str(lr) + "_epochs" + str(n_epochs) + '.png')
#plt.show()
#-------------------------------



test_acc = test(model, val_dataloader, DEVICE)
print("**** VAL ACCURACY ****")
print(test_acc)


test_acc = test(model, test_dataloader, DEVICE)
print("**** TEST ACCURACY ****")
print(test_acc)




