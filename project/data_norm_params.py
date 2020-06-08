""" Finding Datset Normalization Params """

import data_loader_v5 as data_loader
import utils
import torch

path = 'dataset/full_region_data'
json_path = 'model/params.json'
params = utils.Params(json_path)
params.cuda = torch.cuda.is_available()
params.batch_size = 64

dataloaders = data_loader.fetch_dataloader(['train', 'test'], path, params)

train_dl = dataloaders['train']
val_dl = dataloaders['val']
test_dl = dataloaders['test']

image_size = 512
sum = torch.zeros(params.batch_size, 3, image_size, image_size)

for i, (image1, _, _) in enumerate(train_dl):
    if image1.shape == sum.shape:
        sum += image1
    else:
        i -= 1

avg = [torch.mean(sum[:,0,:,:])/(i+1), torch.mean(sum[:,1,:,:])/(i+1), torch.mean(sum[:,2,:,:])/(i+1)]
std = [torch.std(sum[:,0,:,:])/(i+1), torch.std(sum[:,1,:,:])/(i+1), torch.std(sum[:,2,:,:])/(i+1)]
print('Train Avg:', avg)
print('Train Std:', std)
