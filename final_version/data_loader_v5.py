""" Load dataset as tensor format into model"""

import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

dict = {'ca': 0, 'ne': 1, 'se': 2}

# train transformer
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5049, 0.5313, 0.5386), (0.0616, 0.0949, 0.1495))])

# evl and test transformer
eval_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5008, 0.5329, 0.5609), (0.0516, 0.0813, 0.1364))])

class BuildingDatasetWithRegion(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.images = []
        self.labels_type = []
        self.labels_region = []
        self.root = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.root]

        for region in self.filenames:
            type = [os.path.join(region, f) for f in os.listdir(region)]
            for i, file in enumerate(type):
                for image in os.listdir(file):
                    src = os.path.join(file, image)
                    self.images.append(src)
                    self.labels_type.append(i)
                    self.labels_region.append(os.path.basename(os.path.normpath(region)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.transform(image)
        return image, self.labels_type[idx], dict[self.labels_region[idx]]


# load a train, val, text in mini-batch size
def fetch_dataloader(types, data_dir, params):
    dataloaders = {}

    for split in ['train', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(BuildingDatasetWithRegion(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers, pin_memory=params.cuda)
                val_dl_length = int(len(dl) * 0.1)
                train_dl, val_dl = random_split(dl, [int(len(dl) - val_dl_length), val_dl_length])
                # automatically split val data from train data as 10% of total train data

                dataloaders[split] = train_dl.dataset
                dataloaders['val'] = val_dl.dataset
            else:
                dl = DataLoader(BuildingDatasetWithRegion(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                        num_workers=params.num_workers, pin_memory=params.cuda)
                dataloaders[split] = dl

    return dataloaders
