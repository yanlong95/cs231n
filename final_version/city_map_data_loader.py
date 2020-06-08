""" Modified data_loader return image name instead of labels, called by city_map_builder.py"""

import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# loader for evaluation, no horizontal flip
transformer = transforms.Compose([
    transforms.Resize(256),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor


class BuildingMap(Dataset):
    def __init__(self, data_dir, transform):
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, os.path.basename(os.path.normpath(self.filenames[idx]))


def load(data_dir, params):
    dl = DataLoader(BuildingMap(data_dir, transformer), batch_size=params.batch_size, shuffle=False,
                    num_workers=params.num_workers, pin_memory=params.cuda)
    return dl
