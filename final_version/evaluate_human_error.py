""" Randomly select pictures and labels from dataset for human error benchmark """

import argparse
import os, random
from PIL import Image
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default= 'dataset/full_region_data' , help="Directory containing the dataset")
parser.add_argument('--model_dir', default='model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'

def random_file(dir):
    file = os.path.join(dir, random.choice(os.listdir(dir)));
    if os.path.isdir(file):
        return random_file(file)
    else:
        return file

args = parser.parse_args()

file = random_file('C:/Users/jack_minimonster/Documents/231n_dataset/bin/' + args.data_dir)
if '.jpg' in file:
    img = Image.open(file)
    plt.imshow(img)
    print(file)