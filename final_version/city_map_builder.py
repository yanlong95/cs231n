""" City-scale street view classification evaluation """

import argparse
import logging
import os
from tqdm import tqdm

import torch
import pandas as pd
import numpy as np
from torch.autograd import Variable

import utils
import city_map_data_loader as data_loader
import model_v6 as net
import visualize


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default= 'dataset/boston' , help="Directory containing the dataset")
parser.add_argument('--location_dir', default= 'dataset' , help="Directory containing the image geolocation")
parser.add_argument('--model_dir', default='model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()
    params.batch_size = 32

    logging.info("Creating the dataset...")
    dl = data_loader.load(args.data_dir, params)
    logging.info("- done.")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # change CNN architecture
    num_class = 7
    num_region = 3
    model = net.resnet50(params, num_class, num_region).to(device)

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    for index, (images_batch, labels_batch) in enumerate(dl):

        # move to GPU if available
        if params.cuda:
            images_batch = images_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch
        images_batch = Variable(images_batch)

        # compute model output
        output_type_batch, _ = model(images_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_type_batch = output_type_batch.data.cpu().numpy().argmax(1) # list of predicted class number
        
        if index == 0:
            labels_batch_summ = labels_batch
            output_type_batch_summ = output_type_batch
        else:
            labels_batch_summ += labels_batch
            output_type_batch_summ = np.append(output_type_batch_summ, output_type_batch)
    
    # read geolocation txt file
    
    df_lat_lon = pd.read_csv(os.path.join(args.location_dir, 'boston_latlon.txt'), sep = '[/, :, ,]', header=None, 
                          names=['0', '1', 'imgname', 'lat', 'lon'], dtype='str', engine='python')
    df_lat_lon = df_lat_lon[['imgname', 'lat', 'lon']] # data format: image name, lat, lon all in string
    df_lat_lon['pred_type'] = 0 # make a new tag column
    
    for index, imgname in enumerate(labels_batch_summ):
        img_row_index = df_lat_lon[df_lat_lon['imgname'] == imgname].index.values.astype(int)[0]
        df_lat_lon.loc[img_row_index, 'pred_type'] = output_type_batch_summ[index]
        
    visualize.city_map_pred_plot(df_lat_lon)
    print("...saving final result...")
    df_lat_lon.to_csv('boston_city_map_building_category_pred' + '.csv')
    print("saving done!")