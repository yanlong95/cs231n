""" Single Image Classification Confidence Tester and Saliency Map Creation """

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms
from torch.nn import functional as F
import model_v6 as net
import utils
import os
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='model',
                        help="Directory containing params.json")
    parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                         containing weights to load")
                         
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    
    # use GPU if available
    params.cuda = True     # use GPU is available
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # define model architecture
    model = net.resnet50(params, 7, 3).to(device)
     
    checkpoint = torch.load(os.path.join(
            args.model_dir, args.restore_file + '.pth.tar'), map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    
    # evl and test transformer
    eval_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5008, 0.5329, 0.5609), (0.0516, 0.0813, 0.1364))])
    
    # load the class label
    classes_type = ('apartment', 'church', 'house', 'industrial', 'office', 'retail', 'roof')
    classes_region = ('california', 'northeast', 'southeast')
    
    # load the test image
    img_name = 'dataset/outliner/roof_3.jpg'
    
    img = Image.open(img_name)
    input_img = V(eval_transformer(img).unsqueeze(0))
    input_img.requires_grad_()
    
    # forward pass
    logit_type, logit_region = model.forward(input_img.cuda())
    h_x_type = F.softmax(logit_type, 1).data.squeeze()
    h_x_region = F.softmax(logit_region, 1).data.squeeze()
    probs_type, idx_type = h_x_type.sort(0, True)
    probs_region, idx_region = h_x_region.sort(0, True)
    
    print('net prediction on {}'.format(img_name))
    # output the prediction
    for i in range(0, 3):
        print('{:.3f} -> {}'.format(probs_type[i], classes_type[idx_type[i]]))
    
    print()
    
    for i in range(0, 3):
        print('{:.3f} -> {}'.format(probs_region[i], classes_region[idx_region[i]]))
    
    # Generation Activation Map
    
    # comment out one of the following choices:
    logit_type[0, logit_type.argmax()].backward() # max confident class saliency
    # logit_type[0, classes_type.index('church')].backward() # chosen class saliency
    
    saliency, _ = torch.max(input_img.grad.data.abs(),dim=1)
    
    plt.subplot(1,2,2)
    plt.imshow(saliency[0], cmap=plt.cm.hot)
    plt.axis('off')
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
