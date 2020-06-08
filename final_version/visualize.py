""" Plotting and Visualize calling function """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

def plot_train_val_accuracy(train_acc_series, val_acc_series):
    
    train_acc_series = np.array(train_acc_series)
    val_acc_series = np.array(val_acc_series)
    epoch_series = range(1, len(train_acc_series)+1, 1)
    
    fig, ax = plt.subplots()
    ax.plot(epoch_series, train_acc_series, 'k-', epoch_series, val_acc_series, 'k--')
    ax.legend(('train accuracy', 'validation accuracy'), loc='best')
    ax.set(ylabel='accuracy',
           xlabel='epochs',
           ylim=[0,1.0])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.show()
    fig.savefig('train_val_accuracy.png', dpi=fig.dpi)
    
    
def plot_loss(loss_series):
    loss_series = np.array(loss_series)
    epoch_series = range(1, len(loss_series)+1, 1)
    
    fig, ax = plt.subplots()
    ax.plot(epoch_series, loss_series, 'k-')
    ax.set(ylabel='loss',
           xlabel='epochs')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.show()
    fig.savefig('train_loss.png', dpi=fig.dpi)
    
    
def plot_individual_label_f1score(val_metrics, type='val'):

    classes = ['apartments', 'church', 'house', 'industrial', 'office', 'retail', 'roof']
    f1score = []
    for key in val_metrics:
        if 'f1score' in key:
            f1score.append(val_metrics[key])
    
    fig, ax = plt.subplots()
    ax.bar(classes, f1score)
    ax.set(ylabel='F1 Score',
           xlabel='Building Categories',
           ylim=[0, 1])
    plt.xticks(rotation=90)
    # plt.show()
    fig.savefig('f1score_' + type + '.png', dpi=fig.dpi)
    
def plot_individual_label_f1score(val_metrics, type='val'):

    classes = ['apartments', 'church', 'house', 'industrial', 'office', 'retail', 'roof']
    f1score = []
    for key in val_metrics:
        if 'f1score' in key:
            f1score.append(val_metrics[key])
    
    fig, ax = plt.subplots()
    ax.bar(classes, f1score)
    ax.set(ylabel='F1 Score',
           xlabel='Building Categories',
           ylim=[0, 1])
    plt.xticks(rotation=90)
    # plt.show()
    fig.savefig('f1score_' + type + '.png', dpi=fig.dpi)
    
def city_map_pred_plot(df):
    """Evaluate the model on `num_steps` batches.

    Args:
        df: (pandas.DataFrame) dataframe contains building geolocation and predicted category
    """
    df['lat'] = df['lat'].astype(float)
    df['lon'] = df['lon'].astype(float)
    
    ax = df.plot.scatter(x='lon',y='lat',c='pred_type',colormap='plasma', s=0.1)
    
    ax.set(ylabel='lat',
           xlabel='lon')
    plt.tight_layout()
    plt.savefig('city_scale_class.png', dpi=200)
    