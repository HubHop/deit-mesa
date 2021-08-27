import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

keys_dict = {
    'test_acc1': 'Top-1 Acc. (%)',
    'test_acc5': 'Top-5 Acc. (%)',
    'train_loss': 'Train Loss',
    'test_loss': 'Test Loss'
}

def set_figure_pixel_size (width, height, ppi):
    plt.figure(figsize=(width/float(ppi), height/float(ppi)), dpi=ppi)


def plot_log(key):
    plt.clf()    # set_figure_pixel_size(1000, 700)
    plt.figure(figsize=(5,4))
    base_dir = '/data1/cvpr2022/compare'
    exps = os.listdir(base_dir)
    exps = sorted(exps)
    for exp in exps:
        if os.path.isfile(os.path.join(base_dir, exp)):
            continue
        with open(os.path.join(base_dir, exp, 'log.txt')) as f:
            logs = []
            epochs = []

            for line in f.readlines():
                epoch_stat = json.loads(line.strip())
                logs.append(epoch_stat[key])
                epochs.append(epoch_stat['epoch'])
                # if int(epoch_stat['epoch']) > 50:
                #     break
            plt.plot(epochs, logs, label=exp)
    plt.legend()
    plt.xlabel('Epoch')
    ylabel = keys_dict[key]
    plt.ylabel(ylabel)
    plt.tight_layout()
    # plt.grid()
    # plt.show()
    plt.savefig(base_dir + '/%s.pdf'%(ylabel))

if __name__ == '__main__':
    keys = ['test_acc1', 'test_acc5', 'train_loss', 'test_loss']
    for key in keys:
        plot_log(key)
