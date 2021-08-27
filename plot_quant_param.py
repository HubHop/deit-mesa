import json
import os
import matplotlib.pyplot as plt
import numpy
import numpy as np
from collections import defaultdict

def plot_clip_val():
    clip_val_path = 'outputs/previous-best/clip_val.json'
    save_dir = '/data1/cvpr2022/plots/previous-best/clip_vals'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(clip_val_path, 'r') as f:
        clip_vals = json.load(f)
    f.close()

    for k, v in clip_vals.items():
        plt.clf()
        plt.figure(figsize=(5, 4), dpi=150)
        values = numpy.array(v)
        epochs, num_groups = values.shape
        epochs = list(range(epochs))
        for i in range(num_groups):
            plt.plot(epochs, values[:, i], label=k + '-' + str(i))
        plt.legend()
        plt.xlabel('Epoch')
        # plt.tight_layout()
        plt.ylabel('clip_val')
        plt.savefig(save_dir + '/%s.png' % (k))

def plot_shifts():
    clip_val_path = 'outputs/previous-best/shift.json'
    save_dir = '/data1/cvpr2022/plots/previous-best/shifts'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(clip_val_path, 'r') as f:
        clip_vals = json.load(f)
    f.close()

    for k, v in clip_vals.items():
        plt.clf()
        plt.figure(figsize=(6, 4), dpi=150)
        values = numpy.array(v)
        epochs, num_groups = values.shape
        epochs = list(range(epochs))
        for i in range(num_groups):
            plt.plot(epochs, values[:, i], label=k + '-' + str(i))
        plt.legend()
        plt.xlabel('Epoch')
        # plt.tight_layout()
        plt.ylabel('shift')
        # plt.show()
        # print('..')
        plt.savefig(save_dir + '/%s.png' % (k))
#

if __name__ == '__main__':
    plot_clip_val()