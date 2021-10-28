import json
import os
import matplotlib.pyplot as plt
import numpy
import numpy as np
from collections import defaultdict
# plt.rcParams['axes.spines.right'] = False
# plt.rcParams['axes.spines.top'] = False
# plt.rcParams.update({'font.size': 14})
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.family']
# plt.style.use('science')

import seaborn as sns
# sns.set_theme(style="darkgrid")

def sns_test():
    sns.set_theme(style="darkgrid")

    # Load an example dataset with long-form data
    fmri = sns.load_dataset("fmri")

    # Plot the responses for different events and regions
    sns.lineplot(x="timepoint", y="signal",
                 hue="region", style="event",
                 data=fmri)
    plt.show()
    print('..')

def plot_clip_val():
    clip_val_path = '/data1/cvpr2022/final_model/ema_0.9/clip_val.json'
    save_dir = '/data1/cvpr2022/final_model/ema_0.9/clip_vals'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

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
            name = k[7:].replace('clip_val', 'alpha')
            data = np.array([epochs, values[:, i]])
            sns.lineplot(data)
            # plt.plot(epochs, values[:, i], label= name + '.' + str(i))
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('alpha value')
        # plt.grid(linestyle='-', linewidth=0.3)
        # plt.savefig(save_dir + '/%s.png' % (k))
        plt.tight_layout()
        plt.show()
        print('..')

def plot_shifts():
    clip_val_path = '/data1/cvpr2022/final_model/ema_0.9/shift.json'
    save_dir = '/data1/cvpr2022/final_model/ema_0.9/shifts'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

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
    sns_test()
    # plot_shifts()