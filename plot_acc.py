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

def plot_first_fig():
    our_mem_bs_128 = [1836, 3753, 8286]
    deit_mem_bs_128 = [4149, 8372, 17355]
    deit_acc = [72.0, 79.8, 81.8]
    our_acc = [72.1, 79.9, 81.6]
    plt.figure(dpi=200)

    deit_color = (68./255, 114./255, 196./255, 1.)
    # our_color = (237./255, 125./255, 49./255, 1.)
    our_color = (217. / 255, 93. / 255, 93. / 255, 1.)

    # deit_color = (86./255, 134./255, 135./255, 1.)
    # our_color = (189./255, 82./255, 57./255, 1.)


    plt.plot(our_mem_bs_128, our_acc, marker='*', markersize=10, label='Ours', color=our_color)
    plt.plot(deit_mem_bs_128, deit_acc, marker='o', markersize=8, label='DeiT', color=deit_color)

    # deit_labels = ['DeiT-Ti', 'DeiT-S', 'DeiT-B']
    # ours_labels = []
    # for i, txt in enumerate(n):
    #     plt.annotate(txt, (z[i], y[i]))

    plt.xlabel('Training Memory (MB)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.xlim((0, 25000))
    plt.ylim((70, 84))
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig('/data1/cvpr2022/deit-mem-save/plots/deit_plot.pdf')
    # plt.show()
    print('..')




def plot_log_v2(key):
    plt.clf()    # set_figure_pixel_size(1000, 700)
    plt.figure(figsize=(5,4), dpi=200)
    # plt.figure(figsize=(5, 4))
    base_dir = '/data1/cvpr2022/acc_plot/compare_cuda_small'
    exps = os.listdir(base_dir)
    exps = sorted(exps)
    for exp in exps:
        label = exp.split('.')[0]
        if exp.endswith('pdf') or exp.endswith('png'):
            continue
        # if os.path.isfile(os.path.join(base_dir, exp)):
        #     continue
        with open(os.path.join(base_dir, exp)) as f:
            logs = []
            epochs = []

            for line in f.readlines():
                epoch_stat = json.loads(line.strip())
                logs.append(epoch_stat[key])
                epochs.append(epoch_stat['epoch'])
            plt.plot(epochs, logs, label=label, alpha=0.8)
    plt.legend()
    plt.xlabel('Epoch')
    ylabel = keys_dict[key]
    plt.ylabel(ylabel)
    plt.tight_layout()
    # plt.grid()
    # plt.show()
    plt.savefig(base_dir + '/%s.png'%(ylabel))

def plot_log(key):
    plt.clf()    # set_figure_pixel_size(1000, 700)
    plt.figure(figsize=(5,4))
    base_dir = '/data1/cvpr2022/acc_plot/deit-compare'
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
    # plot_first_fig()
    keys = ['test_acc1', 'test_acc5', 'train_loss', 'test_loss']
    for key in keys:
        plot_log_v2(key)
