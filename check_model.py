import torch

if __name__ == '__main__':
    model = torch.load('/data1/cvpr2022/Swin-Transformer/output/swin_tiny_per_channel/default/ckpt_epoch_latest.pth', map_location='cpu')

    for k, v in model['model'].items():
        if 'clip_val' in k:
            print(k, v.size())
        if 'shift' in k:
            print(k, v.size())
        # print(k, v)
        # if 'clip_val' in k:
        #     if torch.any(v > 100):
        #         print(k, v)
        # if 'shift' in k:
        #     if torch.any(v > 100):
        #         print(k, v)