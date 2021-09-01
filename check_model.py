import torch

if __name__ == '__main__':
    model = torch.load('/data1/cvpr2022/models/tiny-cuda-ema-0.9-per-head-fp-forward/last_checkpoint.pth', map_location='cpu')

    for k, v in model['model'].items():
        # print(k, v)
        if 'clip_val' in k:
            if torch.any(v > 100):
                print(k, v)
        if 'shift' in k:
            if torch.any(v > 100):
                print(k, v)