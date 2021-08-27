import torch

if __name__ == '__main__':
    model = torch.load('outputs/torch-quant/last_checkpoint.pth', map_location='cpu')

    for k, v in model['model'].items():
        # print(k, v)
        if 'clip_val' in k:
            print(k, v)
        if 'shift' in k:
            print(k, v)