import torch

if __name__ == '__main__':
    model = torch.load('outputs/channel_wise_cuda_test_ema/last_checkpoint.pth', map_location='cpu')

    for k, v in model['model'].items():
        # if 'clip_val' in k:
        #     print(k, v)
        # if 'shift' in k:
        #     print(k, v)
        # print(k, v)
        if 'clip_val' in k:
            if torch.any(v > 100):
                print(k, v)
        if 'shift' in k:
            if torch.any(v > 100):
                print(k, v)