import torch

if __name__ == '__main__':
    model = torch.load('/data1/cvpr2022/final_model/work_dirs/retinanet_pvt_t_fpn_1x_coco/epoch_6.pth', map_location='cpu')

    for k, v in model['state_dict'].items():
        # if 'iteration' in k:
        #     print(k, v.item())
        # print(k)
        if 'clip_val' in k:
            print(k, v)
            # if torch.any(v > 100):
            #     print(k, v)
        if 'shift' in k:
            print(k, v)
        # print(k, v)
        # if 'clip_val' in k:
        #     if torch.any(v > 100):
        #         print(k, v)
        # if 'shift' in k:
        #     if torch.any(v > 100):
        #         print(k, v)