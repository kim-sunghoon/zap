import os
import torch
import numpy as np
from PIL import Image

def gen_mask(type, dim):
    mask = torch.ones((dim[2], dim[3])).cuda()
    #  print(mask)
    #  print(mask.type())

    reverse = False
    if type < 5:
        type = 10 - type
        reverse = True

    if type == 5:
        mask[0::2, 1::2] = 0
        mask[1::2, 0::2] = 0
    elif type == 6:
        mask[0::5, 2::5] = 0
        mask[0::5, 4::5] = 0
        mask[1::5, 1::5] = 0
        mask[1::5, 3::5] = 0
        mask[2::5, 0::5] = 0
        mask[2::5, 2::5] = 0
        mask[3::5, 1::5] = 0
        mask[3::5, 4::5] = 0
        mask[4::5, 0::5] = 0
        mask[4::5, 3::5] = 0
    elif type == 7:
        mask[0::3, 2::3] = 0
        mask[1::3, 1::3] = 0
        mask[2::3, 0::3] = 0
    elif type == 8:
        mask[1::2, 1::2] = 0
    elif type == 9:
        mask[1::3, 1::3] = 0
        mask[2::3, 0::3] = 0
    elif type == 10:
        mask[1::3, 1::3] = 0
    else:
        raise NotImplementedError

    if reverse is True:
        mask = torch.abs(mask - 1)

    return mask

if __name__ == "__main__":
    mask_types = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    dim = [0, 1, 640, 640]

    for mask_type in mask_types:
        mask = gen_mask(mask_type, dim)
        print("=="*20)
        print(mask_type)
        print(mask)
        mask = mask.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(mask*255)
        img.save(os.path.join('summary_csv', 'mask-{}.png'.format(mask_type)))
        #  img.show()

