import os
import torch
from PIL import Image
from torch.utils.data import Dataset

import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torch import from_numpy
from skimage.color import rgb2lab


class ImageDataset(Dataset):
    def __init__(self, color_dir, gray_dir = None):
        self.names = os.listdir(color_dir)
        self.color_dir = color_dir
        self.gray_dir = gray_dir

    def __len__(self):

        return len(self.names)

    def __getitem__(self, index):
        
        color_path = os.path.join(self.color_dir, self.names[index])
        image = from_numpy(rgb2lab(read_image(color_path).permute(1, 2, 0))).permute(2, 0, 1)

        # The color image consists of the 'a' and 'b' parts of the LAB format.
        color_image = image[1:, :, :]
        # The gray image consists of the `L` part of the LAB format.
        gray_image = image[0, :, :].unsqueeze(0)

        return gray_image.float(), color_image.float()