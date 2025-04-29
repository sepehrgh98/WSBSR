from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Union, List, Tuple
import numpy as np
import random
import cv2
import math
import os
import torch


from dataset.utils import random_mixed_kernels, center_crop_arr, random_crop_arr


class WSDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        output_size: int,
        crop_type: str,
        blur_kernel_size: int,
        kernel_list: list,
        kernel_prob: list,
        blur_sigma: list,
        downsample_range: list,
        valid_extensions: list = [".png", ".jpg", ".jpeg"],
        n_regions: int = 4 # will seperate image to n*n regions
    ) -> "WSDataset":
        super(WSDataset, self).__init__()

        self.dataset_dir = dataset_dir
        self.output_size = output_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"], f"Invalid crop_type: {self.crop_type}"

        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.valid_extensions = valid_extensions
        self.n_regions = n_regions

        self._run()


    def _run(self):
        print("[INFO] Initializing WSDataset...")
        # Load data
        self.image_list = self._load_image(self.dataset_dir)

        

    def _load_image(self, path) -> list:
        image_list = []
        
        for dataset_name in os.listdir(path):
            dataset_path = os.path.join(self.dataset_dir, dataset_name)

            if not os.path.isdir(dataset_path):
                continue

            is_empty = not any(os.scandir(dataset_path))

            if is_empty:
                raise ValueError(f"[ERROR] Dataset '{dataset_name}' is Empty!.")

            print(f"[INFO] Dataset '{dataset_name}' is loaded!.")


            for img in os.listdir(dataset_path):
                if os.path.splitext(img)[1].lower() in self.valid_extensions:
                    img_path = os.path.join(dataset_path, img)
                    
                    image_list.append(img_path)

        
        print(f"[INFO] Loaded {len(image_list)} image pairs successfully.")
        return image_list


    def get_LR(self, img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        
        n = self.n_regions
        h, w, c = img.shape
        ph, pw = h // n, w // n

        np.random.seed(42)
        torch.manual_seed(42)

        
        start_sigma, end_sigma = self.blur_sigma
        sigma_pool = [round(s, 2) for s in np.arange(start_sigma, end_sigma + 1e-6, 0.1)] 
        m = random.randint(1, 5) # number of image level labels
        sigma_values = random.sample(sigma_pool, m)

        downsample_scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])


        out_img = np.zeros_like(img)


        for i in range(n):
            for j in range(n):
                y0, y1 = i * ph, (i + 1) * ph
                x0, x1 = j * pw, (j + 1) * pw
                patch = img[y0:y1, x0:x1]

                kernel_type = random.choices(self.kernel_list, weights=self.kernel_prob)[0]
                sigma = random.choice(sigma_values)

                # blurring
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    self.blur_kernel_size,
                    self.blur_sigma,
                    self.blur_sigma,
                    [-math.pi, math.pi],
                    noise_range=None
                )
                blurred_patch = cv2.filter2D(patch, -1, kernel)

                # downsample
                downsampled_patch = cv2.resize(
                blurred_patch, (int(pw // downsample_scale), int(ph // downsample_scale)), interpolation=cv2.INTER_LINEAR
                )

                # resize to original size
                lq_patch = cv2.resize(downsampled_patch, (pw, ph), interpolation=cv2.INTER_LINEAR)


                out_img[y0:y1, x0:x1] = lq_patch


        return out_img, sigma_values, downsample_scale




    def _clip(self, img: Image) -> np.ndarray:
        if self.crop_type != "none":
            if img.height == self.output_size and img.width == self.output_size:
                img = np.array(img)
            else:
                if self.crop_type == "center":
                    img = center_crop_arr(img, self.output_size)
                elif self.crop_type == "random":
                    img = random_crop_arr(img, self.output_size, min_crop_frac=0.7)
        else:
            img = np.array(img)
        return img

    def __getitem__(self, index: int) ->  Dict[str, Union[np.ndarray, list , str]]:
        img_path = self.image_list[index]
        print(f"[INFO] Loading image : {img_path}")

        # Open the image
        img = Image.open(img_path).convert("RGB")

        # Generate LR
        img = self._clip(img)
        LR, labels, _ = self.get_LR(img)
        prompt = ""

        return LR, labels, prompt

    def __len__(self) -> int:
        return len(self.image_list)
