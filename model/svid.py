import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import numpy as np

class SVID(nn.Module):
    def __init__(self, num_classes=25, grid_size=4):
        super(SVID, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size # n * n regions
        self.sigmas = None
        self.blur_generator = RegionAwareBlurring(grid_size=self.grid_size)

        # Load backbone
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.backbone = vgg.features
        self.feat_dim = 512

        # FC6 + FC7
        self.region_fc = nn.Sequential(
            nn.Linear(self.feat_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
             nn.Dropout(0.5)
        )

        # Classification and Detection branches
        self.cls_branch = nn.Linear(4096, num_classes, bias=False)
        self.det_branch = nn.Linear(4096, num_classes)


    def extract_regions(self, feat_map):
        """
        Split feature map [B, C, H, W] into grid regions and pool.
        Output: [B, R, C] where R = grid_size^2
        """
        B, C, H, W = feat_map.shape
        gh, gw = self.grid_size, self.grid_size

        assert H % gh == 0 and W % gw == 0, "Feature map not divisible by grid size"

        h_stride, w_stride = H // gh, W // gw

        regions = []
        for i in range(gh):
            for j in range(gw):
                region = feat_map[:, :, i*h_stride:(i+1)*h_stride, j*w_stride:(j+1)*w_stride]
                pooled = F.adaptive_avg_pool2d(region, 1).squeeze(-1).squeeze(-1)  # [B, C]
                regions.append(pooled)
        region_feats = torch.stack(regions, dim=1)  # [B, R, C]
        return region_feats

    def set_sigmaPool(self, sigmas):
        self.sigmas = sigmas



    def forward(self, x):
        feats = self.backbone(x)
        region_feats = self.extract_regions(feats)
        B, R, D = region_feats.shape

        # apply FC6/FC7 layers
        region_feats = self.region_fc(region_feats)

        # Classification and detection streams
        X_c = self.cls_branch(region_feats)   # [B, R, C]
        X_d = self.det_branch(region_feats)   # [B, R, C]

        # print("X_c: ", X_c)
        # print("X_d: ", X_d)

        # Normalize
        X_c = F.softmax(X_c, dim=2)      
        X_d = F.softmax(X_d, dim=1)

        # print("X_c: ", X_c)
        # print("X_d: ", X_d)

        # Combined Region Score
        X_r = X_c * X_d               # Region Level Score, [B, R, C]

        # print("X_r: ", X_r)


        i_X_r = X_r.sum(dim=1)        # Image Level Score, [B, C]
        # print("i_X_r: ", i_X_r)
        region_preds = torch.argmax(X_r, dim=2)  # [B, R]
        # print("region_preds: ", region_preds)

        # Create Filter
        blur_filter_fn = self.blur_generator._get_filter(region_preds, self.grid_size, self.sigmas)

        return i_X_r, blur_filter_fn


class RegionAwareBlurring:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def gaussian_kernel2d(self, sigma, ksize, device):
        x = (torch.arange(ksize).float() - ksize // 2).to(device)
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel1d = gauss / gauss.sum()
        kernel2d = torch.outer(kernel1d, kernel1d)
        return kernel2d

    def _get_filter(self, region_preds, grid_size, sigmas):
        def apply_blur(sr_tensor):
            B, C, H, W = sr_tensor.shape
            rh, rw = H // grid_size, W // grid_size
            device = sr_tensor.device

            blurred_tensor = torch.zeros_like(sr_tensor)

            for b in range(B):
                for r in range(grid_size * grid_size):
                    i = r // grid_size
                    j = r % grid_size
                    y1, y2 = i * rh, (i + 1) * rh
                    x1, x2 = j * rw, (j + 1) * rw

                    region = sr_tensor[b:b+1, :, y1:y2, x1:x2]  # shape: (1, C, rh, rw)

                    sigma_idx = region_preds[b, r]  # tensor scalar
                    sigma_val = torch.tensor(sigmas, device=device)[sigma_idx]  # still tensor


                    ksize = int(2 * np.ceil(2 * sigma_val.item()) + 1)  # kernel size as Python int
                    kernel = self.gaussian_kernel2d(sigma_val, ksize, device).to(device)
                    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)

                    pad = ksize // 2
                    region_blurred = F.conv2d(F.pad(region, (pad, pad, pad, pad), mode='reflect'),
                                            kernel, groups=C)

                    blurred_tensor[b:b+1, :, y1:y2, x1:x2] = region_blurred

            return blurred_tensor

        return apply_blur


if __name__ == "__main__":

    x = torch.randn(1, 3, 256, 256)
    sigma_pool = [round(s, 2) for s in np.arange(0.8, 3.2 + 1e-6, 0.1)] 
    print("input batch:", x.shape)
    sd = SVID()
    sd.set_sigma_pool = sigma_pool
    y, z = sd(x)

    print(y)
    print(z)
    