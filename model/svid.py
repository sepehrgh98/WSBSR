import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class SVID(nn.Module):
    def __init__(self, num_classes=5, grid_size=4):
        super(SVID, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size # n * n regions

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


    def forward(self, x):
        feats = self.backbone(x)
        region_feats = self.extract_regions(feats)
        B, R, D = region_feats.shape

        # apply FC6/FC7 layers
        region_feats = self.region_fc(region_feats)

        # Classification and detection streams
        X_c = self.cls_branch(region_feats)   # [B, R, C]
        X_d = self.det_branch(region_feats)   # [B, R, C]

        # Normalize
        X_c = F.softmax(X_c, dim=2)      
        X_d = F.softmax(X_d, dim=1)

        # Combined Region Score
        X_r = X_c * X_d               # Region Level Score, [B, R, C]
        i_X_r = X_r.sum(dim=1)        # Image Level Score, [B, C]

        return X_r, i_X_r




if __name__ == "__main__":

    x = torch.randn(4, 3, 256, 256)
    print("input batch:", x.shape)
    sd = SVID()

    y, z = sd(x)

    print("output batch:",y.shape, z.shape)
    print(z)
    