from dataset.WSDataset import WSDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset_dir = "/speed-scratch/se_gham/WSBSR/WSBSR/data"

dataset = WSDataset(
    dataset_dir = dataset_dir,
    output_size = 512,
    crop_type = "center",
    blur_kernel_size = 41,
    kernel_list = ['iso'],
    kernel_prob  = [1],
    blur_sigma = [0.8, 3.2],
    downsample_range =  [2, 4],
    n_regions = 8 # will seperate image to n*n regions
)

batch = 1
dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)


for img, label, propmt in dataloader:
    print("Labels:", label)
    print(f"propmt :", {propmt})
    print(f"image shape :", {img.shape})

    
    plt.imshow(img[0])
    plt.axis('off')
    plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)

    print("Image saved to output_image.png")

    break

