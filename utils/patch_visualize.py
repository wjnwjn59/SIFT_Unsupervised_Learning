import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from PIL import Image

def visualize_pair(
    data_dir,
    pair_idx=0,
    save_plot_path='./patch_pair_plot.png'
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    img_1_path = os.path.join(
        data_dir, f'pair_{pair_idx}_img_0.jpg'
    )
    img_2_path = os.path.join(
        data_dir, f'pair_{pair_idx}_img_1.jpg'
    )

    image_1 = Image.open(img_1_path)
    axes[0].imshow(image_1)
    axes[0].axis('off')

    image_2 = Image.open(img_2_path)
    axes[1].imshow(image_2)
    axes[1].axis('off')

    plt.tight_layout()

    if save_plot_path is not None:
        plt.savefig(save_plot_path)
        print(f"Plot saved as {save_plot_path}")
    else:
        plt.show()

def visualize_patches(
    data_dir, 
    n_pairs_to_show=4,
    save_plot_path='./patches_plot.png'
):
    fig, axes = plt.subplots(n_pairs_to_show, 2, figsize=(10, 10))
    images = [
        os.path.join(data_dir, filename) \
            for filename in os.listdir(data_dir)
    ]
    images.sort()
    images = images[:(n_pairs_to_show * 2)]
    print(images)
    n_images = len(images)

    for i in range(n_pairs_to_show):
        for j in range(2):
            idx = i * 2 + j
            if idx < n_images:
                image_path = images[idx]
                image = Image.open(image_path)
                axes[i][j].imshow(image)
                axes[i][j].axis('off')

    plt.tight_layout()

    if save_plot_path is not None:
        plt.savefig(save_plot_path)
        print(f"Plot saved as {save_plot_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/patches_dataset'
    )
    parser.add_argument(
        '--pair_idx',
        type=int,
        default=0
    )
    args = parser.parse_args()

    visualize_pair(
        args.data_dir,
        args.pair_idx
    )

if __name__ == '__main__':
    main()