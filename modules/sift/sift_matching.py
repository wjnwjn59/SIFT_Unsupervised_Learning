from __future__ import division

import cv2
import os
import glob
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

def save_pair_patches(
    patch_1, 
    patch_2, 
    patch_idx, 
    patch_savedir
):
    patch_name_0 = f'pair_{patch_idx}_patch_0.jpg'
    patch_name_1 = f'pair_{patch_idx}_patch_1.jpg'
    patch_name_0_savepath = os.path.join(
        patch_savedir, 
        patch_name_0
    )
    patch_name_1_savepath = os.path.join(
        patch_savedir, 
        patch_name_1
    )

    cv2.imwrite(
        patch_name_0_savepath, 
        patch_1
    )
    cv2.imwrite(
        patch_name_1_savepath, 
        patch_2
    )

    return patch_name_0_savepath, patch_name_1_savepath

def extract_patch(
    image, 
    kp, 
    point_idx, 
    pad_size
):
    height, width = image.shape[:2]

    # Get keypoint coordinates
    x = int(kp[point_idx].pt[0])
    y = int(kp[point_idx].pt[1])

    # Calculate patch boundaries
    left, right = x - patch_size, x + patch_size
    top, bottom = y - patch_size, y + patch_size 

    if left < 0 or top < 0 or right > width or bottom > height:
        return None  

    patch = image[top:bottom, left:right]

    return patch

def sift_matching(
    frames_dir, 
    threshold, 
    MIN_MATCH_COUNT,
    PATCH_SIZE=224,
    patch_savedir='./data/patches_samples'
):
    n_ignore_frames = 0
    half_size = PATCH_SIZE // 2
    os.makedirs(patch_savedir, exist_ok=True)
    # Sift and Flann
    sift = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    frames_filenames = os.listdir(frames_dir)
    frames_filenames.sort()
    frames_filenames = frames_filenames[:30]
    frames_filepaths = [
        os.path.join(frames_dir, filename) \
            for filename in frames_filenames
    ]

    siftOut = {}
    print('Start pre-computing SIFT:')
    for filepath in tqdm(frames_filepaths):
        image1 = cv2.imread(filepath)
        kp_1, desc_1 = sift.detectAndCompute(image1, None)
        if desc_1 is None:
            continue
        siftOut[filepath] = (kp_1, desc_1, image1)

    siftout_keys_lst = list(siftOut.keys())
    n_frames = len(siftout_keys_lst)
    if n_frames % 2 != 0:
        last_key = siftout_keys_lst[-1]
        last_value = siftOut.pop(last_key)
        n_frames -= 1
    siftout_keys_lst = list(siftOut.keys())

    print('Start matching:')
    patch_idx = 0
    patches_lst = []
    for idx in tqdm(range(n_frames - 1)):
        frame_filepath_1 = siftout_keys_lst[idx]
        frame_filepath_2 = siftout_keys_lst[idx+1]

        kp_1, desc_1, image_1 = siftOut[frame_filepath_1]
        kp_2, desc_2, image_2 = siftOut[frame_filepath_2]

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []

        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_points.append(m)

        if len(good_points)>MIN_MATCH_COUNT:
            src_pts = np.float32(
                [kp_1[m.queryIdx].pt for m in good_points]
            ).reshape(-1,1,2)

            dst_pts = np.float32(
                [kp_2[m.trainIdx].pt for m in good_points]
            ).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
        else:
            # msg = f'Not enough matches are found - {len(good_points)}/{MIN_MATCH_COUNT}'
            # print(msg)
            matchesMask = None
            n_ignore_frames += 2

        if matchesMask is not None:
            for idx, (m, mask) in enumerate(zip(good_points, matchesMask)):
                if mask == 1:
                    patch_1 = extract_patch(
                        image=image_1, 
                        kp=kp_1, 
                        point_idx=m.queryIdx, 
                        pad_size=half_size
                    )
                    patch_2 = extract_patch(
                        image=image_2, 
                        kp=kp_2, 
                        point_idx=m.trainIdx, 
                        pad_size=half_size
                    )

                    if patch_1 == None or patch_2 == None:
                        continue

                    patch_name_0_savepath, patch_name_1_savepath = save_pair_patches(
                        patch_1=patch_1,
                        patch_2=patch_2,
                        patch_idx=patch_idx, 
                        patch_savedir=patch_savedir,
                    )
                    patches_lst.append(
                        {
                            'patch_left': patch_name_0_savepath,
                            'patch_right': patch_name_1_savepath,
                            'label': 0
                        }
                    )
                    patch_idx += 1

    return patches_lst, n_ignore_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--frames_dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7
    )
    parser.add_argument(
        '--min-match-count',
        type=int,
        default=10
    )
    parser.add_argument(
        '--save_label_path',
        type=str,
        default='./data/label.csv'
    )
    args = parser.parse_args()

    patches_lst, n_ignore_frames = sift_matching(
        frames_dir=args.frames_dir,
        threshold=args.threshold,
        MIN_MATCH_COUNT=args.min_match_count
    )
    print(f'''
    Patches Extraction complete!
    + Total frames ignore: {n_ignore_frames}
    ''')
    df = pd.DataFrame(patches_lst)
    df.to_csv(args.save_label_path, index=False, header=True)
    print('Save label.csv complete!')

if __name__ == '__main__':
    main()