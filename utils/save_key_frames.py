import cv2
import argparse
import numpy as np 
import os

def save_video_frames(
    video_path, 
    save_dir,
    interval,
):
    video_name = video_path.split('/')[-1].split('.')[0]
    cap = cv2.VideoCapture(video_path)
    #cap.set(cv2.CAP_PROP_FPS, fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(frame_rate * interval)
    frame_idx = 0

    print('Start extracing frames...')
    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            output_path = f'{save_dir}/{video_name}_frame_{frame_idx:05d}.jpg'
            print(f'Saving to {output_path}')
            cv2.imwrite(output_path, frame)
            frame_idx += 1
    print('Extraction completed!')
        
    cap.release()


def main():
    parser = argparse.ArgumentParser(description='Arguments for save frames program') 
    parser.add_argument(
        '--video_path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./resources/save_extracted_frames'
    )   
    parser.add_argument(
        '--time_interval',
        type=float,
        default=1
    )

    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_video_frames(
        args.video_path,
        args.save_dir,
        args.time_interval
    )


if __name__ == '__main__':
    main()