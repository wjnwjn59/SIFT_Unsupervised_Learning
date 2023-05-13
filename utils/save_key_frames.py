import cv2
import argparse
import numpy as np 
import os
import multiprocessing as mp
import fnmatch

def collect_video_filepaths(
    data_dir, 
    extensions=[
        '*.mp4', 
        '*.avi', 
        '*.mkv', 
        '*.mov'
    ]
):
    video_filepath_lst = []
    for root, dir, files in os.walk(data_dir):
        for extension in extensions:
            for filename in fnmatch.filter(
                map(str.lower, files),
                extension
            ):
                video_filepath_lst.append(
                    os.path.join(root, filename)
                )

    return video_filepath_lst

def save_video_frames(
    video_path, 
    save_dir,
    interval,
):
    video_name = video_path.split('/')[-1].split('.')[0]
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(frame_rate * interval)
    frame_idx = 0

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

def process_video(args):
    save_video_frames(*args)

def main():
    parser = argparse.ArgumentParser(
        description='Arguments for save frames program'
    ) 
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./data/save_extracted_frames'
    )   
    parser.add_argument(
        '--time_interval',
        type=float,
        default=1
    )

    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    video_filepath_lst = collect_video_filepaths(
        data_dir=args.data_dir
    )

    num_processes = mp.cpu_count() - 4
    print('Total num processes: ', num_processes)

    with mp.Pool(num_processes) as p:
        p.map(
            process_video, 
            [
                (f, args.save_dir, args.time_interval) \
                    for f in video_files
            ]
        )

if __name__ == '__main__':
    main()