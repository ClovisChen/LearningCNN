import  os
import numpy as np
import scipy
from glob import glob

data_root = "/home/chen-tian/data/data/KITTI/"
data_seq = data_root + "sequences/"
dataset_name = "kitti_odom"
train_dir = data_root + "geom_train/"

IMG_HEIGHT = 128
IMG_WIDTH = 416
SEQ_LENGTH = 3

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def save_kitti_odom_train(n):
    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        return
    image_seq = concat_image_seq(example['image_seq'])
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    dump_dir = os.path.join(train_dir, example['folder_name'])
    # if not os.path.isdir(dump_dir):
    #     os.makedirs(dump_dir, exist_ok=True)
    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

## shuffle images in kitti
def split_train_val():
    np.random.seed(8964)
    subfolders = os.listdir(train_dir)
    with open(train_dir + 'train.txt', 'w') as tf:
        with open(train_dir + 'val.txt', 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(train_dir + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(train_dir, s, '*.jpg'))
                frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                for frame in frame_ids:
                    if np.random.random() < 0.1:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))


if __name__ == '__main__':
    if dataset_name == "kitti_odom":
        from kitti_odom_loader import kitti_odom_loader
        global data_loader
        data_loader = kitti_odom_loader(data_root,
                                        img_height=IMG_HEIGHT,
                                        img_width=IMG_WIDTH,
                                        seq_length=SEQ_LENGTH)

        [save_kitti_odom_train(n) for n in range(data_loader.num_train)]
        # Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n) for n in range(data_loader.num_train))

        # Split into train/val
        split_train_val()
