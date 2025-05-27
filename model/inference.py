import argparse
from path import Path
import torch
import torch.nn
import torch.backends.cudnn as cudnn
import models
from tqdm import tqdm
from imageio import imwrite
import numpy as np
from PIL import Image
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='MaskDICNet inference',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', default='MaskDICNet', choices=['MaskDICNet'], help='network')
parser.add_argument('--data', metavar='DIR',
                    help='path to images folder, image names must match \'[name]1.[ext]\' and \'[name]2.[ext]\'')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model')
parser.add_argument('--output', '-o', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument("--img-exts", metavar='EXT', default=['tif', 'png', 'jpg', 'bmp', 'ppm'], nargs='*', type=str,
                    help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()

    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))

    if args.output is None:
        save_path = data_dir/'flow'
    else:
        save_path = Path(args.output)
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()

    img_pairs = []
    for ext in args.img_exts:
        test_files = data_dir.files('*1.{}'.format(ext))
        for file in test_files:
            img_pair = file.parent / (file.stem[:-1] + '2.{}'.format(ext))
            if img_pair.isfile():
                img_pairs.append([file, img_pair])

    print('{} samples found'.format(len(img_pairs)))

    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](network_data).to(device)
    
    model = torch.nn.DataParallel(model)
    
    model.eval()
    cudnn.benchmark = True

    total_time = 0

    for (img1_file, img2_file) in tqdm(img_pairs):
        start_time = time.time()

        img1 = np.array(Image.open(img1_file))
        img2 = np.array(Image.open(img2_file))

        img1 = img1[np.newaxis, ...]
        img2 = img2[np.newaxis, ...]

        img1 = img1[np.newaxis, ...]
        img2 = img2[np.newaxis, ...]

        img1 = torch.from_numpy(img1 / 255.0).float()
        img2 = torch.from_numpy(img2 / 255.0).float()

        img1 = torch.cat([img1], 1).to(device)
        img2 = torch.cat([img2], 1).to(device)

        flow1, flow2, flow3, mask1, mask2 = model(img2, img1)

        mask_output = torch.round(mask1)

        output_to_write = flow1.data.cpu()
        output_mask_to_write = mask_output.data.cpu()
        output_to_write = output_to_write.numpy()
        output_mask_to_write = output_mask_to_write.numpy()

        disp_x = output_to_write[0, 0, :, :]
        disp_x = disp_x
        disp_y = output_to_write[0, 1, :, :]
        disp_y = disp_y

        mask_output_img = output_mask_to_write[0, 0, :, :]
        mask_output_img = mask_output_img * 255
        mask_output_img = mask_output_img.astype(np.uint8)

        filenamex = save_path/'{}{}'.format(img1_file.stem[:-1], '_disp_x')
        filenamey = save_path/'{}{}'.format(img1_file.stem[:-1], '_disp_y')
        filename_mask = save_path / '{}{}'.format(img1_file.stem[:-1], '_mask_output.tif')

        np.savetxt(filenamex + '.csv', disp_x,delimiter=',')
        np.savetxt(filenamey + '.csv', disp_y,delimiter=',')
        imwrite(filename_mask, mask_output_img)

        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time

        print(f'Inference time for {img1_file.stem}: {inference_time:.4f} seconds')

    print(f'Total inference time: {total_time:.4f} seconds')

   
if __name__ == '__main__':
    main()

