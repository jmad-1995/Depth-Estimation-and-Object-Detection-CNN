import argparse
import torch
import networks
import skvideo
skvideo.setFFmpegPath(r'C:\Program Files\FFmpeg\bin')
import skvideo.io as io
import numpy as np
import matplotlib.cm as cm

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-video_in', required=True, help='Path to the input video file')
parser.add_argument('-video_out', required=True, help='Path to the output video file')
parser.add_argument('-saved_model', required=True, help='Path to the trained model')
parser.add_argument('-device', default=0, type=int, help='CUDA device')
parser.add_argument('-axis', default=1, type=int, help='Axis to stack RGB image and depth prediction')
args = parser.parse_args()

# CUDA device
device = "cuda:{}".format(args.device) if torch.cuda.is_available() else 'cpu'

# Create the reader and writer
reader = io.FFmpegReader(filename=args.video_in)
writer = io.FFmpegWriter(filename=args.video_out)

# Trained model
model = networks.MultiPurposeCNN().to(device)
model.load_state_dict(torch.load(args.saved_model))
model.eval()

cmapper = cm.get_cmap('plasma')
idx = 0
for frame_in in reader.nextFrame():
    idx += 1
    print("Predicting frame {:05d} of {}".format(idx, reader.getShape()[0]))

    rgb, depth = model.predict(frame_in, device)

    depth_color = cmapper(np.clip(depth / 11., 0., 1.))[..., :3]

    frame_out = np.concatenate([rgb, depth_color], axis=args.axis)

    writer.writeFrame(np.uint8(frame_out * 255.))

reader.close()

