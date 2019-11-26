import argparse
import torch
import networks
import skvideo.io as io

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-video_in', required=True, help='Path to the input video file')
parser.add_argument('-video_out', required=True, help='Path to the output video file')
parser.add_argument('-saved_model', required=True, help='Path to the trained model')
parser.add_argument('-device', default=0, type=int, help='CUDA device')
parser.add_argument('-axis', default=0, type=int, help='Axis to stack RGB image and depth prediction')
args = parser.parse_args()

# Create the reader and writer
reader = io.FFmpegReader(filename=args.video_in)
writer = io.FFmpegWriter(filename=args.video_out)

#
model = networks.MultiPurposeCNN()
model.load_state_dict(torch.load(args.saved_model))
model.eval()

for idx in reader.getShape()[0]:

    frame_in = reader.nextFrame()

