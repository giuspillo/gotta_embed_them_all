import os
import torch
import random
import torchvision.transforms
import numpy as np

import itertools
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.io.video_reader import VideoReader

from tqdm import tqdm
from itertools import islice, takewhile

def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return seed

feature_layer = -2
original_model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_kinetics", num_classes=400, pretrained=True)

if isinstance(feature_layer, int):
    feature_layer = list(dict(original_model.named_modules()).keys())[feature_layer]

model = create_feature_extractor(original_model, {feature_layer: "feature_layer"}).to('cuda:0').eval()

for params in model.parameters():
    params.requires_grad = False

files_dir = "ml1m_videos"
videos = [os.path.join(files_dir, x) for x in sorted(os.listdir(files_dir))]

videos_outs = {}

for video_path in tqdm(videos):

    # reseed everything so that the order of the videos doesn't matter
    seed_everything(42)

    reader = VideoReader(video_path)
    frame_iter = (frame['data'] for frame in takewhile(lambda x: x['pts'] <= 30, reader.seek(0)))

    outs = []

    while True:

        try:

            clips_batch = []

            for _ in range(5):

                try:
                    clip = torch.stack(list(islice(frame_iter, 32))).to(torch.float32) / 255

                    clip = torchvision.transforms.Resize((128, 171))(clip)
                    clip = torchvision.transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])(clip)
                    clip = torchvision.transforms.CenterCrop((112, 112))(clip)

                    clips_batch.append(clip)
                    
                except Exception as e:
                    break

            # if last clip size doesn't match drop it
            if clips_batch[-1].shape[0] != 32:
                clips_batch = clips_batch[:-1]

            if len(clips_batch) == 0:
                break

            clips_batch = torch.stack(clips_batch)

            with torch.no_grad():
                out = model(clips_batch.moveaxis(1, 2).to('cuda:0'))['feature_layer'].cpu().detach().flatten(start_dim=1)
            outs.append(out)

        except Exception as e:
            break

    videos_outs[os.path.basename(video_path).split('.')[0]] = torch.vstack(outs)

np.save('features_r2_30.npy', videos_outs)

print("clip")
