import os
import torch
import numpy as np

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.io.video_reader import VideoReader

from tqdm import tqdm
import random
import itertools
from src.ml1m.torchaudio_vggish import VGGISH


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


feature_layer = -1
original_model = VGGISH.get_model().to('cuda:0')

if isinstance(feature_layer, int):
    feature_layer = list(dict(original_model.named_modules()).keys())[feature_layer]

model = create_feature_extractor(original_model, {feature_layer: 'feature_layer'}).to("cuda:0").eval()
input_processor = VGGISH.get_input_processor()

for params in model.parameters():
    params.requires_grad = False

files_dir = "ml1m_audio"
videos = [os.path.join(files_dir, x) for x in sorted(os.listdir(files_dir))]

videos_outs = {}

for video_path in tqdm(videos):

    # reseed everything so that the order of the videos doesn't matter
    seed_everything(42)

    reader = VideoReader(video_path, stream="audio")
    # frames = torch.vstack([frame['data'] for frame in reader])

    frames = []
    for frame in itertools.takewhile(lambda x: x['pts'] <= 30, reader.seek(0)):
        frames.append(frame['data'])
    frames = torch.vstack(frames)

    outs = []

    audio = input_processor(frames.mean(1))

    with torch.no_grad():
        out = model(audio.to('cuda:0'))['feature_layer'].cpu().detach()
    outs.append(out)

    videos_outs[os.path.basename(video_path).split('.')[0]] = torch.vstack(outs)

np.save('features_audio_30_vggish.npy', videos_outs)
