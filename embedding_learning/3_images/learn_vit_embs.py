from transformers import AutoImageProcessor, ViTModel
from datasets import load_dataset
from PIL import Image 
from tqdm import tqdm
import torch
import numpy as np
import os
import pickle


# load pre-trained vit model
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
print(model.config)

# note that, in our setting, we named the images with the corresponding ID in the dataset
# so the book x, whose ID is y, will be associated to the image called 'y.png'

embs = dict()
paths = os.listdir('ml1m_or_dbbook_images/')
print('Number of images:', len(paths))

for path in tqdm(paths, total=len(paths)):

  name = path.split('.')[0]

  image = Image.open('ml1m_or_dbbook_images/'+path).convert('RGB')
  inputs = image_processor(image, return_tensors="pt")

  with torch.no_grad():
    outputs = model(**inputs)

  last_hidden_states = outputs.last_hidden_state
  embs[name] = last_hidden_states

pickle.dump(embs, open('embs/vit_cls.pickle', 'wb'))