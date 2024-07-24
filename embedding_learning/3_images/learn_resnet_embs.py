import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152
from PIL import Image
import os
from tqdm import tqdm
import pickle
import numpy as np

# Load pre-trained ResNet-152 model
resnet_model = resnet152(pretrained=True)
resnet_model.eval() 

# Pre-processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# note that, in our setting, we named the images with the corresponding ID in the dataset
# so the book x, whose ID is y, will be associated to the image called 'y.png'

paths = os.listdir('ml1m_or_dbbook_images/')
print('Number of images:', len(paths))
resnet_embs = dict()

for path in tqdm(paths, total=len(paths)):

  name = path.split('.')[0]
  
  image = Image.open('ml1m_or_dbbook_images/'+path).convert('RGB')
  input_image = transform(image).unsqueeze(0)

  with torch.no_grad():
      output = resnet_model(input_image)

  embedding = output.squeeze().numpy()

  if len(resnet_embs) == 0:
    print(f'Dim of the embedding: {len(embedding)}')

  resnet_embs[int(name)] = embedding

pickle.dump(resnet_embs, open('embs/resnet152_embs.pkl', 'wb'))

