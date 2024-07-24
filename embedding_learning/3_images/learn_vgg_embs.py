from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import numpy as np
import os
import pickle

# load vgg19
vgg19_model = VGG19(weights='imagenet')

# read the fc2 layer
image_emb = vgg19_model.get_layer('fc2').output

# define the overall model
model = Model(inputs=vgg19_model.input, outputs=image_emb)
model.summary()


emb_dict = dict()

# learn the image embeddings
for i in range(6040, 9251):

  if i % 100 == 0:
    print('Progress:', str(i))

  img_path = 'ml1m_or_dbbook_images/'+str(i)+'.jpg'

  if os.path.exists(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    emb = model.predict(x)

    emb_dict[i]=emb[0]


pickle.dump(emb_dict, open('embs/vgg_embs.pkl','wb'))
