import numpy as np
from keras.preprocessing import image
from keras.applications import mobilenet
from IPython.display import Image
from keras.models import load_model

model = load_model('ir-model-v0')

def process_image(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  pImg = mobilenet.preprocess_input(img_array)
  return pImg

img_location = 'test-img1.jpg'
Image(img_location)
im = process_image(img_location)
prediction = model.predict(im)
print(prediction)