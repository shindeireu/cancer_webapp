import streamlit as st
import tensorflow as tf
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('us_model_web2.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Cancer Classification
         """
         )

file = st.file_uploader("Please upload an ultrasound image file", type=["jpg", "png"])
import cv2
import PIL
import numpy as np
from PIL import ImageOps, Image
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (128,128)    
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = PIL.Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Benign', 'Malignant']
    if (predictions[0]<=0.5):
      index=0
    else:
      index=1
    # score = tf.nn.softmax(predictions[0])
    # st.write(prediction)
    # st.write(score)
    string="This above ultrasound is " + class_names[index]
    st.success(string)
