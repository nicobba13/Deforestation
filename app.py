import pathlib
import random

import streamlit as st
import torch

from PIL import Image
from fastai.vision.all import *

if platform.system() == 'Windows':
    # fastai Load_Learner uses PosixPath, use a trick to load the model
    _saved_hack = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    classification_model = load_learner('models/palmoil_fastai_resnet18.pth')
    pathlib.PosixPath = _saved_hack
else:
    classification_model = load_learner('models/palmoil_fastai_resnet18.pth')

def classify_image(image) -> bool:
    """Classify if an image has palm trees or not
    
    Parms:
    - image(bytes): image to be classify
    
    Returns:
    - bool: True if image contains palm trees
    """
    tensor = torch.tensor(np.asarray(Image.open(image).convert('RGB').resize((256, 256))))
    results = classification_model.predict(tensor)
    return results[0] == '1'
    # return random.choice([True, False])
    
def analyze_image(image):
    """Analyze parts of image that most contributed to classification
    
    Parms:
    - image(bytes): image to analyse
    
    Returns:
    - bytes: image showing most contributive part of the image
    """
    return Image.open(image).convert("L")
    
st.write("# Palm Tree Detector🌴")

st.write("Oh yeah baby 😎")

with st.sidebar:
    image_file = st.file_uploader('Choose an image💵', 
                              type=['png', 'jpg'],
                              help='Select image to be classified',
                              )

if image_file is not None:
    bytes_image = image_file.getvalue()
    st.image(bytes_image, use_column_width=True)

if st.button("Detect palm trees", disabled=(image_file is None)):
    result = classify_image(image_file)
    if result:
        st.write('Palm trees detected')
    else:
        st.write('Palm trees not detected')
        
    with st.expander('Sensitivity Analysis', expanded=False):
        analysis_image = analyze_image(image_file)
        st.image(analysis_image, width=300)
    with st.expander('Sensitivity Analysis', expanded=False):
        analysis_image = analyze_image(image_file)
        st.image(analysis_image, width=300)
