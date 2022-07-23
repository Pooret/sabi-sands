from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import IPython.display as display
import PIL
from .forms import ImageForm,  ImageFormStyleTransfer
import numpy as np

import tensorflow as tf

#import matplotlib.pyplot as plt
#import matplotlib as mpl
from PIL import Image
import numpy as np

import tensorflow as tf

# Create your views here.
def hello(request):
    if request.method=='POST':
        
        # Database
        ## Do WE NEED TO SAVE IT TO DATABASE?
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance
            stylized_image = tensor_to_image(apply_something(img_obj))

            stylized_image_form = ImageFormStyleTransfer(stylized_image)
            stylized_image_form.save()


            return render(request, 'homeApp/index.html', {'image':stylized_image_form.instance})
        
    form = ImageForm()
    return render(request, 'homeApp/index.html', {
        'form':form
    })

def apply_something(img_obj):
    #print(img_obj.first_image.url)

    PATH = "/Users/tylerpoore/Workspace/Biased Outliers/django/sabisands_clone/sabi-sands"

    img1 = tf.io.read_file(PATH + img_obj.first_image.url)
    img2 = tf.io.read_file(PATH + img_obj.second_image.url)

    content_image = image_norm(img1)
    style_image = image_norm(img2)

    import tensorflow_hub as hub

    hub_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
    return hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    #return tensor_to_image(stylized_image)

def image_norm(img, max_dim=512):

    max_dim = max_dim
    img = tf.image.decode_image(img, channels=3)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape*scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return(img)

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)