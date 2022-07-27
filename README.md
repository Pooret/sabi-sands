# Welcome to Sabi Sands!

## Overview
Sabi Sands is a magical place where you can transfer a style of an image to another one.

The user will be prompt to upload two images. The first image will the content image and the second image will be the style image. Once the images are uploaded, the user will click on Transform and the style from the style image will be transferred to the content image.

## Technology Details
This app is built using Django and Python. It uses Tensorflow to provide the image processing.

## How to Run it Locally
1. Clone the Repository
```
git clone git@github.com:Biased-Outliers/sabi-sands.git
```
2. Create a Virtual Environment and activate it (Recommended)
```
python3 -m venv VIRTUAL_ENVIRONMENT_NAME
source VIRTUAL_ENVIRONMENT_NAME/bin/activate
```
3. Go into the cloned repository sabi-sands while in your virutal environment
```
cd sabi-sands
```
4. Install the necessary requirements
```
pip install -r requirements.txt
```
5. Run Sabi Sands locally 
```
python3 manage.py runserver
```

