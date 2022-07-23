from django import forms

from .models import Image, StyleImage

class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = {'first_image', 'second_image'}

class ImageFormStyleTransfer(forms.ModelForm):
    class Meta:
        model = StyleImage
        fields = {'stylized_image'}