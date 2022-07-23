from django.db import models

# Creating the instance in the Database

# Create your models here.
class Image(models.Model):
    first_image = models.ImageField()
    second_image = models.ImageField()

    def __str__(self):
        return "I am an image"


class StyleImage(models.Model):
    stylized_image = models.ImageField()

    def __str__(self):
        return "I am an image"




