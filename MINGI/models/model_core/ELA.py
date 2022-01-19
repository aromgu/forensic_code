from PIL import Image
from PIL import Image, ImageChops, ImageEnhance

from PIL import Image, ImageChops, ImageEnhance
import sys, os.path

filename = '../monk.tif'
resaved = filename + '.resaved.jpg'
ela = filename + '.ela.png'

im = Image.open(filename)

im.save(resaved, 'JPEG', quality=95)
resaved_im = Image.open(resaved)

ela_im = ImageChops.difference(im, resaved_im)
extrema = ela_im.getextrema()
max_diff = max([ex[1] for ex in extrema])
scale = 255.0/max_diff

ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

print("Maximum difference was %d" % (max_diff))
ela_im.save(ela)
ela_im.show()