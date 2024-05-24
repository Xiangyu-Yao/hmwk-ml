import os
import numpy as np
import cv2 as cv
path="hmwk3/omniglot-py/images_background"

p= os.walk(path)
paths=[]

count=0
for filepath,dirnames,filenames in p:
    if len(filenames)!=20:
        continue
    if(count==200):
        break
    count+=1
    for filename in filenames:
        paths.append(os.path.join(filepath,filename))
        label=filename.split('_')[0]

print(paths[-1])
print(len(paths))
print(count)

img=cv.imread(paths[-1])
print(img.shape)

