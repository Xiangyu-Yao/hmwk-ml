import os
import numpy as np
path="hmwk3/omniglot-py/images_background"

p= os.walk(path)
paths=[]
for i,(filepath,dirnames,filenames) in enumerate(p):
    if(i==200):
        break
    for filename in filenames:
        paths.append(os.path.join(filepath,filename))
        
print(paths[:10])
print(len(paths))
    
    