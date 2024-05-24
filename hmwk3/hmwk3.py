import numpy as np  
from sklearn import svm
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import cv2 as cv
import os

class MyDataset(Dataset):
    def __init__(self,path):
        self.transform = transforms.ToTensor()
        self.root=path
        
    def get_files(self,path):
        paths=os.walk(path)
        