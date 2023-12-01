import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.cluster import KMeans
from param import *
import os
from utils import save_pickle, load_pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Color Hist
def extract_color_hist(img, hist_size):
    channels = {}
    for i in range(3):
        hist = cv2.calcHist(img, [i], None, [hist_size], [0, 256])
        channels[i] = hist
    feature_vect = np.concatenate(list(channels.values())).squeeze()
    feature_vect = feature_vect / (img.shape[0] * img.shape[1])
    return feature_vect.astype(np.float32())

#GLCM
class GLCM:
    def __init__(self):
        self.properties = ['contrast', 'homogeneity', 'energy', 'correlation']

    def process_image(self, img, img_size):
        # convvert to gray, resize, and quantize image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_CUBIC) // 64
        return img
    
    def extract_features(self, img, img_size, block_size):
        img = self.process_image(img, img_size)
        features = []
        for y in range(0, img_size - block_size + 1, block_size):
            for x in range(0, img_size - block_size + 1, block_size):
                block = img[y:y+block_size, x:x+block_size]
                glcm = graycomatrix(block, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=4)
                block_features = np.ravel([graycoprops(glcm, prop).reshape(1, -1) for prop in self.properties])
                features.append(block_features)
        return np.concatenate(features, dtype=np.float32())

#SIFT
class SIFT_BOVW:
    #extract bag of visual words from the dataset using k-means
    def __init__(self):
        self.data_path = data_path
        self.k = kmeans_nocenters
        self.sift = cv2.SIFT_create()
        self.load_kmeans()

    def load_kmeans(self):
        try:
            self.kmeans = load_pickle('./pickle/kmeans')
        except:
            self.run_kmeans()

    def run_kmeans(self):
        dataset_descriptors = self.get_descriptors()
        points = np.vstack(list(dataset_descriptors.values()))
        del(dataset_descriptors)
        self.kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=5)
        self.kmeans.fit(points)
        save_pickle(self.kmeans, './pickle/kmeans')

    def get_descriptors(self):
        try: 
            dataset_descriptors = load_pickle('./pickle/sift_descriptors')
        except:
            dataset_descriptors = {}
            for label in os.listdir(self.data_path):
                images = os.listdir(os.path.join(self.data_path, label))
                for image in images:
                    img_path = os.path.join(self.data_path, label, image)
                    img = cv2.imread(img_path)
                    _, desc = self.sift.detectAndCompute(img, None)
                    if not type(desc) == type(None):
                        dataset_descriptors[img_path] = desc
            save_pickle(dataset_descriptors, './pickle/sift_descriptors')
        return dataset_descriptors

    def extract_features(self, img):
        _, desc = self.sift.detectAndCompute(img, None)
        if type(desc) == type(None):
            histogram = np.zeros(self.k)
        else:
            visual_words = self.kmeans.predict(desc)
            histogram, _ = np.histogram(visual_words, bins=range(self.k + 1))
            histogram = histogram / desc.shape[0]
        return histogram
        
#CNN
class CNN:
    def __init__(self):
        pretrained = resnet18(weights='DEFAULT')
        self.model = torch.nn.Sequential(*list(pretrained.children())[:-1]).eval()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def extract_features(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            features = self.model(img).squeeze().numpy()
        return features
    
#PCA
def train_pca(dataset):
    pca = PCA(n_components=pca_nocomponents)
    scaler_pca_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', pca)
    ])
    scaler_pca_pipeline.fit(dataset)
    return scaler_pca_pipeline
