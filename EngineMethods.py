from FeatureExtractors import extract_color_hist, GLCM, SIFT_BOVW, CNN, train_pca
from param import *
import numpy as np
from utils import save_pickle, load_pickle
import os
import cv2

class FeatureExtraction:
    def __init__(self):
        self.glcm = GLCM()
        self.sift_bovw = SIFT_BOVW()
        self.cnn = CNN()
        self.pca_model = self.get_pca_model()
    
    def extract_features(self, img):
        color_hist = extract_color_hist(img, hist_size)
        glcm_feat = self.glcm.extract_features(img, glcm_img_size, block_size)
        sift_bovw_feat = self.sift_bovw.extract_features(img)
        cnn_feat = self.cnn.extract_features(img)
        features = np.concatenate([color_hist, glcm_feat, sift_bovw_feat, cnn_feat])
        return features
    
    def extract_pca_features(self, img):
        features = self.extract_features(img)
        pca_features = self.pca_model.transform(features.reshape(1, -1))
        return pca_features[0]
        
    def load_database_features(self, features_type='features'):
        try:
            database_features = load_pickle(f'./pickle/{features_type}')
        except:
            database_features = {}
            for label in os.listdir(data_path):
                images = os.listdir(os.path.join(data_path, label))
                for image in images:
                    img_path = os.path.join(data_path, label, image)
                    img = cv2.imread(img_path)
                    if features_type == 'features':
                        features = self.extract_features(img)
                    elif features_type == 'pca_features':
                        features = self.extract_pca_features(img)
                    database_features[img_path] = features
            save_pickle(database_features, f'./pickle/{features_type}')
        return database_features
    
    def get_pca_model(self):
        try:
            pca_model = load_pickle('./pickle/pca_model')
        except:
            database_features = np.vstack(list(self.load_database_features('features').values()))
            pca_model = train_pca(database_features)
            save_pickle(pca_model, './pickle/pca_model')
        return pca_model


def get_labels(path):
    labels = load_pickle('./pickle/labels')
    return labels['/'.join(path.split('/')[2:])]
