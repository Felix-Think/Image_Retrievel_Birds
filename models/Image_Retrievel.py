#|%%--%%| <wF8dcpASeB|k8KU2k6VcN>


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

class ImageProcessor:
    def __init__(self, image_dir='images'):
        self.image_dir = image_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(self.device)

    def load_images(self, filenames):
        images = []
        for filename in filenames:
            filepath = os.path.join(self.image_dir, filename)
            image = cv2.imread(filepath)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
        return images

    def preprocess_images(self, images):
        inputs = self.processor(images, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0, :].cpu().numpy()
        return outputs

    def cosine_similarity(self, query_vector, src_vectors):
        dot_product = np.dot(src_vectors, query_vector)
        src_norm = np.linalg.norm(src_vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        return dot_product / (src_norm * query_norm)

    def rank_images(self, query_vector, src_vectors, top_k=5):
        scores = self.cosine_similarity(query_vector, src_vectors)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return ranked_indices, scores[ranked_indices]

    def display_images(self, query_image, ranked_indices, scores, src_images, dataset_dir):
        print('Query Image')
        plt.figure(figsize=(3, 3))
        plt.imshow(query_image)
        plt.axis('off')
        plt.show()
        
        print(f'Top {len(ranked_indices)} similar images:')
        for idx, src_idx in enumerate(ranked_indices):
            print(dataset_dir['Index'][src_idx])
            plt.figure(figsize=(3, 3))
            plt.imshow(src_images[src_idx])
            plt.title(f'Similarity: {scores[idx]:.10f}')
            plt.axis('off')
            plt.show()


    def get_similar_images(self, query_image_path, preprocessed_src_images, src_images, dataset_dir, top_k=5):
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            print(f"Error reading query image {query_image_path}")
            return [], []

        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        preprocessed_query_image = self.preprocess_images([query_image])[0]
        ranked_indices, scores = self.rank_images(preprocessed_query_image, preprocessed_src_images, top_k)
        return ranked_indices, scores

#|%%--%%| <k8KU2k6VcN|onk7zfjwDA>
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
image_processor = ImageProcessor()
dataset_dir = pd.read_csv('images.csv')
src_images = image_processor.load_images(dataset_dir['Links Image'])
preprocessed_src_images = np.load('preprocessed_src_images.npy')
query_image_path = '/home/felix/ML1/Project/models/images/1.jpg'
image_processor.get_similar_images(query_image_path, preprocessed_src_images, src_images, dataset_dir, top_k=5)

