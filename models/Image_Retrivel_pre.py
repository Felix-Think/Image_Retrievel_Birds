import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

#|%%--%%| <JzhYP5DmHI|adcoTZCQ1o>
#Declare source images direction path
dataset_dir = pd.read_csv('images.csv')
name_dir = 'images'
#Get all image filenames inside  'dataset_dir'
image_filenames = dataset_dir['Links Image'].values
Index = dataset_dir['Index'].values
Index
#|%%--%%| <adcoTZCQ1o|CiwQGwNHQN>

#Declare empty list
src_images = []

for filename in image_filenames:
    #Create filepath of current image
    filepath = os.path.join(
        name_dir,
        filename
    )

    image = cv2.imread(filepath)
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
    src_images.append(image)




#|%%--%%| <CiwQGwNHQN|9hYIRGcjuT>
r"""°°°
# BASE LINE
°°°"""
#|%%--%%| <9hYIRGcjuT|pew5K2cxZN>

#1. RESIZE IMAGE

def image_resize(images, target_size = (224, 224)):
    resized_image = cv2.resize(
        images,
        target_size
    )
    return resized_image

#|%%--%%| <pew5K2cxZN|gwo2cGLIsn>

#2. NORMALIZE IMAGE

def calculate_mean_std(images):
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1,2))
    return mean, std

def image_std_normalize(images, mean, std):
    normalized_image = (images - mean) / std
    return normalized_image
#|%%--%%| <gwo2cGLIsn|EffO4gJ2Pw>

#3. FLATTEN IMAGE

def image_flatten(images, is_batch = False):
    if is_batch:
        flatten_images = images.reshape(images.shape[0], -1)
    else:
        flatten_images = images.reshape(-1)
    return flatten_images

#|%%--%%| <EffO4gJ2Pw|YTdkEGZS8e>

#. PREPROCESS IMAGE

def preprocess_images(images):
    resized_images = [
        image_resize(image) for image in images
    ]
    image_arr = np.array(resized_images)
    mean, std = calculate_mean_std(image_arr)
    normalized_images = image_std_normalize(
        image_arr,
        mean,
        std
    )

    flattened_images = image_flatten(
        normalized_images,
        is_batch = True
    )
    return flattened_images, mean, std

def process_image(image):
    resized_image = image_resize(image)
    mean, std = calculate_mean_std(resized_image)
    normalized_image = image_std_normalize(
        resized_image,
        mean,
        std
    )
    flattened_image = image_flatten(normalized_image)
    return flattened_image

#|%%--%%| <YTdkEGZS8e|r0mwmMQcGz>
r"""°°°
# PIPELINE
°°°"""
#|%%--%%| <r0mwmMQcGz|xrrhCKASdM>

#5. Defile metrics
def mean_squared_error(query_vector, src_vectors):
    square_diff = (query_vector - src_vectors) ** 2
    mse = np.mean(square_diff, axis = 1)
    return mse

def mean_absolute_error(query_vector, src_vectors):
    abs_diff = np.abs(query_vector - src_vectors)
    abs_diff = np.abs(query_vector - src_vectors)
    mae = np.mean(abs_diff, axis = 1)
    return mae

def cosine_similarity(query_vector, src_vectors):
    dot_product = np.dot(
        src_vectors,
        query_vector
    )
    src_norm = np.linalg.norm(src_vectors, axis = 1)
    query_norm = np.linalg.norm(query_vector)
    cosine_similarity = dot_product / (src_norm * query_norm)
    return cosine_similarity

def correlation_coefficient(query_vector, src_vectors):
    corr_coeff = np.corrcoef(
        src_vectors,
        query_vector
    )[:-1, -1]

    return corr_coeff



#|%%--%%| <xrrhCKASdM|VRPfH2ToTw>
r"""°°°
# 5.RANKING
°°°"""
#|%%--%%| <VRPfH2ToTw|VpNUBUKo2I>

def ranking(
    preprocessed_query_image,
    preprocessed_src_images,
    top_k = 10
):
    scores = cosine_similarity(
        preprocessed_query_image,
        preprocessed_src_images
    )

    ranked_list = np.argsort(scores)[::-1][:top_k]
    scores = scores[ranked_list]
    return ranked_list, scores



#|%%--%%| <VpNUBUKo2I|TOZW7I8FXl>

#RANKING PROGRAM
query_image_paths = [
   '/home/felix/ML1/Project/modules/images/300.jpg'
]

top_k = 5

for query_image_path in query_image_paths:
    query_image = cv2.imread(query_image_path, 1)
    preprocessed_query_image = process_image(query_image)
    preprocessed_src_images, mean,std = preprocess_images(src_images)
    ranked_list, scores = ranking(
        preprocessed_query_image,
        preprocessed_src_images,
        top_k = top_k
    )

    print('Query Image')
    plt.figure(figsize = (3, 3))
    plt.imshow(query_image)
    plt.axis('off')
    plt.show()
    print(f'Top {top_k} similar images')
    for idx in range(len(ranked_list)):
        print(dataset_dir['Index'][ranked_list[idx]])
        src_image_idx = ranked_list[idx]
        similarity_score = scores[idx]
        plt.figure(figsize = (3, 3))
        plt.imshow(src_images[src_image_idx])
        plt.title(f'Similarity: {similarity_score:.10f}')
        plt.axis('off')
        plt.show()




#|%%--%%| <TOZW7I8FXl|xvHLz8owDU>
r"""°°°
# Apply pre-trained ViT
°°°"""
#|%%--%%| <xvHLz8owDU|FzoV848uq1>

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

#|%%--%%| <FzoV848uq1|V63B66tSjj>


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)

#|%%--%%| <V63B66tSjj|MrfqkO4ESi>

#Define preprocessing  function
def preprocessing(images):
    if isinstance(images, list):
        inputs = processor(images, return_tensors='pt', padding=True).to(device)
    else:
        inputs = processor(images, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0, :].detach().cpu().numpy()
    return outputs


#|%%--%%| <MrfqkO4ESi|AdhIR2Uv98>

def ranking(preprocessed_query_image, preprocessed_src_images, top_k=10):
    scores = cosine_similarity(preprocessed_query_image, preprocessed_src_images)
    ranked_list = np.argsort(scores)[::-1][:top_k]
    scores = scores[ranked_list]
    return ranked_list, scores

#|%%--%%| <AdhIR2Uv98|JAWSv0c1TB>

#RANKING PROGRAM


query_image_paths = [
   '/home/felix/ML1/Project/modules/images/123.jpg'
]

top_k = 5

for query_image_path in query_image_paths:
    query_image = cv2.imread(query_image_path, 1)
    if query_image is None:
        print(f"Error reading query image {query_image_path}")
        continue

    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    preprocessed_query_image = preprocessing([query_image])[0]
    preprocessed_src_images = preprocessing(src_images)
    ranked_list, scores = ranking(preprocessed_query_image, preprocessed_src_images, top_k=top_k)

    print('Query Image')
    plt.figure(figsize=(3, 3))
    plt.imshow(query_image)
    plt.axis('off')
    plt.show()
    print(f'Top {top_k} similar images')
    for idx in range(len(ranked_list)):
        print(dataset_dir['Index'][ranked_list[idx]])
        src_image_idx = ranked_list[idx]
        similarity_score = scores[idx]
        plt.figure(figsize=(3, 3))
        plt.imshow(src_images[src_image_idx])
        plt.title(f'Similarity: {similarity_score:.10f}')
        plt.axis('off')
        plt.show()

#|%%--%%| <JAWSv0c1TB|I1dEybLHo4>


#preprocessed_src_images = preprocessing(src_images)

#|%%--%%| <I1dEybLHo4|mCU6yUQXiv>

np.save('preprocessed_src_images.npy', preprocessed_src_images)

#|%%--%%| <mCU6yUQXiv|fAUdzxtajg>

preprocessed_src_images = np.load('preprocessed_src_images.npy')

#|%%--%%| <fAUdzxtajg|7pakm9dH5Y>

def get_similar_images(query_image_path,preprocessed_src_images, top_k = 5):
    query_image = cv2.imread(query_image_path, 1)
    if query_image is None:
        print(f"Error reading query image {query_image_path}")
        return

    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    preprocessed_query_image = preprocessing([query_image])[0]
    ranked_list, scores = ranking(preprocessed_query_image, preprocessed_src_images, top_k=top_k)
    print('Query Image')
    plt.figure(figsize=(3, 3))
    plt.imshow(query_image)
    plt.axis('off')
    plt.show()
    print(f'Top {top_k} similar images')
    for idx in range(len(ranked_list)):
        print(dataset_dir['Index'][ranked_list[idx]])
        src_image_idx = ranked_list[idx]
        similarity_score = scores[idx]
        plt.figure(figsize=(3, 3))
        plt.imshow(src_images[src_image_idx])
        plt.title(f'Similarity: {similarity_score:.10f}')
        plt.axis('off')
        plt.show()
#|%%--%%| <7pakm9dH5Y|wF8dcpASeB>

query_image_path ='/home/felix/ML1/Project/models/images/1.jpg'

get_similar_images(query_image_path,preprocessed_src_images, top_k = 5)






