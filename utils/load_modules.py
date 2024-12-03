import torch

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import gc
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(path):
    return Image.open(path)
# Function to save the cropped images to a folder
def save_images(cropped_images, folder_path="clustered"):
    """
    Saves the cropped images into a specified folder.
    
    Args:
    - cropped_images (list of PIL.Image): List of cropped images to save.
    - folder_path (str): The path to the folder where images will be saved.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        # Delete all files in the folder before saving new ones
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file {file_path}")
    # Save each cropped image with a unique filename
    for idx, img in enumerate(cropped_images):
        save_path = os.path.join(folder_path, f"cluster_{idx + 1}.jpg")
        img.save(save_path)
        print(f"Saved image {save_path}")


    
def unload_model(model):
    """
    Unload the model and free up GPU memory.
    
    Args:
    - model: The model to unload (e.g., LlavaNextForConditionalGeneration).
    """
    # Delete the model
    del model
    # Empty CUDA cache to release GPU memory
    torch.cuda.empty_cache()
    # Run garbage collection to clean up any unused objects
    gc.collect()
    print("Model unloaded and memory freed.")
    
# Function to generate masks for the image
def generate_masks(mask_generator, image):
    """
    Generate segmentation masks for the input image using the mask generator.
    """
    # Convert the PIL image to a NumPy array (SAM expects a NumPy array)
    image_np = np.array(image)
    
    # Move the image to the same device as the model (CUDA or CPU)
    image_tensor = torch.tensor(image_np).to(device)
    
    # Generate masks for the image
    masks = mask_generator.generate(image_tensor.cpu().numpy())  # Use .cpu() because mask_generator expects a NumPy array
    
    return masks

# Function to calculate bounding boxes for each mask
def get_bounding_boxes(masks):
    """
    Calculate bounding boxes for each segmented mask.
    """
    bounding_boxes = []
    
    for mask in masks:
        # Get the segmentation mask (binary mask)
        seg_mask = mask['segmentation']  # A binary mask: 1 for the object, 0 for background
        
        # Find the coordinates of all points in the mask (where the value is 1)
        y, x = np.where(seg_mask == 1)  # Get the row (y) and column (x) coordinates of all the pixels in the mask
        
        # Get the bounding box of the segmented area
        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()
        
        bounding_boxes.append((min_x, max_x, min_y, max_y))
    
    return bounding_boxes

# Function to crop the image based on bounding boxes
def crop_segments(image, bounding_boxes):
    """
    Crop the original image based on the bounding boxes for each segment.
    """
    cropped_images = []
    
    for (min_x, max_x, min_y, max_y) in bounding_boxes:
        cropped_img = image.crop((min_x, min_y, max_x + 1, max_y + 1))  # +1 to include max_x and max_y
        cropped_images.append(cropped_img)
    
    return cropped_images

# Function to display the original image with bounding boxes and cropped segments
def display_segments(image, bounding_boxes, cropped_images):
    """
    Display the original image with bounding boxes and each cropped segment.
    """
    # Display the original image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for (min_x, max_x, min_y, max_y) in bounding_boxes:
        plt.plot([min_x, max_x], [min_y, min_y], color='red', linewidth=2)
        plt.plot([min_x, max_x], [max_y, max_y], color='red', linewidth=2)
        plt.plot([min_x, min_x], [min_y, max_y], color='red', linewidth=2)
        plt.plot([max_x, max_x], [min_y, max_y], color='red', linewidth=2)
    plt.axis('off')
    plt.show()

    # Display the cropped segments
    for i, cropped_img in enumerate(cropped_images):
        plt.figure(figsize=(5, 5))
        plt.imshow(cropped_img)
        plt.title(f"Cropped Segment {i+1}")
        plt.axis('off')
        plt.show()

# Function to display the original image with bounding boxes and cropped segments
def display_segments_with_cluster(image, bounding_boxes, clustered_boxes, cropped_images):
    """
    Display the original image with bounding boxes and each cropped segment.
    """
    # Display the original image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for (min_x, max_x, min_y, max_y) in bounding_boxes:
        plt.plot([min_x, max_x], [min_y, min_y], color='red', linewidth=2)
        plt.plot([min_x, max_x], [max_y, max_y], color='red', linewidth=2)
        plt.plot([min_x, min_x], [min_y, max_y], color='red', linewidth=2)
        plt.plot([max_x, max_x], [min_y, max_y], color='red', linewidth=2)
    plt.axis('off')
    plt.show()

    # Display the cropped segments for each cluster
    for label, cropped_img in enumerate(cropped_images):
        plt.figure(figsize=(5, 5))
        plt.imshow(cropped_img)
        plt.title(f"Cropped Cluster {label+1}")
        plt.axis('off')
        plt.show()
        
# Function to crop the image based on grouped bounding boxes
def crop_by_group(image, clustered_boxes):
    """
    Crop the image by the grouped bounding boxes (one for each cluster).
    """
    cropped_images = []
    
    for label, boxes in clustered_boxes.items():
        # Combine bounding boxes for each group
        min_x = min([box[0] for box in boxes])
        max_x = max([box[1] for box in boxes])
        min_y = min([box[2] for box in boxes])
        max_y = max([box[3] for box in boxes])
        
        # Crop the image using the combined bounding box
        cropped_img = image.crop((min_x, min_y, max_x + 1, max_y + 1))  # +1 to include max_x and max_y
        cropped_images.append(cropped_img)
    
    return cropped_images
# Function to perform K-means clustering on the bounding boxes
def cluster_segments(bounding_boxes, n_clusters=2):
    """
    Perform K-means clustering on the bounding boxes' centroids.
    """
    # Compute the centroids (mid-points of bounding boxes)
    centroids = np.array([[(min_x + max_x) / 2, (min_y + max_y) / 2] for min_x, max_x, min_y, max_y in bounding_boxes])
    
    # Perform K-means clustering on the centroids
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(centroids)
    
    # Return the cluster labels for each segment
    return kmeans.labels_

# Function to group bounding boxes by cluster
def group_bounding_boxes_by_cluster(bounding_boxes, cluster_labels):
    """
    Group bounding boxes by their cluster labels.
    """
    clustered_boxes = {}
    for label, (min_x, max_x, min_y, max_y) in zip(cluster_labels, bounding_boxes):
        if label not in clustered_boxes:
            clustered_boxes[label] = []
        clustered_boxes[label].append((min_x, max_x, min_y, max_y))
    
    return clustered_boxes


#CLUSTER BY MAX LIMIT
# Function to perform K-means clustering on the bounding boxes' centroids with a max limit per cluster
def cluster_segments_with_limit(bounding_boxes, n_clusters=2, max_bboxes_per_cluster=10):
    """
    Perform K-means clustering on the bounding boxes' centroids, ensuring that no cluster contains more than
    a specified number of bounding boxes.

    Args:
    - bounding_boxes: List of bounding boxes as tuples (min_x, max_x, min_y, max_y)
    - n_clusters: Number of initial clusters to form
    - max_bboxes_per_cluster: Maximum number of bounding boxes allowed per cluster
    
    Returns:
    - clustered_boxes: A dictionary where keys are cluster labels and values are lists of bounding boxes
    """
    # Compute the centroids (mid-points of bounding boxes)
    centroids = np.array([[(min_x + max_x) / 2, (min_y + max_y) / 2] for min_x, max_x, min_y, max_y in bounding_boxes])
    
    # Perform initial K-means clustering on the centroids
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(centroids)
    
    # Group bounding boxes by their cluster labels
    clustered_boxes = {}
    for label, (min_x, max_x, min_y, max_y) in zip(kmeans.labels_, bounding_boxes):
        if label not in clustered_boxes:
            clustered_boxes[label] = []
        clustered_boxes[label].append((min_x, max_x, min_y, max_y))

    # Check for clusters that exceed the max limit and split them further if necessary
    final_clustered_boxes = {}
    for label, boxes in clustered_boxes.items():
        # If the cluster has more than the max limit, split it further
        if len(boxes) > max_bboxes_per_cluster:
            print(f"Cluster {label} has {len(boxes)} bounding boxes. Splitting it further.")
            
            # Apply K-means again on the centroids of this cluster
            sub_centroids = np.array([[(min_x + max_x) / 2, (min_y + max_y) / 2] for min_x, max_x, min_y, max_y in boxes])
            sub_kmeans = KMeans(n_clusters=(len(boxes) // max_bboxes_per_cluster) + 1, random_state=42)
            sub_kmeans.fit(sub_centroids)

            # Group the boxes from the original cluster into new sub-clusters
            sub_clustered_boxes = {}
            for sub_label, (min_x, max_x, min_y, max_y) in zip(sub_kmeans.labels_, boxes):
                if sub_label not in sub_clustered_boxes:
                    sub_clustered_boxes[sub_label] = []
                sub_clustered_boxes[sub_label].append((min_x, max_x, min_y, max_y))
            
            # Add the sub-clusters to the final clustered_boxes
            for sub_label, sub_boxes in sub_clustered_boxes.items():
                final_clustered_boxes[f"{label}_{sub_label}"] = sub_boxes
        else:
            # If the cluster is small enough, keep it as is
            final_clustered_boxes[label] = boxes
    
    return final_clustered_boxes

def visualize_clusters(image, bounding_boxes, clustered_boxes):
    """
    Visualizes two images side by side: 
    1. The first image shows all boundary boxes (before clustering).
    2. The second image shows only the clustered boxes, each in a different color.
    
    Args:
    - image (PIL.Image): The original image on which the bounding boxes will be drawn.
    - bounding_boxes (list): List of bounding boxes as (min_x, max_x, min_y, max_y).
    - clustered_boxes (dict): A dictionary where keys are cluster labels and values are lists of bounding boxes in each cluster.
    
    Displays both images side by side.
    """
    # Generate random colors for each cluster
    cluster_colors = {label: (random.random(), random.random(), random.random()) for label in clustered_boxes.keys()}
    
    print(f"Cluster colors: {cluster_colors}")  # Debugging step

    # Create two subplots to display the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot 1: Image with all boundary boxes (before clustering)
    ax = axes[0]
    ax.imshow(image)
    
    # Color for boundary boxes (before clustering)
    boundary_color = (0.0, 0.0, 0.0)  # Black for boundary boxes

    # Draw boundary boxes in black
    for (min_x, max_x, min_y, max_y) in bounding_boxes:
        ax.plot([min_x, max_x], [min_y, min_y], color=boundary_color, linewidth=2)
        ax.plot([min_x, max_x], [max_y, max_y], color=boundary_color, linewidth=2)
        ax.plot([min_x, min_x], [min_y, max_y], color=boundary_color, linewidth=2)
        ax.plot([max_x, max_x], [min_y, max_y], color=boundary_color, linewidth=2)

    ax.axis('off')  # Hide axis

    # Plot 2: Image with only clustered boxes
    ax = axes[1]
    ax.imshow(image)

    # Draw the bounding boxes for each cluster
    for label, boxes in clustered_boxes.items():
        cluster_color = cluster_colors[label]  # Get the color for this cluster
        
        # Draw the bounding boxes for the current cluster with the cluster color
        for (min_x, max_x, min_y, max_y) in boxes:
            ax.plot([min_x, max_x], [min_y, min_y], color=cluster_color, linewidth=3)
            ax.plot([min_x, max_x], [max_y, max_y], color=cluster_color, linewidth=3)
            ax.plot([min_x, min_x], [min_y, max_y], color=cluster_color, linewidth=3)
            ax.plot([max_x, max_x], [min_y, max_y], color=cluster_color, linewidth=3)

    ax.axis('off')  # Hide axis

    # Show both images side by side
    plt.show()



# Function to crop the image based on grouped bounding boxes (no cluster has more than 10 bounding boxes)
def crop_by_group_with_limit(image, clustered_boxes):
    """
    Crop the image by the grouped bounding boxes (one for each cluster), ensuring no cluster exceeds the max limit.

    Args:
    - image: PIL Image object of the original image
    - clustered_boxes: Dictionary of clustered bounding boxes (label -> list of bounding boxes)
    
    Returns:
    - cropped_images: List of cropped image segments based on bounding boxes
    """
    cropped_images = []
    
    for label, boxes in clustered_boxes.items():
        # Combine bounding boxes for each group (cluster)
        min_x = min([box[0] for box in boxes])
        max_x = max([box[1] for box in boxes])
        min_y = min([box[2] for box in boxes])
        max_y = max([box[3] for box in boxes])
        
        # Crop the image using the combined bounding box
        cropped_img = image.crop((min_x, min_y, max_x + 1, max_y + 1))  # +1 to include max_x and max_y
        cropped_images.append(cropped_img)
    
    return cropped_images
    

def display_image(image):
    plt.imshow(image)
    plt.axis('off')  # Turn off axis
    plt.show()

# Function to process each image in the folder
def process_images_in_folder(folder_path,model,processor):
    """
    Process each image in the specified folder by asking questions and getting responses.
    
    Args:
    - folder_path (str): Path to the folder containing the images.
    """
    # List all files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process each image in the folder
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing image: {image_path}")
        
        # Load the image
        image = Image.open(image_path)
        
        # Display the image (optional)
        display_image(image)
        
        # Start continuous loop for asking questions
        while True:
            # Ask for user input (question)
            question = input(f"Enter your question for the image '{image_file}' (or type 'q' to quit): ")

            if question.lower() == 'q':
                print("Exiting...")
                break

            # Combine the image and the question prompt
            prompt = f"[INST] <image>\n{question} [/INST]"

            # Process inputs
            inputs = processor(prompt, image, return_tensors="pt").to(device)

            # Generate output from model
            output = model.generate(**inputs, max_new_tokens=500)

            # Decode and print the model's response
            response = processor.decode(output[0], skip_special_tokens=True)
            print(f"Response for {image_file}: {response}")

            # Clear GPU memory after each iteration (to save memory)
            torch.cuda.empty_cache()
            gc.collect()  # Collect Python garbage to release unused memory


import os
import torch
from PIL import Image
import gc

def get_summary_from_images(folder_path, model, processor):
    """
    Process each image in the specified folder by asking the model to explain the image and 
    collect the responses in a list.
    
    Args:
    - folder_path (str): Path to the folder containing the images.
    - model: The model used for generating responses.
    - processor: The processor used to prepare inputs for the model.
    - device: The device (CPU or GPU) to run the model on.
    
    Returns:
    - List of responses (explanations) for each image.
    """
    # List all files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # List to store all image explanations
    image_explanations = []

    # Process each image in the folder
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing image: {image_path}")
        
        # Load the image
        image = Image.open(image_path)
        
        # Display the image (optional)
        # display_image(image)  # Optional if you want to visually display it
        
        # Ask the question "explain the image"
        question = "explain the image"

        # Combine the image and the question prompt
        prompt = f"[INST] <image>\n{question} [/INST]"

        # Process inputs
        inputs = processor(prompt, image, return_tensors="pt").to(device)

        # Generate output from model
        output = model.generate(**inputs, max_new_tokens=500)

        # Decode the model's response
        full_response = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract only the explanation part after [INST] and [/INST]
        start_idx = full_response.find("[INST]  \nexplain the image [/INST]") + len("[INST]  \nexplain the image [/INST]")
        explanation = full_response[start_idx:].strip()

        #print(f"Explanation for {image_file}: {explanation}")

        # Add the explanation to the list
        image_explanations.append(explanation)

        # Clear GPU memory after each iteration (to save memory)
        torch.cuda.empty_cache()
        gc.collect()  # Collect Python garbage to release unused memory

    # Return the list of image explanations
    return image_explanations
