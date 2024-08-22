#!/usr/bin/env python
# coding: utf-8

import os
import torch
import shutil
import random
import numpy as np
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from skimage.color import lab2rgb


# Function to remove grayscale images from the raw dataset
def remove_grayscale(directory):
    # Check the directory for grayscale images
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            file_path = os.path.join(directory, filename)
            image = io.imread(file_path)

            # Check if image is grayscale
            if len(image.shape) == 2 or (
                len(image.shape) == 3 and image.shape[2] == 3 and
                np.all(image[:, :, 0] == image[:, :, 1]) and
                np.all(image[:, :, 1] == image[:, :, 2])
                ):
                os.remove(file_path)
    
    print("## Grayscale images are removed ##")


# Function to resize the images
def resize_images(input_folder, size=(128, 128)):
    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        
        if filename.endswith(".jpg"):
            # Open the image file
            with Image.open(os.path.join(input_folder, filename)) as img:
                # Resize the image
                img_resized = img.resize(size)
                img_resized.save(os.path.join(input_folder, filename))
    print("## Images are resized ##")


# Function to split images (used for train-test split)
def split_files(source, destination, num):

    # Ensure the destination folder exists
    if not os.path.exists(destination):
        os.makedirs(destination)

    # List all files in the source folder
    all_files = os.listdir(source)

    # Randomly select files
    selected_files = random.sample(all_files, num)

    # Move selected files to the destination folder
    for file_name in selected_files:
        source_file_path = os.path.join(source, file_name)
        destination_file_path = os.path.join(destination, file_name)
        shutil.move(source_file_path, destination_file_path)

    # Delete moved files from the source folder
    for file_name in selected_files:
        file_path = os.path.join(source, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
    
    print(f"## {num} files moved successfully. ##")


# Function to show colored images
def show_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    gray, colored = next(iter(loader))

    fig, axes = plt.subplots(2, 5, figsize=(15, 5))

    # Plot colored images on top row
    for i in range(5):
        colored_rgb = lab_to_rgb(gray[i], colored[i])
        axes[0, i].imshow(colored_rgb)
        axes[0, i].axis('off')

    # Plot grayscale images on bottom row
    for i in range(5):
        axes[1, i].imshow(np.transpose(gray[i], (1, 2, 0)), cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


# Function to convert LAB image (L, AB) to RGB image
def lab_to_rgb(gray, ab):
    lab_image = np.zeros((gray.shape[1], gray.shape[2], 3))
    lab_image[:, :, 0] = gray[0]  # L channel
    lab_image[:, :, 1:] = ab.permute(1, 2, 0).numpy()  # AB channels
    rgb_image = (255 * np.clip(lab2rgb(lab_image), 0, 1)).astype(np.uint8)
    return rgb_image


# Function to plot results
def plot_results(L_batch, AB_pred_batch, AB_true_batch, batch_size=5):

    num_plots = min(batch_size, len(L_batch))
    fig, axes = plt.subplots(3, num_plots, figsize=(15, 10))

    for i in range(num_plots):
        L = L_batch[i:i+1]
        AB_pred = AB_pred_batch[i:i+1]
        AB_true = None if AB_true_batch is None else AB_true_batch[i:i+1]

        # Plot grayscale image on the first row
        axes[0, i].imshow(L.squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Grayscale", fontsize=12, pad=5)

        
        # Concatenate L and AB tensors for both predicted and true AB values
        LAB_pred = torch.cat((L, AB_pred), dim=1)
        LAB_true = torch.cat((L, AB_true), dim=1)

        LAB_pred_permuted = LAB_pred.permute(0, 2, 3, 1).cpu().detach().numpy()
        LAB_true_permuted = LAB_true.permute(0, 2, 3, 1).cpu().detach().numpy()

        # Convert LAB to RGB for both predicted and true images
        RGB_pred = lab2rgb(LAB_pred_permuted.squeeze(0))
        RGB_true = lab2rgb(LAB_true_permuted.squeeze(0))

        # Plot the RGB images on the second and third rows
        axes[2, i].imshow(RGB_true)
        axes[2, i].axis('off')
        axes[2, i].set_title("Actual", fontsize=12, pad=5)
        axes[1, i].imshow(RGB_pred)
        axes[1, i].axis('off')
        axes[1, i].set_title("Predicted", fontsize=12, pad=5)

    plt.tight_layout()
    plt.show()


# Function to plot all predicted images for comparison
def plot_comparison(L_batch, AB_pred_batch1, AB_pred_batch2, AB_pred_batch3, AB_true_batch=None, batch_size=5):

    num_plots = min(batch_size, len(L_batch))
    num_models = 3  # Number of models
    fig, axes = plt.subplots(num_models + 1, num_plots, figsize=(15, 10))  # Adjusted the number of rows

    for i in range(num_plots):
        L = L_batch[i:i+1]
        AB_true = None if AB_true_batch is None else AB_true_batch[i:i+1]

        # Plot grayscale image on the first row
        axes[0, i].imshow(L.squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Grayscale", fontsize=12, pad=5)

        # Plot predictions for Model 1 on the second row
        AB_pred1 = AB_pred_batch1[i:i+1]
        LAB_pred1 = torch.cat((L, AB_pred1), dim=1)
        LAB_pred1_permuted = LAB_pred1.permute(0, 2, 3, 1).cpu().detach().numpy()
        RGB_pred1 = lab2rgb(LAB_pred1_permuted.squeeze(0))
        axes[1, i].imshow(RGB_pred1)
        axes[1, i].axis('off')
        axes[1, i].set_title("Model 1 Prediction", fontsize=12, pad=5)

        # Plot predictions for Model 2 on the third row
        AB_pred2 = AB_pred_batch2[i:i+1]
        LAB_pred2 = torch.cat((L, AB_pred2), dim=1)
        LAB_pred2_permuted = LAB_pred2.permute(0, 2, 3, 1).cpu().detach().numpy()
        RGB_pred2 = lab2rgb(LAB_pred2_permuted.squeeze(0))
        axes[2, i].imshow(RGB_pred2)
        axes[2, i].axis('off')
        axes[2, i].set_title("Model 2 Prediction", fontsize=12, pad=5)

        # Plot predictions for Model 3 on the fourth row
        AB_pred3 = AB_pred_batch3[i:i+1]
        LAB_pred3 = torch.cat((L, AB_pred3), dim=1)
        LAB_pred3_permuted = LAB_pred3.permute(0, 2, 3, 1).cpu().detach().numpy()
        RGB_pred3 = lab2rgb(LAB_pred3_permuted.squeeze(0))
        axes[3, i].imshow(RGB_pred3)
        axes[3, i].axis('off')
        axes[3, i].set_title("Model 3 Prediction", fontsize=12, pad=5)

    plt.tight_layout()
    plt.show()


# Fuction to plot losses
def plot_loss(train_losses, epochs):
    x = range(epochs)
    best_epoch = train_losses.index(min(train_losses))
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(x, train_losses, label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    remove_grayscale("mirflickr\\")
    resize_images("mirflickr\\")
    split_files("mirflickr\\", "dataset\\train_gray", "dataset\\train_colored", 5000)
    split_files("mirflickr\\", "dataset\\val_gray", "dataset\\val_colored", 1000)
    split_files("mirflickr\\", "dataset\\test_gray", "dataset\\test_colored", 1000)