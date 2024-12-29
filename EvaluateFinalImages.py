import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import os
import csv



def calculate_psnr(original, Enhanced):
  
    # Ensure the images have the same dimensions
    if original.shape != Enhanced.shape:
        raise ValueError("Input images must have the same dimensions")

    # Calculate the mean squared error (MSE) for each channel
    mse = np.mean((original - Enhanced) ** 2)
    
    # If the MSE is zero, the images are identical
    if mse == 0:
        return float("inf")

    # PSNR calculation
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr,mse

def calculate_snr(original, Enhanced):
   
    # Ensure the images have the same dimensions
    if original.shape != Enhanced.shape:
        raise ValueError("Input images must have the same dimensions")

    # Calculate the power of the signal (original image)
    signal_power = np.mean(original ** 2)

    # Calculate the power of the noise (difference between original and Enhanced image)
    noise = original - Enhanced
    noise_power = np.mean(noise ** 2)

    # Avoid division by zero if noise power is zero
    if noise_power == 0:
        return float("inf")

    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power)

    return snr

def calculate_ssim(original, Enhanced):
   
    image1  = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    image2 = cv2.cvtColor(Enhanced, cv2.COLOR_BGR2GRAY)
    
    # Check if images are read properly
    if image1 is None or image2 is None:
        raise ValueError("One or both image paths are invalid or images could not be read.")

    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Calculate SSIM
    ssim, _ = compare_ssim(image1, image2, full=True)
    return ssim


def evaluateImages(EnahncedReconstructed_path, original_path):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Name", "ssim","PSNR","mse","SNR"])
        for file_name in os.listdir(EnahncedReconstructed_path):
            Enhanced_img_path = os.path.join(EnahncedReconstructed_path, file_name)
            original_img_path = os.path.join(original_path, file_name)
            Enhanced_img = cv2.imread(Enhanced_img_path)
            original_img = cv2.imread(original_img_path)
            ssim_value = calculate_ssim(original_img, Enhanced_img)
            psnr_value, mse = calculate_psnr(original_img, Enhanced_img)
            snr_value = calculate_snr(original_img, Enhanced_img)
            writer.writerow([file_name, ssim_value,psnr_value,mse,snr_value])

csv_file = "./Results.csv"
evaluateImages('./assets/EnahncedReconstructed','./assets/original')