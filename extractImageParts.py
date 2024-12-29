import os
import cv2
import dlib
import numpy as np





def preprocessImages(folder_path, output_folder_lowFull ,output_folder_original,output_folder_low):
    height = 1024
    width = 1024
    scale_factor = 0.25
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            img = cv2.imread(file_path)
            if img is not None:
                low_res = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                low_res_upscaled = cv2.resize(low_res, (width, height), interpolation=cv2.INTER_NEAREST)
                lowFull_path = os.path.join(output_folder_lowFull, file_name)
                cv2.imwrite(lowFull_path, low_res_upscaled)
                extract_face_parts(img,file_name,output_folder_original)
                extract_face_parts(low_res_upscaled,file_name,output_folder_low)
            else:
                print(f"Skipping file {file_name}: Not a valid image or cannot be read.")
       
def extract_face_parts(image,img_name ,output_folder):
    # Load the image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face detector from dlib
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    # Load the facial landmarks predictor
    predictor = dlib.shape_predictor(r"./models/shape_predictor_68_face_landmarks.dat")

    for face in faces:
        landmarks = predictor(gray, face)

        # Define the regions of interest based on facial landmarks
        left_eye = landmarks.parts()[36:42]
        right_eye = landmarks.parts()[42:48]
        left_eyebrow = landmarks.parts()[17:22]
        right_eyebrow = landmarks.parts()[22:27]
        nose = landmarks.parts()[27:36]
        mouth = landmarks.parts()[48:68]
        chin = landmarks.parts()[6:11]  # Select a subset of chin landmarks

        # Save the regions of interest into separate class folders
        save_cutout(image, left_eye, os.path.join(output_folder, "left_eye", img_name))
        save_cutout(image, right_eye, os.path.join(output_folder, "right_eye", img_name))
        save_cutout(image, left_eyebrow, os.path.join(output_folder, "left_eyebrow", img_name))
        save_cutout(image, right_eyebrow, os.path.join(output_folder, "right_eyebrow", img_name))
        save_cutout(image, nose, os.path.join(output_folder, "nose", img_name))
        save_cutout(image, mouth, os.path.join(output_folder, "mouth", img_name))
        save_cutout(image, chin, os.path.join(output_folder, "chin", img_name))

def save_cutout(image, landmarks, filename):
    if len(landmarks) < 2:
        return

    # Convert the landmarks to a numpy array
    landmarks_np = np.array([[p.x, p.y] for p in landmarks])

    # Get the bounding box of the region
    x, y, w, h = cv2.boundingRect(landmarks_np)

    # Ensure the bounding box does not exceed image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(image.shape[1] - x, w)
    h = min(image.shape[0] - y, h)

    # Crop the region from the image
    region = image[y:y+h, x:x+w]

    # Save the region to a new file
    cv2.imwrite(filename, region)





# Example usage
preprocessImages("./assets/original","./assets/lowFull","./assets/originalExtract","./assets/lowRes")
