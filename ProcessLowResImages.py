import os
import cv2
import dlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from models.LoadModels import SaimeseLeft_eyebrowModel
from models.LoadModels import SaimeseLeft_eyeModel
from models.LoadModels import SaimeseMouthModel
from models.LoadModels import SaimeseNoseModel
from models.LoadModels import SaimeseRight_eyebrowModel
from models.LoadModels import SaimeseRight_eyeModel
from models.LoadModels import SaimeseChinModel


left_eye_model = SaimeseLeft_eyeModel.load_left_eye_model()
right_eye_model = SaimeseRight_eyeModel.load_right_eye_model()
left_eyebrow_model = SaimeseLeft_eyebrowModel.load_left_eyebrow_model()
right_eyebrow_model = SaimeseRight_eyebrowModel.load_right_eyebrow_model()
nose_model = SaimeseNoseModel.load_nose_model()
mouth_model = SaimeseMouthModel.load_mouth_model()
chin_model = SaimeseChinModel.load_Chin_model()


def flatAndPadImage(img):
     flattened = img.flatten()
     padding_size = 150000 - len(flattened)
     padded = np.pad(flattened, (0, padding_size), mode='constant', constant_values=0)
     return padded

def getPart(image, landmarks):
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
    box =  (x,y,w,h)
    return region, box
 
def getBestMatchImage(OriginalExtract_path, part_name,ancorImg, PartModel):
    
    bestSimilarity = 0.0
    fileName = ""

    orginalPrat_Path = os.path.join(OriginalExtract_path, part_name)
    files = [f for f in os.listdir(orginalPrat_Path)]
    for i in range(0,len(files),2):
        if i+1 >= len(files):
            break

        img1_path = os.path.join(orginalPrat_Path, files[i])
        img2_path = os.path.join(orginalPrat_Path, files[i+1])
        posImg = cv2.imread(img1_path)
        negImg = cv2.imread(img2_path)
        anchors = []
        positives = []
        negatives = []
        anchors.append(flatAndPadImage(ancorImg))
        positives.append(flatAndPadImage(posImg))
        negatives.append(flatAndPadImage(negImg))
        anchors =  np.array(anchors)/255
        positives =  np.array(positives)/255
        negatives =  np.array(negatives)/255
        emb_dim = 512
        result = PartModel.predict([anchors, positives, negatives])
        

        emb_anc = result[0, :emb_dim]       # First 512 values for anchor
        emb_pos = result[0, emb_dim:2*emb_dim]  # Next 512 values for positive
        emb_neg = result[0, 2*emb_dim:]

        # Compute similarity between anchor and positive embeddings
        similarity_anc_pos = cosine_similarity([emb_anc], [emb_pos])
        similarity_anc_neg = cosine_similarity([emb_anc], [emb_neg])
        if similarity_anc_pos > bestSimilarity:
            bestSimilarity = similarity_anc_pos
            fileName = files[i]
        if similarity_anc_neg > bestSimilarity:
            bestSimilarity = similarity_anc_neg
            fileName = files[i+1]
    return fileName


def extract_face_parts_and_replacments(image,OriginalExtract_path):
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

        partsName = []
        replaceParts = []
        boxes = []
        
        left_eye_img, box1 =  getPart(image, left_eye)
        left_eye_img_org_name = getBestMatchImage(OriginalExtract_path,'left_eye',left_eye_img,left_eye_model)
        replaceParts.append( left_eye_img_org_name)
        boxes.append(box1)
        partsName.append('left_eye')

        right_eye_img, box2 =  getPart(image, right_eye)
        right_eye_img_org_name = getBestMatchImage(OriginalExtract_path,'right_eye',right_eye_img,right_eye_model)
        replaceParts.append(right_eye_img_org_name)
        boxes.append(box2)
        partsName.append('right_eye')

        left_eyebrow_img, box3 =  getPart(image, left_eyebrow)
        left_eyebrow_img_org_name = getBestMatchImage(OriginalExtract_path,'left_eyebrow',left_eyebrow_img,left_eyebrow_model)
        replaceParts.append(left_eyebrow_img_org_name)
        boxes.append(box3)
        partsName.append('left_eyebrow')
        
        right_eyebrow_img, box4 =  getPart(image, right_eyebrow)
        right_eyebrow_img_org_name = getBestMatchImage(OriginalExtract_path,'right_eyebrow',right_eyebrow_img,right_eyebrow_model)
        replaceParts.append(right_eyebrow_img_org_name)
        boxes.append(box4)
        partsName.append('right_eyebrow')

        nose_img, box5 =  getPart(image, nose)
        nose_img_org_name = getBestMatchImage(OriginalExtract_path,'nose',nose_img,nose_model)
        replaceParts.append(nose_img_org_name)
        boxes.append(box5)
        partsName.append('nose')

        mouth_img, box6 =  getPart(image, mouth)
        mouth_img_org_name = getBestMatchImage(OriginalExtract_path,'mouth',mouth_img,mouth_model)
        replaceParts.append(mouth_img_org_name)
        boxes.append(box6)
        partsName.append('mouth')

        chin_img, box7 =  getPart(image, chin)
        chin_img_org_name = getBestMatchImage(OriginalExtract_path,'chin',chin_img,chin_model)
        replaceParts.append(chin_img_org_name)
        boxes.append(box7)
        partsName.append('chin')


        return (partsName,replaceParts, boxes)



   
    x, y, w, h = box

    # Resize the part image to fit the box dimensions
    resized_part = cv2.resize(part, (w, h))

    # Create a mask for blending (if the part image has transparency)
    if resized_part.shape[-1] == 4:  # RGBA image
        alpha = resized_part[:, :, 3] / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])  # Convert to 3-channel alpha
        resized_part = resized_part[:, :, :3]  # Remove the alpha channel
    else:
        alpha = np.ones((h, w, 3))  # Fully opaque

    # Extract the region of interest (ROI) from the original image
    roi = original[y:y+h, x:x+w]

    # Blend the images using the mask
    blended = (alpha * resized_part + (1 - alpha) * roi).astype(np.uint8)

    # Insert the blended region back into the original image
    result = original.copy()
    result[y:y+h, x:x+w] = blended
    return result

def insert_part(original, part, box):
    
    x, y, w, h = box

    # Resize the part image to fit the box dimensions
    resized_part = cv2.resize(part, (w, h))

    # Create a mask for blending (if the part image has transparency)
    if resized_part.shape[-1] == 4:  # RGBA image
        alpha = resized_part[:, :, 3] / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])  # Convert to 3-channel alpha
        resized_part = resized_part[:, :, :3]  # Remove the alpha channel
    else:
        alpha = np.ones((h, w, 3))  # Fully opaque

    # Extract the region of interest (ROI) from the original image
    roi = original[y:y+h, x:x+w]

    # Blend the images using the mask
    blended = (alpha * resized_part + (1 - alpha) * roi).astype(np.uint8)

    # Update the original image in place
    original[y:y+h, x:x+w] = blended

   
def reconstructLowResImage(image,img_name ,OriginalExtract_path,Reconstructed_Path):
    partNames, replaceParts, boxes =  extract_face_parts_and_replacments(image,OriginalExtract_path)
    selectedPart = None
    counts = Counter(replaceParts)
    for part, count in counts.items():
        if count >= 4:
            selectedPart = part

    for i in range(len(partNames)):
        part_img_path = os.path.join(OriginalExtract_path, partNames[i])
        if selectedPart is not None: 
            part_img_path = os.path.join(part_img_path, selectedPart)
            part = cv2.imread(part_img_path)
            insert_part(image,part,boxes[i])
        else:
            part_img_path = os.path.join(part_img_path, replaceParts[i])
            part = cv2.imread(part_img_path)
            insert_part(image,part,boxes[i])
    Reconstructed_image_path = os.path.join(Reconstructed_Path, img_name)
    cv2.imwrite(Reconstructed_image_path, image)


def preprocessLowResImages(LowFull_path, OriginalExtract_path,Reconstructed_Path):
    for file_name in os.listdir(LowFull_path):
            file_path = os.path.join(LowFull_path, file_name)
            img = cv2.imread(file_path)
            if img is not None:
                reconstructLowResImage(img,file_name,OriginalExtract_path,Reconstructed_Path)
            else:
                print(f"Skipping file {file_name}: Not a valid image or cannot be read.")
       
# Example usage
preprocessLowResImages("./assets/lowFull","./assets/originalExtract","./assets/Reconstructed")


#evaluation
# height = 1024
# width = 1024
# scale_factor = 0.25
# new_width = int(width * scale_factor)
# new_height = int(height * scale_factor)
# file_path = os.path.join("./assets/original/test", "SFHQ_pt1_00000715.jpg")
# img = cv2.imread(file_path)
# if img is not None:
#     low_res = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
#     low_res_upscaled = cv2.resize(low_res, (width, height), interpolation=cv2.INTER_NEAREST)
#     lowFull_path = os.path.join("./assets/lowFull/testLow", "SFHQ_pt1_00000715.jpg")
#     cv2.imwrite(lowFull_path, low_res_upscaled)
# preprocessLowResImages("./assets/lowFull/testLow","./assets/originalExtract","./assets/Reconstructed/testOutput")
