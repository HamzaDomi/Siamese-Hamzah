
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2
import random




def get_image_patches(part_name):
    anchors = []
    positives = []
    negatives = []
    originalExtarct_path = os.path.join("./assets/originalExtract", part_name)
    lowRes_path = os.path.join("./assets/lowRes", part_name)
    files = [f for f in os.listdir(originalExtarct_path)]
    for file_name in os.listdir(lowRes_path):
                lowRes_img_path = os.path.join(lowRes_path, file_name)
                originalExtarct_img_path = os.path.join(originalExtarct_path, file_name)
                lowRes_img = cv2.imread(lowRes_img_path)
                originalExtarct_img = cv2.imread(originalExtarct_img_path)
                if lowRes_img is not None:
                    flattened = lowRes_img.flatten()
                    padding_size = 150000 - len(flattened)
                    padded = np.pad(flattened, (0, padding_size), mode='constant', constant_values=0)
                    anchors.append(padded)
                    flattened = originalExtarct_img.flatten()
                    padding_size = 150000 - len(flattened)
                    padded = np.pad(flattened, (0, padding_size), mode='constant', constant_values=0)
                    positives.append(padded)
                    random_file = random.choice(files)
                    while(file_name == random_file):
                            random_file = random.choice(files)
                    random_img_path = os.path.join(originalExtarct_path, random_file)
                    originalExtarct_random_img =cv2.imread(random_img_path)
                    flattened = originalExtarct_random_img.flatten()
                    padding_size = 150000 - len(flattened)
                    padded = np.pad(flattened, (0, padding_size), mode='constant', constant_values=0)
                    negatives.append(padded)
                    
      
    anchors =  np.array(anchors)/255
    positives =  np.array(positives)/255
    negatives =  np.array(negatives)/255
    return (anchors,positives,negatives)


def triplet_loss(alpha, emb_dim):
    def loss(y_true, y_pred):
        anc, pos, neg = y_pred[:, :emb_dim], y_pred[:, emb_dim:emb_dim*2], y_pred[:, 2*emb_dim:]
        dp = tf.reduce_mean(tf.square(anc - pos), axis=1)
        dn = tf.reduce_mean(tf.square(anc - neg), axis=1)
        return tf.nn.relu(dp - dn + alpha)  
    return loss

def create_tf_dataset(batch_size, emb_dim,part_name):
    def generator():
        while True:
            anchors, positives, negatives = get_image_patches(part_name = part_name)
            x = (anchors, positives, negatives)
            y = np.zeros((batch_size, 3 * emb_dim))
            yield x, y
    output_signature = (
        (
            tf.TensorSpec(shape=(batch_size, 150000), dtype=tf.float32),  # Anchors
            tf.TensorSpec(shape=(batch_size, 150000), dtype=tf.float32),  # Positives
            tf.TensorSpec(shape=(batch_size, 150000), dtype=tf.float32),  # Negatives
        ),
        tf.TensorSpec(shape=(batch_size, 3 * emb_dim), dtype=tf.float32),  # Targets
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

def SiameseModelTrain(part_name):
     
    emb_dim = 512
    alpha = 0.2
    batch_size = 70
    epochs = 3
    steps_per_epoch = int(60000/batch_size)

    embedding_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu',input_shape=(150000,)),
        tf.keras.layers.Dense(emb_dim, activation='sigmoid'),
        
    ])

    inp_anc = tf.keras.layers.Input(shape=(150000,))
    inp_pos = tf.keras.layers.Input(shape=(150000,))
    inp_neg = tf.keras.layers.Input(shape=(150000,))

    emb_anc = embedding_model(inp_anc)
    emb_pos = embedding_model(inp_pos)
    emb_neg = embedding_model(inp_neg)

    outp = tf.keras.layers.concatenate([emb_anc, emb_pos, emb_neg], axis=1) 

    net = tf.keras.models.Model(
        [inp_anc, inp_pos, inp_neg], 
        outp
    )

    net.compile(loss=triplet_loss(alpha=0.2, emb_dim=emb_dim), optimizer='adam')

    dataset = create_tf_dataset(batch_size, emb_dim, part_name)
    _ = net.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        # Pass test data for visualization
    )

    net.save_weights( "./models/weights/" + part_name + '.weights.h5')


SiameseModelTrain("chin")
SiameseModelTrain("left_eye")
SiameseModelTrain("right_eye")
SiameseModelTrain("left_eyebrow")
SiameseModelTrain("right_eyebrow")
SiameseModelTrain("nose")
SiameseModelTrain("mouth")