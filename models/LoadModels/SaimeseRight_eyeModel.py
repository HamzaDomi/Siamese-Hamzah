import tensorflow as tf



def triplet_loss(alpha, emb_dim):
    def loss(y_true, y_pred):
        anc, pos, neg = y_pred[:, :emb_dim], y_pred[:, emb_dim:emb_dim*2], y_pred[:, 2*emb_dim:]
        dp = tf.reduce_mean(tf.square(anc - pos), axis=1)
        dn = tf.reduce_mean(tf.square(anc - neg), axis=1)
        return tf.nn.relu(dp - dn + alpha)  
    return loss


def load_right_eye_model():
    embedding_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu',input_shape=(150000,)),
            tf.keras.layers.Dense(512, activation='sigmoid')
        ])

    inp_anc = tf.keras.layers.Input(shape=(150000,))
    inp_pos = tf.keras.layers.Input(shape=(150000,))
    inp_neg = tf.keras.layers.Input(shape=(150000,))

    emb_anc = embedding_model(inp_anc)
    emb_pos = embedding_model(inp_pos)
    emb_neg = embedding_model(inp_neg)

    outp = tf.keras.layers.concatenate([emb_anc, emb_pos, emb_neg], axis=1) # column wise that's why

    Model = tf.keras.models.Model(
        [inp_anc, inp_pos, inp_neg], 
        outp
    )

    Model.compile(loss=triplet_loss(alpha=0.2, emb_dim=512), optimizer='adam')
    Model.load_weights('././models/weights/right_eye.weights.h5')
    return Model


