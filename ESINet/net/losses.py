import tensorflow as tf
from tensorflow.keras import backend as K

def weighted_mse_loss(w=1, min_val=1e-3):
    w = tf.cast(w, tf.float32)
    
    def loss(true, pred):

        error = K.square(true - pred)

        error = K.switch(K.less(true, min_val), w * error , error)

        return K.mean(error) 

    return loss

def weighted_huber_loss(w=1.0, delta=1.0, min_val=1e-3):
    '''
    '''
    w = tf.cast(w, tf.float32)
    delta = K.clip(tf.cast(delta, tf.float32), K.epsilon(), 10e2)

    def loss(true, pred):
        differences = true-pred

        error = delta * ( K.sqrt(1 + K.square(differences/delta)) -1 )


        error = K.switch(K.less(true, min_val), w * error , error)

        return K.mean(error)

    return loss

def weighted_mae_loss(w=1, min_val=1e-3):
    w = tf.cast(w, tf.float32)
    
    def loss(true, pred):

        error = K.abs(true - pred)

        error = K.switch(K.less(true, min_val), w * error , error)

        return K.mean(error) 

    return loss