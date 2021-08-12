import tensorflow as tf
from tensorflow.keras import backend as K

def weighted_mse_loss(weight=1, min_val=1e-3):
    ''' Weighted mean squared error (MSE) loss. A loss function that can be 
    used with tensorflow/keras which calculates the MSE with a weighting of 
    false positive predicitons. Set weight high for more conservative 
    predictions.

    Parameters
    ----------
    weight : float
        Weighting factor which penalizes false positives.
    min_val : float
        The threshold below which the target is set to zero.

    Return
    ------
    loss : loss function

    '''
    weight = tf.cast(weight, tf.float32)
    
    def loss(true, pred):

        error = K.square(true - pred)

        error = K.switch(K.less(true, min_val), weight * error , error)

        return K.mean(error) 

    return loss

def weighted_huber_loss(weight=1.0, delta=1.0, min_val=1e-3):
    ''' Weighted Huber loss. A loss function that can be 
    used with tensorflow/keras which calculates the Huber loss with 
    a weighting of false positive predicitons. Set weight high 
    for more conservative predictions.

    Parameters
    ----------
    weight : float
        Weighting factor which penalizes false positives.
    delta : float
        The delta parameter of the Huber loss. Must be non-negative.
    min_val : float
        The threshold below which the target is set to zero.

    Return
    ------
    loss : loss function
    '''
    weight = tf.cast(weight, tf.float32)
    delta = K.clip(tf.cast(delta, tf.float32), K.epsilon(), 10e2)

    def loss(true, pred):
        differences = true-pred

        error = delta * ( K.sqrt(1 + K.square(differences/delta)) -1 )


        error = K.switch(K.less(true, min_val), weight * error , error)

        return K.mean(error)

    return loss

def weighted_mae_loss(w=1, min_val=1e-3):
    ''' Weighted mean absolute error (MAE) loss. A loss function that can be 
    used with tensorflow/keras which calculates the MAE with a weighting of 
    false positive predicitons. Set weight high for more conservative 
    predictions.

    Parameters
    ----------
    weight : float
        Weighting factor which penalizes false positives.
    min_val : float
        The threshold below which the target is set to zero.

    Return
    ------
    loss : loss function
    '''
    w = tf.cast(w, tf.float32)
    
    def loss(true, pred):

        error = K.abs(true - pred)

        error = K.switch(K.less(true, min_val), w * error , error)

        return K.mean(error) 

    return loss


def custom_loss(leadfield, fwd_scaler):
    def loss_batch(y_true, y_pred):
        def losss(y_true, y_pred):
            eeg_fwd = tf.matmul(leadfield, tf.expand_dims(y_pred, axis=1))
            eeg_true = tf.matmul(leadfield, tf.expand_dims(y_true, axis=1))
            eeg_fwd_scaled = eeg_fwd / K.max(K.abs(eeg_fwd))
            eeg_true_scaled = eeg_true / K.max(K.abs(eeg_true))
            
            error_mse = K.mean(K.square(y_true - y_pred))
            error_fwd = K.mean(K.square(eeg_fwd_scaled-eeg_true_scaled))
            error = error_mse + error_fwd * fwd_scaler
            return error

        batched_losses = tf.map_fn(lambda x:
                                    losss(x[0], x[1]),
                                    (y_true, y_pred),
                                    dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))  
    return loss_batch