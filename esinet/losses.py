import tensorflow as tf
from tensorflow.keras import backend as K


def combi(y_true, y_pred):
    error_1 = tf.keras.losses.CosineSimilarity()(y_true, y_pred)
    error_2 = tf.keras.losses.MeanSquaredError() (y_true, y_pred)
    return error_1 + error_2

def reg_loss(reg=0.1):
    reg = tf.cast(reg, tf.float32)
    def mse(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred) + K.mean(K.abs(y_pred))*reg
    return mse

def nmse_loss(reg=0.05):
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
    reg = tf.cast(reg, tf.float32)

    def loss(true, pred):
        # Scale to max(abs(x)) == 1
        pred = scale_mat(pred)
        true = scale_mat(true)
        
        # Calc squared error
        error = K.square(true - pred)
        

        return K.mean(error) #+ K.mean(K.abs(pred))*reg

    return loss

def weighted_mse_loss(weight=1, min_val=1e-3, scale=True):
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

        if scale:
            # Scale to max(abs(x)) == 1
            pred = scale_mat(pred)
            true = scale_mat(true)
        
        # Calc squared error
        error = K.square(true - pred)
        
        # False-positive weighting
        error = K.switch(K.less(K.abs(true), min_val), weight * error , error)

        return K.mean(error) 

    return loss

def weighted_huber_loss(weight=1.0, delta=1.0, min_val=1e-3, scale=True):
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

        if scale:
            # Scale to max(abs(x)) == 1
            pred = scale_mat(pred)
            true = scale_mat(true)
        # Calc error
        differences = true-pred
        
        # Huber Loss
        error = delta * ( K.sqrt(1 + K.square(differences/delta)) -1 )

        # False-positive weighting
        error = K.switch(K.less(K.abs(true), min_val), weight * error , error)

        return K.mean(error)

    return loss

def weighted_mae_loss(w=1, min_val=1e-3, scale=True):
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
        
        if scale:
            # Scale to max(abs(x)) == 1
            pred = scale_mat(pred)
            true = scale_mat(true)
        
        # MAE Loss
        error = K.abs(true - pred)
        
        # False-positive weighting
        error = K.switch(K.less(K.abs(true), min_val), w * error , error)

        return K.mean(error) 

    return loss

def scale_mat(mat):
    ''' Scale matrix such that each row has max value of 1'''
    max_vals = tf.expand_dims(K.max(K.abs(mat), axis=-1), axis=-1)
    max_vals = K.clip(max_vals, K.epsilon(), 999999999999)
    return mat / max_vals


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

def cdist(A, B):
    """
    Computes the pairwise Euclidean distance matrix between two tensorflow matrices A & B, similiar to scikit-learn cdist.
    For example:
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2],
         [3, 4]]
    should return:
        [[0, 2.82],
         [2.82, 0]]
    :param A: m_a x n matrix
    :param B: m_b x n matrix
    :return: euclidean distance matrix (m_a x m_b)
    """
    # squared norms of each row in A and B
    na = K.sum(K.square(A), axis=1)
    nb = K.sum(K.square(B), axis=1)

    # na as a row and nb as a column vectors
    na = tf.expand_dims(na, axis=1)
    nb = tf.expand_dims(nb, axis=0)
    
    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D


# def weighted_hausdorff_distance(pos, alpha=4, thresh=0.1):
#     ''' https://github.com/N0vel/weighted-hausdorff-distance-tensorflow-keras-loss/blob/master/weighted_hausdorff_loss.py
#     from https://arxiv.org/pdf/2010.12876.pdf'''
#     max_dist = 300

#     def hausdorff_loss(y_true, y_pred):
#         y_true = y_true / K.clip(K.max(y_true), K.epsilon(), None)
#         y_pred = y_pred/ K.clip(K.max(y_pred), K.epsilon(), None)

#         eps = 1e-6
#         gt_points = K.squeeze(tf.gather(pos, tf.where(K.abs(y_true)>thresh*K.max(K.abs(y_true))) ), axis=1)
#         num_gt_points = tf.shape(gt_points)[0]

#         p = y_pred

#         p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))

#         d_matrix = cdist(pos, gt_points)

#         num_est_pts = tf.reduce_sum(p)
#         term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

#         d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
#         d_div_p = K.clip(d_div_p, 0, max_dist)
#         term_2 = K.mean(d_div_p, axis=0)

#         return term_1 + term_2

#     return hausdorff_loss
def tf_nanmean(x):
    return tf.reduce_mean(tf.boolean_mask(K.flatten(x), tf.math.is_nan(K.flatten(x))))

def weighted_hausdorff_distance(pos, alpha=4, thresh=0.1):
    ''' https://github.com/N0vel/weighted-hausdorff-distance-tensorflow-keras-loss/blob/master/weighted_hausdorff_loss.py
    from https://arxiv.org/pdf/2010.12876.pdf'''
    max_dist = 300

    def hausdorff_loss(y_true, y_pred):
        def loss_time(y_true_sample, y_pred_sample):
            def loss(y_true_slice, y_pred_sclice):
                y_true_slice = K.abs(y_true_slice)
                y_pred_sclice = K.abs(y_pred_sclice)

                y_true_slice = y_true_slice / K.clip(K.max(K.abs(y_true_slice)), K.epsilon(), None)
                y_pred_sclice = y_pred_sclice/ K.clip(K.max(K.abs(y_pred_sclice)), K.epsilon(), None)

                eps = 1e-6
                gt_points = K.squeeze(tf.gather(pos, tf.where(K.abs(y_true_slice)>thresh*K.max(K.abs(y_true_slice))) ), axis=1)
                num_gt_points = tf.shape(gt_points)[0]

                p = y_pred_sclice

                p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))

                d_matrix = cdist(pos, gt_points)

                num_est_pts = tf.reduce_sum(p)
                term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

                d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
                d_div_p = K.clip(d_div_p, 0, max_dist)
                term_2 = K.mean(d_div_p, axis=0)

                result = K.switch(K.sum(K.abs(y_true_slice))>0, term_1+term_2, tf.convert_to_tensor(0.0, dtype=tf.float32))
                return result

            fun = lambda x: loss(x[0], x[1])
            batched_losses_time = tf.map_fn(fun,
                                (y_true_sample, y_pred_sample),
                                   dtype=tf.float32)
            return K.mean(tf.stack(batched_losses_time))

        fun = lambda x: loss_time(x[0], x[1])
        batched_losses = tf.map_fn(fun,
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss

# for a, b in zip(y_true, y_pred):
#     for a_slice, b_sclice in zip(a,b):
#         fun(a_sclice, b_sclice)