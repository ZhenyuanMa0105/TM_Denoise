import tensorflow as tf
import numpy as np
from network.pconv_layer import PConv2D
import tensorflow_probability as tfp
import math


def get_weight(shape, gain=np.sqrt(2)):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)
    w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
    return w


def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])


def Pconv2d_bias(x, fmaps, kernel, mask_in=None):
    assert kernel >= 1 and kernel % 2 == 1
    x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "SYMMETRIC")
    mask_in = tf.pad(mask_in, [[0, 0], [0, 0], [1, 1], [1, 1]], "CONSTANT", constant_values=1)
    conv, mask = PConv2D(fmaps, kernel, strides=1, padding='valid',
                         data_format='channels_first')([x, mask_in])
    return conv, mask


def conv2d_bias(x, fmaps, kernel, gain=np.sqrt(2)):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain)
    w = tf.cast(w, x.dtype)
    x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "SYMMETRIC")
    return apply_bias(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW'))


def Pmaxpool2d(x, k=2, mask_in=None):
    ksize = [1, 1, k, k]
    x = tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')
    mask_out = tf.nn.max_pool(mask_in, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')
    return x, mask_out


def maxpool2d(x, k=2):
    ksize = [1, 1, k, k]
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x


def conv_lr(name, x, fmaps, p=0.7):
    with tf.variable_scope(name):
        x = tf.nn.dropout(x, p)
        return tf.nn.leaky_relu(conv2d_bias(x, fmaps, 3), alpha=0.1)


def conv(name, x, fmaps, p):
    with tf.variable_scope(name):
        x = tf.nn.dropout(x, p)
        return tf.nn.sigmoid(conv2d_bias(x, fmaps, 3, gain=1.0))


def Pconv_lr(name, x, fmaps, mask_in):
    with tf.variable_scope(name):
        x_out, mask_out = Pconv2d_bias(x, fmaps, 3, mask_in=mask_in)
        return tf.nn.leaky_relu(x_out, alpha=0.1), mask_out


def partial_conv_unet(x, mask, channel=3, width=256, height=256, p=0.7, **_kwargs):
    x.set_shape([None, channel, height, width])
    mask.set_shape([None, channel, height, width])
    skips = [x]

    n = x
    n, mask = Pconv_lr('enc_conv0', n, 48, mask_in=mask)
    n, mask = Pconv_lr('enc_conv1', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv2', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv3', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv4', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv5', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    n, mask = Pconv_lr('enc_conv6', n, 48, mask_in=mask)

    # -----------------------------------------------
    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv5', n, 96, p=p)
    n = conv_lr('dec_conv5b', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv4', n, 96, p=p)
    n = conv_lr('dec_conv4b', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv3', n, 96, p=p)
    n = conv_lr('dec_conv3b', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv2', n, 96, p=p)
    n = conv_lr('dec_conv2b', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv1a', n, 64, p=p)
    n = conv_lr('dec_conv1b', n, 32, p=p)
    n = conv('dec_conv1', n, channel, p=p)

    return n


def concat(x, y):
    bs1, c1, h1, w1 = x.shape.as_list()
    bs2, c2, h2, w2 = y.shape.as_list()
    x = tf.image.crop_to_bounding_box(tf.transpose(x, [0, 2, 3, 1]), 0, 0, min(h1, h2), min(w1, w2))
    y = tf.image.crop_to_bounding_box(tf.transpose(y, [0, 2, 3, 1]), 0, 0, min(h1, h2), min(w1, w2))
    return tf.transpose(tf.concat([x, y], axis=3), [0, 3, 1, 2])


def build_denoising_unet(noisy, mu_matrix, p=0.7, is_realnoisy=False, a=0.00001, Threshold=200000):
    _, h, w, c = np.shape(noisy)
    noisy_tensor = tf.identity(noisy)
    noisy_tensor_flat = tf.reshape(noisy_tensor, [-1])
    factor = tf.reduce_max(noisy_tensor_flat) / 255.0
    is_flip_lr = tf.placeholder(tf.int16, name='is_flip_lr')
    is_flip_ud = tf.placeholder(tf.int16, name='is_flip_ud')
    step_placeholder = tf.placeholder(tf.int64, name='step_placeholder')
    noisy_tensor = data_arg(noisy_tensor, is_flip_lr, is_flip_ud)
    response = tf.transpose(noisy_tensor, [0, 3, 1, 2])
    mask_tensor = tf.ones_like(response)
    mask_tensor = tf.nn.dropout(mask_tensor, 0.7) * 0.7
    response = tf.multiply(mask_tensor, response)
    slice_avg = tf.get_variable('slice_avg', shape=[_, h, w, c], initializer=tf.initializers.zeros())
    if is_realnoisy:
        response = tf.squeeze(tf.random_poisson(25 * response, [1]) / 25, 0)

    response = partial_conv_unet(response, mask_tensor, channel=c, width=w, height=h, p=p)
    response = tf.transpose(response, [0, 2, 3, 1])
    mask_tensor = tf.transpose(mask_tensor, [0, 2, 3, 1])

    original_loss = mask_loss(response, noisy_tensor, 1. - mask_tensor)
    data_loss = original_loss
    #original_loss = original_loss + 0.0001
    response = data_arg(response, is_flip_lr, is_flip_ud)
    print("a:", a)
    print("mask_tensor shape:", mask_tensor.shape)
    masked_threshold = 0.1
    masked_indices = tf.where(mask_tensor < masked_threshold)
    print("masked_indices:", masked_indices)
    print("mask_tensor shape:", masked_indices.shape)
    '''
    #response = np.squeeze(np.uint8(np.clip(response, 0, 1) * 255))
    # Compute the conditional probabilities as an additional loss term
    def compute_cond_prob():
        return compute_conditional_probabilities(response, mask_tensor, mu_matrix)

    # 定义不计算概率的函数
    def do_nothing():
        return tf.constant(0.0, dtype=tf.float32)

    # 使用 tf.cond 根据步数条件来决定是否计算概率
    # conditional_prob = tf.Variable(0.0, dtype=tf.float32, name='conditional_prob')
    
    conditional_prob = tf.cond(
        data_loss < 0.001
        # step_placeholder > Threshold,
        compute_cond_prob,  # 如果步数超过阈值，计算概率
        do_nothing         # 否则什么都不做
    )
    '''
    conditional_prob = tf.Variable(0.0, dtype=tf.float32, name='conditional_prob')

    def compute_conditional_actions():
        prob = compute_conditional_probabilities(response, mask_tensor, mu_matrix, factor)
        #tf.debugging.check_numerics(data_loss, "data_loss contains NaN or Inf before update")
        updated_data_loss = tf.subtract(original_loss, tf.math.multiply(0.001, prob))
        #tf.debugging.check_numerics(updated_data_loss, "updated_data_loss contains NaN or Inf after update")

        #tf.print(updated_data_loss)
        #updated_data_loss = tf.maximum(updated_data_loss, 0)  # Ensuring loss doesn't go below 0
        return updated_data_loss, prob

    # Define a function to update conditional_prob and use the updated loss
    def conditional_update():
        prob = compute_conditional_probabilities(response, mask_tensor, mu_matrix, factor)
        scaled_prob = 1e-3 * prob
        updated_data_loss = original_loss
        update_op = conditional_prob.assign(prob)
        with tf.control_dependencies([update_op]):
            updated_data_loss = tf.identity(updated_data_loss)
            return updated_data_loss, tf.identity(prob)  # Use tf.identity to respect the control_dependencies
        
    def compute_conditional_calculate():
        prob = compute_conditional_probabilities(response, mask_tensor, mu_matrix, factor)
        #updated_data_loss = original_loss

        #tf.print(updated_data_loss)
        #updated_data_loss = tf.maximum(updated_data_loss, 0)  # Ensuring loss doesn't go below 0
        return prob

    # Define a function to update conditional_prob and use the updated loss
    def conditional_calculate():
        prob = compute_conditional_probabilities(response, mask_tensor, mu_matrix, factor)
        loss = original_loss
        update_op = conditional_prob.assign(prob)
        with tf.control_dependencies([update_op]):
            return loss, tf.identity(prob)
    # Modify the tf.cond to use the new function
    '''
    data_loss = tf.cond(
        tf.greater_equal(step_placeholder, 20000),
        conditional_update,
        lambda: original_loss
    )
    
    tf.cond(
        tf.logical_and(
            tf.greater_equal(step_placeholder+1, 5000),
            
            tf.equal((step_placeholder+1) % 1000, 0)
        ),
        conditional_calculate,
        lambda: tf.constant(0.0, dtype=tf.float32) 
    )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    '''
    data_loss, conditional_prob = tf.cond(
        tf.logical_and(
            tf.greater_equal(step_placeholder+1, 50000),
            tf.equal((step_placeholder+1) % 1000, 0)
        ),
        conditional_calculate,
        lambda: (original_loss, tf.constant(0.0, dtype=tf.float32))
    )
    #conditional_prob = tf.cast(conditional_prob, tf.float64)
    #data_loss = tf.cast(data_loss, tf.float64)
    #data_loss = data_loss + conditional_prob
    
    
    
    # Ensure you initialize all variables, including conditional_prob
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

    #data_loss = original_loss - a * conditional_prob
    #data_loss = tf.maximum(data_loss, 0)  # Ensuring loss doesn't go below 0
    # print("Data loss shape:", data_loss.get_shape())

    # Maintain a running average of the output images
    avg_op = slice_avg.assign(slice_avg * 0.99 + response * 0.01)
    our_image = response
    scaled_prob = 2*(tf.sigmoid(conditional_prob)-0.5)  # Scales to (0,1)
    alpha = 0.0002  # Adjust this scaling factor
    training_error = data_loss
    tf.summary.scalar('data loss', data_loss)

    # Additional summaries, savers, etc.
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=3)
    model = {
        'training_error': training_error,
        'data_loss': data_loss,
        'saver': saver,
        'summary': merged,
        'our_image': our_image,
        'is_flip_lr': is_flip_lr,
        'is_flip_ud': is_flip_ud,
        'avg_op': avg_op,
        'slice_avg': slice_avg,
        'prob': conditional_prob,
        'step_placeholder': step_placeholder,
    }
    return model


def bessel_I0(z):
    """Modified Bessel function of the first kind, I_0(x), as a piecewise function."""
    pi = tf.constant(math.pi, dtype=tf.float32)  # Define pi as a TensorFlow constant

    # Ensure z is of type float32
    z = tf.cast(z, tf.float32)

    # Calculate the value for z > 0.5
    value_for_large_z = tf.exp((z - 1) / (z + 1)) / tf.sqrt(4 * pi * (z - z ** 3))
    value_for_small_z = tf.exp(-(z ** 2 + 1) / (1 - z ** 2)) / (1 - z ** 2)
    
    # Use tf.where to choose between the constant 1 and the calculated value based on the condition
    result = tf.where(z <= tf.sqrt(5.0) - 2.0, value_for_small_z, value_for_large_z)
    
    return result

def compute_conditional_probabilities(response, mask_tensor, mu_matrix, factor):
    # Ensure mu_matrix values are between -1 and 1
    mu_matrix = tf.clip_by_value(mu_matrix, -1 + 1e-8, 1 - 1e-8)

    # Define thresholds for identifying masked and unmasked pixels
    masked_threshold = 0.1

    # 将 mask_tensor 展开成一维
    flat_mask_tensor = tf.reshape(mask_tensor, [-1])

    # 找到满足条件的元素的索引
    masked_indices = tf.where(flat_mask_tensor < masked_threshold)

    # Flatten the response for easier indexing
    flat_response = tf.reshape(response, [-1])
    
    # 归一化 flat_response 到 0-255
    flat_response = tf.cast(flat_response, tf.float32)  # 确保数据类型为 float32
    flat_response = flat_response / factor
    #flat_response = (flat_response - tf.reduce_min(flat_response)) / (tf.reduce_max(flat_response) - tf.reduce_min(flat_response)) * 255.0
    flat_mu_matrix = tf.reshape(mu_matrix, [1024, 1024])  # Assuming response shape is (1, 128, 128, 1)

    # Compute conditional probabilities for each masked pixel
    def compute_probability(masked_idx):
        masked_threshold = 0.1
        unmasked_threshold = 0.7

        # Find indices of masked and unmasked pixels
        unmasked_indices = tf.where(flat_mask_tensor >= unmasked_threshold)
        #linear_idx = masked_idx[: 1] * 32 + masked_idx[: 2]
        linear_idx = tf.cast(masked_idx, tf.int64)
        #linear_unmasked_idx = unmasked_indices[: 1] * 32 + unmasked_indices[: 2]
        linear_unmasked_idx = tf.cast(unmasked_indices, tf.int64)
        mu_values = tf.gather_nd(flat_mu_matrix, tf.stack([linear_idx * tf.ones_like(linear_unmasked_idx), linear_unmasked_idx], axis=1))

        I_i = tf.gather(flat_response, linear_idx)

        # Calculate the probabilities
        exp_component = tf.exp(-((1 + mu_values ** 2) / (1 - mu_values ** 2)))
        bessel_component = tf.math.bessel_i0((2 * mu_values) / (1 - mu_values ** 2), name=None)
        probabilities = (1 / (I_i * (1 - mu_values ** 2))) * exp_component * bessel_component
        #probabilities = bessel_I0(mu_values) / I_i
        # 确保仅包括有效的概率值，即除去NaN, Inf，并且值域在(0, 1)之间
        valid_prob_mask = tf.math.is_finite(probabilities)
        '''
        valid_prob_mask = tf.logical_and(
            tf.math.is_finite(probabilities),  # 移除 NaN 和 Inf
            tf.logical_and(
                tf.greater(probabilities, 0.0),  # 仅包含大于0的值
                tf.less(probabilities, 1.0)      # 仅包含小于1的值
            )
        )
        '''
        # 应用 boolean mask 来筛选有效的概率值
        probabilities = tf.boolean_mask(probabilities, valid_prob_mask)
        avg_row = tf.reduce_mean(probabilities)
        tf.print("Processing index:", masked_idx, "Average Probability:", avg_row)

        return avg_row

    #masked_indices_flat = tf.cast(masked_indices[:, 1] * 32 + masked_indices[:, 2], tf.int64)
    #unmasked_indices_flat = tf.cast(unmasked_indices[:, 1] * 32 + unmasked_indices[:, 2], tf.int64)
    #mesh_masked, mesh_unmasked = tf.meshgrid(masked_indices_flat, unmasked_indices_flat, indexing='ij')
    #all_pairs = tf.stack([tf.reshape(mesh_masked, [-1]), tf.reshape(mesh_unmasked, [-1])], axis=1)

    probabilities = tf.map_fn(compute_probability, masked_indices, dtype=tf.float32, parallel_iterations=1)
    valid_prob_mask = tf.math.is_finite(probabilities)
    valid_probabilities = tf.boolean_mask(probabilities, valid_prob_mask)

    # 检查是否存在有效的概率值
    has_valid_probabilities = tf.reduce_any(valid_prob_mask)

    # 使用 tf.cond 根据是否存在有效的概率值来返回结果
    total_average_prob = tf.cond(
        has_valid_probabilities,
        lambda: tf.reduce_mean(valid_probabilities),
        lambda: tf.constant(0.1, dtype=tf.float32)
    )

    return total_average_prob


'''
def compute_conditional_probabilities(response, mask_tensor, mu_matrix):
    mu_matrix = tf.cast(mu_matrix, dtype=tf.float32)
    # Clip mu values to ensure they are between -1 and 1
    mu_matrix = tf.clip_by_value(mu_matrix, -1 + 1e-8, 1 - 1e-8)
    
    mu_squared = tf.square(mu_matrix)
    denom = 1 - mu_squared
    
    # Get the values of I_i at the masked locations
    I_i = tf.expand_dims(response * (1-mask_tensor), axis=-1)
    
    # Calculate new prefactor considering I_i
    prefactor = 1 / (I_i * denom)
    
    exp_term = - (1 + mu_squared) / denom
    bessel_term = 2 * mu_matrix / denom
    
    exp_component = tf.exp(exp_term)
    bessel_component = bessel_I0(bessel_term)
    
    conditional_prob = prefactor * exp_component * bessel_component
    
    # Apply mask to select only known-to-unknown influences
    known_mask = mask_tensor  # 1 where pixels are known
    unknown_mask = (1-mask_tensor)    # 1 where pixels are unknown
    
    # Compute the conditional probabilities for known-to-unknown only
    conditional_prob_masked = conditional_prob * unknown_mask
    
    # Filter out inf and nan values
    conditional_prob_masked = tf.where(tf.math.is_finite(conditional_prob_masked), conditional_prob_masked, tf.zeros_like(conditional_prob_masked))
    
    # Sum over all known pixels to get the average influence on each unknown pixel from all known pixels
    sum_prob = tf.reduce_sum(conditional_prob_masked, axis=1)
    count_known = tf.reduce_sum(tf.cast(tf.math.is_finite(conditional_prob_masked), tf.float32), axis=1)  # Adjust count to only include finite values
    
    # Calculate average probabilities
    average_prob_per_unknown = sum_prob / (count_known + 1e-8)
    
    # Now calculate the overall average for the image by averaging over all unknown pixels
    total_average_prob = tf.reduce_mean(average_prob_per_unknown)

    return total_average_prob
'''


def build_inpainting_unet(img, mask, p=0.7):
    _, h, w, c = np.shape(img)
    img_tensor = tf.identity(img)
    mask_tensor = tf.identity(mask)
    response = tf.transpose(img_tensor, [0, 3, 1, 2])
    mask_tensor_sample = tf.transpose(mask_tensor, [0, 3, 1, 2])
    mask_tensor_sample = tf.nn.dropout(mask_tensor_sample, 0.7) * 0.7
    response = tf.multiply(mask_tensor_sample, response)
    slice_avg = tf.get_variable('slice_avg', shape=[_, h, w, c], initializer=tf.initializers.zeros())
    response = partial_conv_unet(response, mask_tensor_sample, channel=c, width=w, height=h, p=p)
    response = tf.transpose(response, [0, 2, 3, 1])
    mask_tensor_sample = tf.transpose(mask_tensor_sample, [0, 2, 3, 1])
    data_loss = mask_loss(response, img_tensor, mask_tensor - mask_tensor_sample)
    avg_op = slice_avg.assign(slice_avg * 0.99 + response * 0.01)
    our_image = img_tensor + tf.multiply(response, 1 - mask_tensor)

    training_error = data_loss
    tf.summary.scalar('data loss', data_loss)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=3)
    model = {
        'training_error': training_error,
        'data_loss': data_loss,
        'saver': saver,
        'summary': merged,
        'our_image': our_image,
        'avg_op': avg_op,
        'slice_avg': slice_avg,
    }

    return model


def mask_loss(x, labels, masks):
    cnt_nonzero = tf.to_float(tf.count_nonzero(masks))
    loss = tf.reduce_sum(tf.multiply(tf.math.pow(x - labels, 2), masks)) / cnt_nonzero
    return loss


def data_arg(x, is_flip_lr, is_flip_ud):
    x = tf.cond(is_flip_lr > 0, lambda: tf.image.flip_left_right(x), lambda: x)
    x = tf.cond(is_flip_ud > 0, lambda: tf.image.flip_up_down(x), lambda: x)
    return x
