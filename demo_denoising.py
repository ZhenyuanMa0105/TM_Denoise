import tensorflow as tf
import network.Punet
import numpy as np

import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SAVE = 1000
N_STEP = 400000
GLOBAL_A = 0.0000001
Threshold = 500000

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# 设置分布式策略
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync)) 
with strategy.scope():
    def train(file_path, dropout_rate, sigma=25, is_realnoisy=False):
        print(file_path)
        tf.reset_default_graph()
        gt = util.load_np_image(file_path)
        _, w, h, c = np.shape(gt)
        # Extract index from file name and load corresponding mu_matrix
        index = file_name.split('_')[-1].split('.')[0]
        mu_matrix_path = os.path.join('data/train/output_corr_0', f'complex_train_speckle_{index}.npy')
        mu_matrix = np.load(mu_matrix_path)

        model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/stat/"
        os.makedirs(model_path, exist_ok=True)
        #gt_flat = np.reshape(gt, [-1])
        #factor = np.max(gt_flat) / 255
        noisy = gt  # You can add noise here if needed
        step_placeholder = tf.placeholder(tf.int64, shape=(), name='step_placeholder')

        model = network.Punet.build_denoising_unet(noisy, mu_matrix, 1 - dropout_rate, is_realnoisy, GLOBAL_A, Threshold)

        loss = model['training_error']
        summary = model['summary']
        saver = model['saver']
        our_image = model['our_image']
        is_flip_lr = model['is_flip_lr']
        is_flip_ud = model['is_flip_ud']
        avg_op = model['avg_op']
        slice_avg = model['slice_avg']
        prob = model['prob']
        step_placeholder = model['step_placeholder']
        #optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        grads_and_vars = optimizer.compute_gradients(loss)
        capped_grads_and_vars = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grads_and_vars if grad is not None]
        
        train_op = optimizer.apply_gradients(capped_grads_and_vars)
        

        avg_loss = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(model_path, sess.graph)
            for step in range(N_STEP):
                feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2), step_placeholder: step}
                _, _op, loss_value, merged, o_image, prob = sess.run([train_op, avg_op, loss, summary, our_image, model['prob']], feed_dict=feet_dict)
                avg_loss += loss_value
                #print(f"Conditional Probability: {prob}")
                if (step + 1) % N_SAVE == 0:

                    print("After %d training step(s)" % (step + 1),
                          "loss  is {:.9f}".format(loss_value))
                    print(f"Conditional Probability: {prob}")
                    avg_loss = 0
                    sum = np.float32(np.zeros(our_image.shape.as_list()))
                    for j in range(N_PREDICTION):
                        feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2), step_placeholder: step}
                        o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                        sum += o_image
                    o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                    o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))
                    if is_realnoisy:
                        cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '.png', o_avg)
                    else:
                        cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '.png', o_image)
                    saver.save(sess, model_path + "model.ckpt-" + str(step + 1))

                summary_writer.add_summary(merged, step)


if __name__ == '__main__':
    path = './data/train/demo_L/'
    file_list = os.listdir(path)
    for sigma in [150]:
        for file_name in file_list:
            if not os.path.isdir(path + file_name):
                train(path + file_name, 0.3, sigma)

