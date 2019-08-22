# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''

from verification import evaluate
from scipy.optimize import brentq
from scipy import interpolate
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import os
import pickle
import mxnet as mx
import cv2


slim = tf.contrib.slim


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=1)
    parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='/home/yww/faces_vgg_112x112', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--saver_maxkeep', default=50, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--pretrained_model', type=str, default='./output/ckpt_best/mobilefacenet_best_ckpt',
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--model_type', default=0, help='MobileFaceNet or TinyMobileFaceNet or SqueezeFaceNet or ShuffleFaceNet or ShuffleFaceNet_v2')
    parser.add_argument('--gpu', default='1', help='GPU index')

    args = parser.parse_args()
    return args

def load_data(db_name, image_size, args):
    bins, issame_list = pickle.load(open(os.path.join(args.eval_db_path, db_name+'.bin'), 'rb'), encoding='bytes')
    datasets = np.empty((len(issame_list)*2, image_size[0], image_size[1], 3))
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        # img = cv2.imdecode(np.fromstring(_bin, np.uint8), -1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (112,112))
        img = img - 127.5
        img = img * 0.0078125
        datasets[i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(datasets.shape)

    return datasets, issame_list

if __name__ == '__main__':
    with tf.Graph().as_default():
        args = get_parser()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        # define placeholder
        inputs = tf.placeholder(name='img_inputs',
                                shape=[None, args.image_size[0], args.image_size[1], 3],
                                dtype=tf.float32)
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None,
                                                              name='phase_train')

        # prepare validate datasets
        ver_list = []
        ver_name_list = []
        for db in args.eval_datasets:
            print('begin db %s convert.' % db)
            data_set = load_data(db, args.image_size, args)
            ver_list.append(data_set)
            ver_name_list.append(db)
            
        pretrained_model = './output/ckpt_best/MobileFaceNet_9925_9680.pb'

        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping,
                                gpu_options=gpu_options,
                                )
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        print('Restoring pretrained model: %s' % pretrained_model)
        output_graph_def = tf.GraphDef()
        with open(pretrained_model, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _= tf.import_graph_def(output_graph_def, name="")
        
        inputs= sess.graph.get_tensor_by_name("img_inputs:0")  
        print("input_X:",inputs)
        out = sess.graph.get_tensor_by_name("embeddings:0")  
        print("input_X:",out)
        
        op = sess.graph.get_operations()
        for i,m in enumerate(op):
            print('op{}:'.format(i),m.values())

        total_accuracy = {}

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        print('testing...')
        for ver_step in range(len(ver_list)):
            start_time = time.time()
            data_sets, issame_list = ver_list[ver_step]
            emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
            nrof_batches = data_sets.shape[0] // args.test_batch_size
            for index in range(nrof_batches):  # actual is same multiply 2, test data total
                start_index = index * args.test_batch_size
                end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])
                
                img_inputs = sess.graph.get_tensor_by_name('img_inputs:0')
                phase_train = sess.graph.get_tensor_by_name('phase_train:0')
                feed_dict = {img_inputs: data_sets[start_index:end_index, ...], phase_train: False}
                embeddings = sess.graph.get_tensor_by_name('embeddings:0')
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                
            duration = time.time() - start_time
            tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list, nrof_folds=args.eval_nrof_folds)

            print("total time %.3f to evaluate %d images of %s" % (duration,
                                                                   data_sets.shape[0],
                                                                   ver_name_list[ver_step]))
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f\n' % eer)

            coord.request_stop()
            coord.join(threads)