# IOT FACE recognition based on Facenet 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import cv2
import align.detect_face
import random
import math
import pickle

from sklearn.svm import SVC
from time import sleep

def main(args):
    #Load MTCNN model for detecting and aligning Faces in the Captured Photos
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    nrof_successfully_aligned = 0
    
    # Save faces files locally just to varify. You may want to remove this once your system is set up.
    output_filename = 'd:\PhotoCaptured.png'

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # args.seed defaulted to 666
            np.random.seed(seed=666)
        
        # Load the model once
        print('Loading feature extraction model')
        
        # Use your path where you have saved pretrained facenet model
        facenet.load_model('./models/20170512-110547.pb')
        
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # your custom classifier trained the last layer with your own image database. Please refer to Facenet repo for training custom classifier
        classifier_filename_exp = os.path.expanduser('./models/my_classifier.pkl')

        # Classify images
        print('Testing classifier')
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        print('Loaded classifier model from file "%s"' % classifier_filename_exp)

    #Start Video Capture
    video_capture = cv2.VideoCapture(0)
    
    #All the pre-loading is done. Now loop through capturing photos and recognizing faces in the frames
    while True:
    
        try:
            ret, frame = video_capture.read()
            img = frame

        except (IOError, ValueError, IndexError) as e:
            print("Error")
        else:
            if img.ndim<2:
                print('Unable to align "%s"' % image_path)
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:,:,0:3]
    
            bounding_boxes, box_cord = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            
            nrof_faces = bounding_boxes.shape[0]
            #Define npArray of 3x2 and assign scaled to it. XXXXXXXXXXXXXXxx
            #face_array = np.array(160,160,3)
            face_list = []
            print('Number of faces ******* %s', nrof_faces)
            #for rectangle in range(0,nrof_faces):
                #cv2.rectangle(img,box_cord[rectangle],(0,255,0),5)
            print('Type of Box Cord ******* %s',type(box_cord))
            print('shape of Box Cord ******* %s', box_cord.shape)
            # Display the resulting frame
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
              
            if nrof_faces>0:
                det = bounding_boxes[:,0:4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces>1:
                    #if args.detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    det_arr.append(np.squeeze(det))
    
                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    # Hardcoding
                    # args.margin = 32 image_size 160
                    bb[0] = np.maximum(det[0]-32/2, 0)
                    bb[1] = np.maximum(det[1]-32/2, 0)
                    bb[2] = np.minimum(det[2]+32/2, img_size[1])
                    bb[3] = np.minimum(det[3]+32/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
                    nrof_successfully_aligned += 1
                    filename_base, file_extension = os.path.splitext(output_filename)
                    
                    #if args.detect_multiple_faces: #Try keeping it in nparray insted of writing
                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                    #else:
                        #output_filename_n = "{}{}".format(filename_base, file_extension)
                    misc.imsave(output_filename_n, scaled)
                    print('type of scaled************',type(scaled))
                    #Appending each face to face_array
                    face_list.append(scaled)
            else:
                print('No Image or - Unable to align')
                continue
              
            #Invoke Classifier Code
        
            # Run forward pass to calculate embeddings
            print('Calculating features for images')

            nrof_images = nrof_faces
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 1000))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                #start_index = i * args.batch_size - Hardcoded Batch Size
                start_index = i * 1000
                #end_index = min((i + 1) * args.batch_size, nrof_images)
                end_index = min((i + 1) * 1000, nrof_images)
                 
                images = Face_load_data(face_list, False, False, 160)
                
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            
            
            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            
            #Print Face recognization result for each Face in the Frame
            for i in range(len(best_class_indices)):
                print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                                
    video_capture.release()
    
def Face_load_data(face_list, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(face_list)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = face_list[i]
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


if __name__ == '__main__':
    main(sys.argv)
