
from keras.models import Model
from keras.models import Sequential, Model
from keras.models import load_model
from keras import layers
from keras.layers import Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import BatchNormalization
from keras.layers import Input
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from Results import Results
import time

#tf.enable_eager_execution()

import util

class SAEModel:

    def __init__(self, kernel_shape, num_filters, input_shape, num_blocks, pool_size, with_batch_normalization, dropout, optimizer, bn_axis, considered_classes):
        self.kernel_shape = kernel_shape
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.pool_size = pool_size
        self.with_batch_normalization = with_batch_normalization
        self.optimizer = optimizer
        self.bn_axis = bn_axis
        self.dropout = dropout
        self.considered_classes = considered_classes

        self.autoencoder = self.build_model()
        #self.autoencoder.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])


        #self.autoencoder.build((None, input_shape[0], input_shape[1], 1))
        self.autoencoder.summary()


    def getModelPath(self, fold_name, db_train, batch_size):
        if self.dropout > 0:
            dropout_str = "drop_" + str(self.dropout)
        else:
            dropout_str = ""

        path_model = "models/SAE/" + fold_name + "/" + \
                    str(db_train).replace("/", "-").replace("[", "").replace("'", "").replace(", ", "_").replace("]", "") + "_" +\
                    "cls_" + str(self.considered_classes).replace("[", "").replace("'", "").replace(", ", "_").replace("]", "") + "_" +\
                    "k_" + str(self.kernel_shape[0]) + "x" + str(self.kernel_shape[1]) + "_" + \
                    "f_" + str(self.num_filters) + "_" + \
                    "w_" + str(self.input_shape[0]) + "x" + str(self.input_shape[1]) + "_" + \
                    "l_" + str(self.num_blocks) + "_" + \
                    "p_" + str(self.pool_size) + "_" + \
                    "b_" + str(batch_size) + "_" + \
                    "bn_" + str(self.with_batch_normalization) + "_" + \
                    dropout_str + \
                    "opt_" + str(type(self.optimizer).__name__)  + \
                    ".h5"
        return path_model
      

    def addConvolutionalBlock(self, input_layer, num_filters, kernel_shape, pool_size, with_batch_normalization):
        layer = Convolution2D(num_filters,kernel_size=kernel_shape, padding='same')(input_layer)

        if (with_batch_normalization == True):
            layer = BatchNormalization(axis=self.bn_axis)(layer)

        layer = Activation('relu')(layer) 

        if self.dropout > 0:
            layer = Dropout(self.dropout)(layer)

        if (pool_size > 1):
            layer = MaxPooling2D(pool_size=(pool_size, pool_size))(layer)

        return layer


    def addUpsamplingBlock(self, input_layer, num_filters, kernel_shape, pool_size, with_batch_normalization):
        layer = Convolution2D(num_filters,kernel_size=kernel_shape, padding='same')(input_layer)

        if (with_batch_normalization == True):
            layer = BatchNormalization(axis=self.bn_axis)(layer)

        layer = Activation('relu')(layer)

        if self.dropout > 0:
            layer = Dropout(self.dropout)(layer)

        if (pool_size > 1):
            layer = UpSampling2D((pool_size, pool_size))(layer)

        return layer


    def build_model(self):
        input_img = Input(shape=self.input_shape)
        
        # Encoding
        autoencoder = input_img

        for i in range(self.num_blocks):
            autoencoder = self.addConvolutionalBlock(autoencoder, self.num_filters, self.kernel_shape, self.pool_size, self.with_batch_normalization)

        self.latent_code = autoencoder
        
        for i in range(self.num_blocks):
            autoencoder = self.addUpsamplingBlock(autoencoder, self.num_filters, self.kernel_shape, self.pool_size, self.with_batch_normalization)
            
        # Prediction
        autoencoder = Convolution2D(1, kernel_size=self.kernel_shape, padding='same')(autoencoder)
        autoencoder = Activation('sigmoid')(autoencoder)

        #autoencoder = Model(inputs=input, outputs=autoencoder)
        #model.summary()

        model = Model(input_img, autoencoder)

        return model

    
    
    def train(
                self, db_source, db_target, fold_name, 
                epochs, batch_size, sample_filter, super_epochs,
                verbose,
                considered_classes,
                list_files_db1_train, list_files_db1_train_json,  
                list_files_db1_val, list_files_db1_val_json,  
                list_files_db2_val, list_files_db2_val_json):
        
        assert(len(db_source) == 1)
        best_fscore = -1

        db_source_name = str(db_source[0]).replace("/", "-")
        db_target_name = str(db_target[0]).replace("/", "-")

        path_model = self.getModelPath(fold_name, db_source, batch_size)
        util.mkdirp(os.path.dirname(path_model))
        print("Model will be saved in: " + str(path_model))

        train_generator = util.create_generator(list_files_db1_train, list_files_db1_train_json, self.input_shape, sample_filter, batch_size, considered_classes)
        #val_generator = util.create_generator(val_files, self.input_shape, None, None, batch_size)
        accuracy = keras.metrics.Accuracy()
        loss_label = keras.losses.BinaryCrossentropy()

        windows_shape = (self.input_shape[0], self.input_shape[1])
        best_epoch = 0

        total_number_samples = len(list_files_db1_train)
        start_train = time.time()
        for epoch in range(1, epochs+1):
            idx_progress = 0
            if(verbose):
                progress_bar = util.createProgressBar("Epoch-" + str(epoch), total_number_samples)
                progress_bar.start()
            else:
                print('*'*80)
                print("Epoch-" + str(epoch))

            start = time.time()
            for gr_batch_imgs, gt_batch_imgs in train_generator:
                with tf.GradientTape() as tape:

                    gr_batch_imgs = gr_batch_imgs.reshape(gr_batch_imgs.shape[0], gr_batch_imgs.shape[1], gr_batch_imgs.shape[2], 1)
                    '''
                    print(np.amin(gr_batch_imgs[0,:,:,0]))
                    print(np.amax(gr_batch_imgs[0,:,:,0]))
                    print(np.amin(gt_batch_imgs[0,:,:]))
                    print(np.amax(gt_batch_imgs[0,:,:]))
                    util.saveImage(gr_batch_imgs[0,:,:,0]*255, "output/src.png")
                    util.saveImage(gt_batch_imgs[0,:,:]*255, "output/gt.png")
                    '''
                    logits = self.autoencoder(tf.convert_to_tensor(gr_batch_imgs, dtype=tf.float32))
                    # Compute the loss value for this batch.
                    loss_value = loss_label(tf.convert_to_tensor(gt_batch_imgs, dtype=tf.float32), logits)

                    idx_progress += len(gr_batch_imgs)

                    if(verbose):
                        progress_bar.update(idx_progress%total_number_samples)

                # Update the state of the `accuracy` metric.
                accuracy.update_state(gt_batch_imgs, logits)
                
                # Update the weights of the model to minimize the loss value.
                gradients = tape.gradient(loss_value, self.autoencoder.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_weights))

                if idx_progress >= total_number_samples-batch_size:
                    break

            if(verbose):
                progress_bar.finish()

            idx_progress = 0
            print ("Train")
            print (
                    "Loss: %.3f" % loss_value, 
                    "Accuracy: %.3f" % accuracy.result()
                )

            if (epoch % super_epochs) == 0:
                print("Source Validation...")
                results_source = util.evaluateModelListFolders("CNN/" + db_source_name + "-" + db_target_name+ "/" + db_source_name, self.autoencoder, list_files_db1_val, list_files_db1_val_json, windows_shape, batch_size, False, considered_classes)
                assert(len(results_source) == 1)
                pseudo_threshold_source = results_source[0].getPseudoThreshold()

                print("Target Validation...")
                results_target = util.evaluateModelListFolders("CNN/" + db_source_name + "-" + db_target_name+ "/" + db_target_name, self.autoencoder, list_files_db2_val, list_files_db2_val_json, windows_shape, batch_size, False, considered_classes, pseudo_threshold_source)
                
                print('-'*80)
                print ("EPOCH SUMMARY... (epoch %d)" % epoch)
                print("Source " + str(db_source))
                util.printResults(db_source, results_source)
                
                print("Target " + str(db_target))
                util.printResults(db_target, results_target)

                print('-'*80)

                pseudo_fscore_source = results_source[0].getPseudoFscore()
                if pseudo_fscore_source > best_fscore:
                    print("Pseudo-F1 improved (%.3f -> %.3f) in epoch %d (superepoch %d)" % (best_fscore, pseudo_fscore_source, epoch, (epoch // super_epochs)))
                    print("Model saved in " + path_model)
                    self.autoencoder.save(path_model)
                    best_fscore = pseudo_fscore_source
                    best_epoch = epoch
                else:
                    print("The model does not improve the best result (%.3f in epoch %d)" % (best_fscore, best_epoch))
            # Reset the metric's state at the end of an epoch
            accuracy.reset_states()

            end = time.time()
            print("Time of epoch: " + str(end - start) + " seconds")

        end_train = time.time()
        print("Time of training: " + str(end_train - start_train) + " seconds")
            
