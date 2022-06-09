
from keras import backend as K
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
import time
from Results import Results
from nfnet_layers import WSConv2D


#tf.enable_eager_execution()

import util
import utilGradientReversalLayer
import utilConst

class SAEDANNModel:

    def __init__(self, 
                    kernel_shape, 
                    num_filters, 
                    input_shape, 
                    num_blocks, 
                    pool_size, 
                    with_batch_normalization, with_adaptative_clipping, with_adaptative_clipping_centered,
                    dropout,
                    optimizer_sae, 
                    optimizer_domain, 
                    bn_axis, 
                    grl_pos, 
                    lambda_init, lambda_inc,
                    considered_classes):
        assert(grl_pos >= -num_blocks and grl_pos <= +num_blocks)

        self.kernel_shape = kernel_shape
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.pool_size = pool_size
        self.with_batch_normalization = with_batch_normalization
        self.optimizer_sae = optimizer_sae
        self.optimizer_domain = optimizer_domain
        self.bn_axis = bn_axis
        self.grl_pos = grl_pos
        self.lambda_init = lambda_init
        self.lambda_inc = lambda_inc
        self.with_adaptative_clipping = with_adaptative_clipping
        self.with_adaptative_clipping_centered = with_adaptative_clipping_centered
        self.dropout = dropout

        self.input_img = Input(shape=self.input_shape)

        self.autoencoder = self.build_model(self.input_img)
        self.domain_classifier = self.build_domain_classifier(self.latent_code, self.input_img)

        self.autoencoder.summary()
        self.considered_classes = considered_classes


    def getModelPath(self, fold_name, db_source, db_target, batch_size, epochs_pretrain, with_domain_stopping):
        
        
        if epochs_pretrain == 0:
            epochs_pretrain_str = ""
        else:
            epochs_pretrain_str = "pre_" + str(epochs_pretrain)

        if self.with_adaptative_clipping:
            with_adaptative_clipping_str = "clip"
        else:
            with_adaptative_clipping_str = ""

        if self.with_adaptative_clipping_centered:
            with_adaptative_clipping_centered_str = "clipcnt"
        else:
            with_adaptative_clipping_centered_str = ""

        if with_domain_stopping:
            basedir = "models/SAEDANN/DOMSTOP/"
        else:
            basedir = "models/SAEDANN/"

        if self.dropout > 0.:
            dropout_str = "drop_" + str(self.dropout)
        else:
            dropout_str = ""

        if str(type(self.optimizer_domain).__name__) == str(type(self.optimizer_sae).__name__):
            str_opt_domain = ""
        else:
            str_opt_domain = "optd_" + str(type(self.optimizer_domain).__name__)

        path_model = basedir + \
                    fold_name + "/" \
                    "dbs_" + str(db_source).replace("/", "-").replace("[", "").replace("'", "").replace(", ", "_").replace("]", "") + "_" + \
                    "dbt_" + str(db_target).replace("/", "-").replace("[", "").replace("'", "").replace(", ", "_").replace("]", "") + "_" + \
                    "cls_" + str(self.considered_classes).replace("[", "").replace("'", "").replace(", ", "_").replace("]", "") + "_" +\
                    "k_" + str(self.kernel_shape[0]) + "x" + str(self.kernel_shape[1]) + "_" + \
                    "f_" + str(self.num_filters) + "_" + \
                    "w_" + str(self.input_shape[0]) + "x" + str(self.input_shape[1]) + "_" + \
                    "l_" + str(self.num_blocks) + "_" + \
                    "p_" + str(self.pool_size) + "_" + \
                    "b_" + str(batch_size) + "_" + \
                    "bn_" + str(self.with_batch_normalization) + "_" + \
                    "opt_" + str(type(self.optimizer_sae).__name__)  + \
                    str_opt_domain +\
                    dropout_str + \
                    "lda_" + str(self.lambda_init) + \
                    "ldainc_" + str(self.lambda_inc) + \
                    "grlpos_" + str(self.grl_pos) + \
                    epochs_pretrain_str + \
                    with_adaptative_clipping_str + \
                    with_adaptative_clipping_centered_str + \
                    ".h5"
        return path_model
      

    def addConvolutionalBlock(self, input_layer, num_filters, kernel_shape, pool_size, with_batch_normalization, with_adaptative_clipping, name_gain_adaptative_clipping):
        layer = Convolution2D(num_filters,kernel_size=kernel_shape, padding='same')(input_layer)

        if (with_batch_normalization == True):
            layer = BatchNormalization(axis=self.bn_axis)(layer)

        if (with_adaptative_clipping == True):
            layer = WSConv2D(num_filters, kernel_size=kernel_shape, padding='same', name=name_gain_adaptative_clipping)(layer)

        layer = Activation('relu')(layer) 

        if self.dropout > 0:
            layer = Dropout(self.dropout)(layer)
            
        if (pool_size > 1):
            layer = MaxPooling2D(pool_size=(pool_size, pool_size))(layer)

        return layer


    def addUpsamplingBlock(self, input_layer, num_filters, kernel_shape, pool_size, with_batch_normalization, with_adaptative_clipping, name_gain_adaptative_clipping):

        layer = Convolution2D(num_filters,kernel_size=kernel_shape, padding='same')(input_layer)

        if (with_batch_normalization == True):
            layer = BatchNormalization(axis=self.bn_axis)(layer)

        if (with_adaptative_clipping == True):
            layer = WSConv2D(num_filters, kernel_size=kernel_shape, padding='same',name=name_gain_adaptative_clipping)(layer)

        layer = Activation('relu')(layer)
        
        if self.dropout > 0:
            layer = Dropout(self.dropout)(layer)

        if (pool_size > 1):
            layer = UpSampling2D((pool_size, pool_size))(layer)

        return layer

    def build_domain_classifier(self, input_domain, input_img):

        if self.with_adaptative_clipping_centered:
            self.clipping_center_layer = WSConv2D(self.num_filters, kernel_size=self.kernel_shape, padding='same', name="clipcenter")(input_domain)
            input_domain = self.clipping_center_layer
        else:
            self.clipping_center_layer = None

        self.grl_layer = utilGradientReversalLayer.GradientReversal(self.lambda_init)
        classifier = self.grl_layer(input_domain)

        num_layers_downsampling = max(self.num_blocks, self.num_blocks - self.grl_pos) - self.num_blocks
        num_layers_upsampling = min(self.num_blocks, self.num_blocks - self.grl_pos)

        name_gain_adaptative_clipping = "domgain"
        for i in range(num_layers_downsampling):
            name_gain_adaptative_clipping_i = name_gain_adaptative_clipping + "enc" + str(i)
            classifier = self.addConvolutionalBlock(
                                    classifier, 
                                    self.num_filters, 
                                    self.kernel_shape, 
                                    self.pool_size, 
                                    self.with_batch_normalization, 
                                    self.with_adaptative_clipping, 
                                    name_gain_adaptative_clipping_i)
        
        for i in range(num_layers_upsampling):
            name_gain_adaptative_clipping_i = name_gain_adaptative_clipping + "dec" + str(i)
            classifier = self.addUpsamplingBlock(
                                    classifier, 
                                    self.num_filters, 
                                    self.kernel_shape, 
                                    self.pool_size, 
                                    self.with_batch_normalization, 
                                    self.with_adaptative_clipping, 
                                    name_gain_adaptative_clipping_i)
            
        classifier = Convolution2D(1, kernel_size=self.kernel_shape, padding='same')(classifier)
        classifier = Activation('sigmoid')(classifier)

        model = Model(input_img, classifier)
        
        return model

    def build_model(self, input_img):
        
        # Encoding
        autoencoder = input_img

        idx_grl_pos = self.grl_pos + self.num_blocks
        idx_layer = 1

        name_gain_adaptative_clipping = "gain"
        for i in range(self.num_blocks):
            name_gain_adaptative_clipping_i = name_gain_adaptative_clipping + "enc" + str(i)
            autoencoder = self.addConvolutionalBlock(   
                                autoencoder, 
                                self.num_filters, 
                                self.kernel_shape, 
                                self.pool_size, 
                                self.with_batch_normalization, 
                                self.with_adaptative_clipping,
                                name_gain_adaptative_clipping_i)
            if (idx_layer == idx_grl_pos):
                self.latent_code = autoencoder
            idx_layer+=1
        
        for i in range(self.num_blocks):
            name_gain_adaptative_clipping_i = name_gain_adaptative_clipping + "dec" + str(i)
            autoencoder = self.addUpsamplingBlock(
                                autoencoder, 
                                self.num_filters, 
                                self.kernel_shape, 
                                self.pool_size, 
                                self.with_batch_normalization, 
                                self.with_adaptative_clipping,
                                name_gain_adaptative_clipping_i)
            if (idx_layer == idx_grl_pos):
                self.latent_code = autoencoder
            idx_layer+=1
            
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
                epochs_pretrain,
                verbose,
                considered_classes,
                list_files_db1_train, list_files_db1_train_json,  
                list_files_db1_val, list_files_db1_val_json,  
                list_files_db2_train, list_files_db2_train_json,
                list_files_db2_val, list_files_db2_val_json):
        
        assert(len(db_source) == 1)

        db_source_name = str(db_source[0]).replace("/", "-")
        db_target_name = str(db_target[0]).replace("/", "-")

        best_fscore = -1
        worst_acc_domain = 1

        total_number_samples = len(list_files_db1_train)

        path_model = self.getModelPath(fold_name, db_source, db_target, batch_size, epochs_pretrain, False)
        path_model_domain_stopping = self.getModelPath(fold_name, db_source, db_target, batch_size, epochs_pretrain, True)

        util.mkdirp(os.path.dirname(path_model))
        util.mkdirp(os.path.dirname(path_model_domain_stopping))
        
        print("Model will be saved in: " + str(path_model))
        print("Model with domain stopping criteria will be saved in: " + str(path_model_domain_stopping))

        train_generator_source = util.create_generator(list_files_db1_train, list_files_db1_train_json, self.input_shape, sample_filter, batch_size, considered_classes)
        train_generator_target = util.create_generator(list_files_db2_train, list_files_db2_train_json, self.input_shape, utilConst.FILTER_WITHOUT, batch_size, considered_classes)

        accuracy_autoencoder = tf.keras.metrics.BinaryAccuracy(
                                    name="binary_accuracy", dtype=None, threshold=0.5
                                )
        accuracy_domain = tf.keras.metrics.BinaryAccuracy(
                                    name="binary_accuracy", dtype=None, threshold=0.5
                                )

        loss_label = keras.losses.BinaryCrossentropy()

        windows_shape = (self.input_shape[0], self.input_shape[1])
        best_epoch = 0
        best_epoch_domain_stopping = 0

        gt_domain_db1 = np.zeros((windows_shape[0], windows_shape[1]), dtype=np.float16)
        gt_domain_db2 = np.ones((windows_shape[0], windows_shape[1]), dtype=np.float16)
        
        gt_domains_db1 = []
        gt_domains_db2 = []
        for idx in range(batch_size):
            gt_domains_db1.append(gt_domain_db1)
        for idx in range(batch_size):
            gt_domains_db2.append(gt_domain_db2)

        gt_domains_db1 = np.asarray(gt_domains_db1)
        gt_domains_db2 = np.asarray(gt_domains_db2)

        
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
            for gr_batch_imgs_source, gt_batch_imgs_source in train_generator_source:
                with tf.GradientTape(persistent=True) as tape:

                    gr_batch_imgs_target, _ = next(train_generator_target)

                    gr_batch_imgs_source = gr_batch_imgs_source.reshape(gr_batch_imgs_source.shape[0], gr_batch_imgs_source.shape[1], gr_batch_imgs_source.shape[2], 1)
                    gr_batch_imgs_target = gr_batch_imgs_target.reshape(gr_batch_imgs_target.shape[0], gr_batch_imgs_target.shape[1], gr_batch_imgs_target.shape[2], 1)

                    logits_autoencoder = self.autoencoder(tf.convert_to_tensor(gr_batch_imgs_source, dtype=tf.float16))
                    loss_value_autoencoder = loss_label(tf.convert_to_tensor(gt_batch_imgs_source, dtype=tf.float16), logits_autoencoder)

                    if epoch > epochs_pretrain:
                        gr_domain = np.concatenate((gr_batch_imgs_source, gr_batch_imgs_target))
                        gt_domain = np.concatenate((gt_domains_db1[0:len(gr_batch_imgs_source)], gt_domains_db2[0:len(gr_batch_imgs_target)]))
                        logits_domain = self.domain_classifier(tf.convert_to_tensor(gr_domain, dtype=tf.float16))
                        loss_value_domain = loss_label(tf.convert_to_tensor(gt_domain, dtype=tf.float16), logits_domain)


                # Update the state of the `accuracy` metric.
                accuracy_autoencoder.update_state(gt_batch_imgs_source, logits_autoencoder)
            
                # Update the weights of the model to minimize the loss value.
                gradients_autoencoder = tape.gradient(loss_value_autoencoder, self.autoencoder.trainable_weights)


                if (self.with_adaptative_clipping_centered and not self.with_adaptative_clipping):
                    self.optimizer_sae.apply_gradients(zip(gradients_autoencoder, self.autoencoder.trainable_weights))
                else:
                    self.optimizer_sae.apply_gradients(
                            (grad, var) 
                            for (grad, var) in zip(gradients_autoencoder, self.autoencoder.trainable_weights) 
                            if grad is not None
                        )

                if epoch > epochs_pretrain:
                    accuracy_domain.update_state(gt_domain, logits_domain)
                    gradients_domain = tape.gradient(loss_value_domain, self.domain_classifier.trainable_weights)

                    #self.optimizer_domain.apply_gradients(zip(gradients_domain, self.domain_classifier.trainable_weights))
                    
                    #'''
                    self.optimizer_domain.apply_gradients(
                            (grad, var) 
                            for (grad, var) in zip(gradients_domain, self.domain_classifier.trainable_weights) 
                            if grad is not None
                        )
                    #'''
                else:
                    loss_value_domain = 0.

                idx_progress += len(gr_batch_imgs_source)

                if(verbose):
                    progress_bar.update(idx_progress%total_number_samples)

                if idx_progress >= total_number_samples-batch_size:
                    break

            if(verbose):
                progress_bar.finish()

            lr_sae = float(K.get_value(self.optimizer_sae.lr))* (1. / (1. + float(K.get_value(self.optimizer_sae.decay)) * float(K.get_value(self.optimizer_sae.iterations)) ))
            
            if str(type(self.optimizer_domain).__name__) == "SGD_AGC":
                lr_domain = 0 
            else:
                lr_domain = float(K.get_value(self.optimizer_domain.lr))* (1. / (1. + float(K.get_value(self.optimizer_domain.decay)) * float(K.get_value(self.optimizer_domain.iterations)) ))
            print(' - Lr(SAE):', lr_sae, ' - Lr(domain):', lr_domain, ' / Lambda:', self.grl_layer.get_hp_lambda())

            if epoch > epochs_pretrain:
                self.grl_layer.increment_hp_lambda_by(self.lambda_inc)

            idx_progress = 0
            print ("Train")
            print (
                    "Loss (SAE): " + str(loss_value_autoencoder) +
                    " Loss (domain): " + str(loss_value_domain) + 
                    " Accuracy (SAE): " + str (accuracy_autoencoder.result()),
                    " Accuracy (domain): " + str(accuracy_domain.result())
                )


            

            if (epoch % super_epochs) == 0:
                print("Source Validation...")
                results_source  = util.evaluateModelListFolders("DANN/" + db_source_name + "-" + db_target_name+ "/" + db_source_name, self.autoencoder, list_files_db1_val, list_files_db1_val_json, windows_shape, batch_size, False, considered_classes)
                assert(len(results_source) == 1)
                pseudo_threshold_source = results_source[0].getPseudoThreshold()
                print("Target Validation...")
                results_target  = util.evaluateModelListFolders("DANN/" + db_source_name + "-" + db_target_name+ "/" + db_target_name, self.autoencoder, list_files_db2_val, list_files_db2_val_json, windows_shape, batch_size, False, considered_classes, pseudo_threshold_source)
                
                acc_source_val, acc_target_val, acc_domain_total = util.evaluateDomainModel(self.domain_classifier, list_files_db1_val, list_files_db2_val, windows_shape, batch_size, False, 0.5)
                
                print('-'*80)
                print ("EPOCH SUMMARY... (epoch %d)" % epoch)
                print("Source " + str(db_source))
                util.printResults(db_source, results_source)

                print("Target " + str(db_target))
                util.printResults(db_target, results_target)

                pseudo_fscore_source = results_source[0].getPseudoFscore()

                if pseudo_fscore_source > best_fscore:
                    print("The model was improved!! (from %.3f in epoch %d to %.3f)" % (best_fscore, best_epoch, pseudo_fscore_source))
                    print("Model saved in " + path_model)
                    self.autoencoder.save(path_model)
                    best_fscore = pseudo_fscore_source
                    best_epoch = epoch
                else:
                    print("The model does not improve the best result (%.3f in epoch %d)" % (best_fscore, best_epoch))

                if epoch > epochs_pretrain:
                    if acc_domain_total.result().numpy() < worst_acc_domain:
                        print("The model with domain stopping was improved!! (from %.3f in epoch %d to %.3f and Fscore from %.3f to %.3f)" % (worst_acc_domain, best_epoch, acc_domain_total.result().numpy(), best_fscore, pseudo_fscore_source))
                        print("Model with domain stopping saved in " + path_model_domain_stopping)
                        self.autoencoder.save(path_model_domain_stopping)
                        worst_acc_domain = acc_domain_total.result().numpy()
                        best_epoch_domain_stopping = epoch
                    else:
                        print("The model with domain stopping does not improve the best result (%.3f in epoch %d)" % (worst_acc_domain, best_epoch_domain_stopping))

            # Reset the metric's state at the end of an epoch
            accuracy_autoencoder.reset_states()
            accuracy_domain.reset_states()

            end = time.time()
            print("Time of epoch: " + str(end - start) + " seconds")
            
        end_train = time.time()
        print("Time of training: " + str(end_train - start_train) + " seconds")
