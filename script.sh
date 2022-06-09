#!/bin/bash

gpu=0

type="cnn"			# type of model. "dann" for domain adaptation and "cnn" for the supervised model (SAE).
window=512          # size of the resized images (squared)
layers=3            # Number of convolutional blocks in SAE
filters=32			# Number of filters
kernel=3			# Kernel size
drop=0.			    # Dropout
e=300				# Maximum number of epochs
se=1                # Superepochs. Default: 1
b=12   				# Batch size
lda=0.01			# Initial value of lambda
lda_inc=0.001		# Increment of lambda per epoch
grl_pos=2           # Position of the GRL connection. Value 0 is the latent code (center of the SAE), negative positions are in the encoder and positive positions in the decoder.
optimizer="sgd"     # Optimizer
pretrain=50         # Number of epochs to pretrain the model with only the source domain.
options=""			# Additional options: --test to evaluate the model
th_iou="0.5"        # In test mode, the IoU minimum to consider the prediction as True Positive.
th=-1               # Threshold used to determine if each pixel can be considered part of the region or background according to the probability obtained by the model. Value -1 explores different values between 0 and 1.
considered_classes=("staff" "empty-staff") # Classes considered for extracting ground-truth data from Muret
verbose=1           # Control the showing of information in console.


options_serial=${options// /.}
options_serial=${options_serial////-}

path="datasets/JSON/Folds/"
fold=0

considered_classes_serial=""
for considered_class in "${considered_classes[@]}" ; do
    considered_classes_serial=""${considered_classes_serial}" "${considered_class}
done


considered_classes_serial_out=""
for considered_class in "${considered_classes[@]}" ; do
    considered_classes_serial_out=${considered_classes_serial_out}"_"${considered_class}
done

for lda in 0.01; do
    for source in "Patriarca" "Guatemala" "Mus-Tradicional/c-combined" "Mus-Tradicional/m-combined" "b-59-850"; do
        for target in "Patriarca" "Guatemala" "Mus-Tradicional/c-combined" "Mus-Tradicional/m-combined" "b-59-850"; do
            for lda_inc in 0.001; do

                if [ $source == $target ]; then
                    continue
                fi

                source_serial=${source// /.}
                source_serial=${source_serial////-}

                target_serial=${target// /.}
                target_serial=${target_serial////-}

                echo ${target_serial}

                output_file="out/out_${type}_${source_serial}-${target_serial}_w${window}_l${layers}_f${filters}_k${kernel}_drop${drop}_lda${lda}_ldainc${lda_inc}_e${e}_b${b}_se${se}${options_serial}_grlpos${grl_pos}_opt${optimizer}_pre${pretrain}_cl${considered_classes_serial_out}.txt"
                echo $output_file

                db1_path_train=""${path}${source}"/fold"${fold}"/train.txt"
                db2_path_train=""${path}${target}"/fold"${fold}"/train.txt"
                
                db1_path_val=""${path}${source}"/fold"${fold}"/val.txt"
                db2_path_val=""${path}${target}"/fold"${fold}"/val.txt"

                echo ${considered_classes_serial}

		    	python -u main.py -type ${type} \
								        -db1_name ${source} \
                                        -db2_name ${target} \
                                        -db1_train ${db1_path_train} \
                                        -db2_train ${db2_path_train} \
                                        -db1_val ${db1_path_val} \
                                        -db2_val ${db2_path_val} \
                                        -classes ${considered_classes_serial} \
								        -w ${window} \
								        -l ${layers} -f ${filters} -k ${kernel} -drop ${drop} \
								        -lda ${lda} \
				                        -lda_inc ${lda_inc} \
								        -e ${e} -b ${b} -se ${se} \
								        -gpu ${gpu} \
                                        -gpos ${grl_pos} \
                                        -opt ${optimizer} \
                                        -pre ${pretrain} \
                                        -th_iou ${th_iou} \
								        ${options} \
								        &> ${output_file}
            done        
        done
    done
done

