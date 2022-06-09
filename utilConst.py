# -*- coding: utf-8 -*-

X_SUFIX = '_GR'
Y_SUFIX = '_GT'

WEIGHTS_CNN_FOLDERNAME = 'WEIGHTS_CNN'
WEIGHTS_DANN_FOLDERNAME = 'WEIGHTS_DANN'

LOGS_CNN_FOLDERNAME = 'LOGS_CNN'
LOGS_DANN_FOLDERNAME = 'LOGS_DANN'
CSV_LOGS_CNN_FOLDERNAME = 'CSV_LOGS_CNN'
CSV_LOGS_DANN_FOLDERNAME = 'CSV_LOGS_DANN'

ARRAY_DBS =[
            'ein-text','ein-staff','ein-notes','ein-ink',
            'sal-text','sal-staff','sal-notes','sal-ink',
            'sal-rec-text','sal-rec-staff','sal-rec-notes','sal-rec-ink',
            'b-59-850-text','b-59-850-staff','b-59-850-notes','b-59-850-ink' 
            ]


FILTER_ENTROPY = 'entropy'
FILTER_WITH_GT_INFO = 'with_gt_info'
FILTER_WITHOUT = 'without'

ARRAY_FILTERS=[
            FILTER_ENTROPY,
            FILTER_WITH_GT_INFO,
            FILTER_WITHOUT
            ]
        
OPT_ADADELTA = 'adadelta'
OPT_SGD = 'sgd'
OPT_ADAM = 'adam'
ARRAY_OPTIMIZERS =[
            OPT_ADADELTA, 
            OPT_SGD,
            OPT_ADAM
            ]

