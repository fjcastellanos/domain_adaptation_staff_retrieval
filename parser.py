
import cv2
import os, sys
from os import listdir
from os.path import isfile, join, exists
import numpy as np
import shutil
import argparse

def nameOfDirFromPath(path_file):
    assert type(path_file) == str
    splits = path_file.split('/')

    dir_path = ''
    is_full_path = False
    for split in splits:
        if ".." in split:
            dir_path = ".."
        else:
            if "." not in split:
                if dir_path == '':
                    if (is_full_path):
                        dir_path = dir_path + "/" + split
                    else:
                        dir_path = split
                        is_full_path = True
                else:
                    dir_path = dir_path + "/" + split

    return dir_path
    
def saveString(content_string, path_file, close_file):
    assert type(content_string) == str
    assert type(path_file) == str
    assert type(close_file) == bool
    
    path_dir = nameOfDirFromPath(path_file)

    if (path_dir != ""):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, 493)
            
    f = open(path_file,"w+")
    f.write(content_string)
    
    if (close_file == True):
        f.close()

def readStringFile(path_file):
    assert type(path_file) == str

    f = open(path_file)
    
    content = f.read()
    f.close()
    
    assert type(content) == str

    return content

def menu():
    parser = argparse.ArgumentParser(description='Parser')

    parser.add_argument('-type',   default='dann', type=str,     choices=['dann', 'cnn', 'autodann', 'dann-test', 'cnn-test'],  help='Training type')

    parser.add_argument('-path',  required=True,   help='path to the file to be parsed')

    args = parser.parse_args()

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args


def extractResults(line):
    line_splitted = line.replace(" ", ";").split(";")
    prec = line_splitted[1]
    recall = line_splitted[3]
    fscore = line_splitted[5]
    threshold = line_splitted[7]
    
    return prec, recall, fscore, threshold


def extractEpoch(line_epoch):
    line_epoch = line_epoch.split("(")
    line_epoch = line_epoch[len(line_epoch) - 1]
    line_epoch = line_epoch.split(")")
    line_epoch = line_epoch[0]
    line_epoch = line_epoch.split(" ")
    epoch = int(line_epoch[len(line_epoch) - 1])
    return epoch


def getResultsDB(data, index):
    dbnames = []
    precs = []
    recalls = []
    fscores = []
    ths = []
    pseudo_precs = []
    pseudo_recalls = []
    pseudo_fscores = []
    pseudo_ths = []


    num_dbs = data[index].count("'") // 2

    index += 1
    for i in range(num_dbs):
        dbname = data[index]
        normal_results = data[index+1]
        pseudo_results = data[index+2]

        prec, recall, fscore, threshold = extractResults(normal_results)
        pseudo_prec, pseudo_recall, pseudo_fscore, pseudo_threshold = extractResults(pseudo_results)

        dbnames.append(dbname)
        precs.append(prec)
        recalls.append(recall)
        fscores.append(fscore)
        ths.append(threshold)
        pseudo_precs.append(pseudo_prec)
        pseudo_recalls.append(pseudo_recall)
        pseudo_fscores.append(pseudo_fscore)
        pseudo_ths.append(pseudo_threshold)

        index += 2

    return dbnames, precs, recalls, fscores, ths, pseudo_precs, pseudo_recalls, pseudo_fscores, pseudo_ths, index



def serializeResults(separator, prec, recall, fscore, threshold, pseudo_prec, pseudo_recall, pseudo_fscore, pseudo_threshold):
    
    serial = ""

    if type(prec) is list:
        assert(len(prec) == len(recall) == len(fscore) == len(threshold) == len(pseudo_prec) == len(pseudo_recall) == len(pseudo_fscore) == len(pseudo_threshold))
    
        for i in range(len(prec)):
            prec_i = prec[i]
            recall_i = recall[i]
            fscore_i = fscore[i]
            threshold_i = threshold[i]
            pseudo_prec_i = pseudo_prec[i]
            pseudo_recall_i = pseudo_recall[i]
            pseudo_fscore_i = pseudo_fscore[i]
            pseudo_threshold_i = pseudo_threshold[i]

            serial +=   str(prec_i) + separator +\
                        str(recall_i) + separator +\
                        str(fscore_i) + separator +\
                        str(threshold_i) + separator +\
                        str(pseudo_prec_i) + separator +\
                        str(pseudo_recall_i) + separator +\
                        str(pseudo_fscore_i) + separator +\
                        str(pseudo_threshold_i) + separator

    else:
        serial +=   str(prec) + separator +\
                    str(recall) + separator +\
                    str(fscore) + separator +\
                    str(threshold) + separator +\
                    str(pseudo_prec) + separator +\
                    str(pseudo_recall) + separator +\
                    str(pseudo_fscore) + separator +\
                    str(pseudo_threshold) + separator

    return serial          



def serializeHeadersSAE(separator, dbname_sources, dbname_targets, dbname_adds):
    
    serial = "Epoch" + separator

    for dbname in dbname_sources:
        serial +=       "PrS-" + str(dbname) + separator + \
                        "ReS-" + str(dbname) + separator + \
                        "F1S-" + str(dbname) + separator + \
                        "ThS-" + str(dbname) + separator +\
                        "pPrS-" + str(dbname) + separator +\
                        "pReS-" + str(dbname) + separator +\
                        "pF1S-" + str(dbname) + separator + \
                        "pThS-" + str(dbname) + separator

    for dbname in dbname_targets:
        serial +=       "PrT-" + str(dbname) + separator + \
                        "ReT-" + str(dbname) + separator + \
                        "F1T-" + str(dbname) + separator + \
                        "ThT-" + str(dbname) + separator + \
                        "pPrT-" + str(dbname) + separator + \
                        "pReT-" + str(dbname) + separator + \
                        "psF1T-" + str(dbname) + separator + \
                        "psThT-" + str(dbname) + separator

    for dbname in dbname_adds:
        serial +=       "PrA-" + str(dbname) + separator + \
                        "ReA-" + str(dbname) + separator + \
                        "F1A-" + str(dbname) + separator + \
                        "ThA-" + str(dbname) + separator + \
                        "pPrA-" + str(dbname) + separator + \
                        "pReA-" + str(dbname) + separator + \
                        "psF1A-" + str(dbname) + separator + \
                        "psThA-" + str(dbname) + separator

    serial +=           "Loss"

    return serial    

def serializeHeaders(separator, dbname_sources, dbname_targets, dbname_adds):
    
    serial = "Epoch" + separator

    for dbname in dbname_sources:
        serial +=       "PrS-" + str(dbname) + separator + \
                        "ReS-" + str(dbname) + separator + \
                        "F1S-" + str(dbname) + separator + \
                        "ThS-" + str(dbname) + separator +\
                        "pPrS-" + str(dbname) + separator +\
                        "pReS-" + str(dbname) + separator +\
                        "pF1S-" + str(dbname) + separator + \
                        "pThS-" + str(dbname) + separator

    for dbname in dbname_targets:
        serial +=       "PrT-" + str(dbname) + separator + \
                        "ReT-" + str(dbname) + separator + \
                        "F1T-" + str(dbname) + separator + \
                        "ThT-" + str(dbname) + separator + \
                        "pPrT-" + str(dbname) + separator + \
                        "pReT-" + str(dbname) + separator + \
                        "psF1T-" + str(dbname) + separator + \
                        "psThT-" + str(dbname) + separator

    for dbname in dbname_adds:
        serial +=       "PrA-" + str(dbname) + separator + \
                        "ReA-" + str(dbname) + separator + \
                        "F1A-" + str(dbname) + separator + \
                        "ThA-" + str(dbname) + separator + \
                        "pPrA-" + str(dbname) + separator + \
                        "pReA-" + str(dbname) + separator + \
                        "psF1A-" + str(dbname) + separator + \
                        "psThA-" + str(dbname) + separator

    serial +=           "LsS" + separator + \
                        "LsD" + separator + \
                        "LrS" + separator + \
                        "LrD" + separator + \
                        "accDS" + separator + \
                        "accDT" + separator + \
                        "accD" + separator + \
                        "lda"

    return serial    



def extractTestResultsLine(line_str):
    prec = line_str.split(";")[0].split(": ")[1]
    recall = line_str.split(";")[1].split(": ")[1]
    fscore = line_str.split(";")[2].split(": ")[1]
    th = line_str.split(";")[3].split(": ")[1]
    return prec, recall, fscore, th

def extractResultsTestAutoDANN(config):
    file_content = readStringFile(config.path)

    file_content_splitted = file_content.split("\n")

    is_header = True
    within_summary = False
    epoch = 0

    separator = ";"

    list_number_samples = []
    list_number_samples_ideal = []
    list_metrics_selection = []

    i = 0
    while i < len(file_content_splitted):
        try:
            if "Counting number of selected samples (SAE or DANN)" in file_content_splitted[i]:
                i+=1
                number_SAE_samples = int(file_content_splitted[i].split(": ")[1])
                i+=1
                number_DANN_samples = int(file_content_splitted[i].split(": ")[1])
                i+=1
                list_number_samples.append((number_SAE_samples, number_DANN_samples))
                i+=2
                number_SAE_samples = int(file_content_splitted[i].split(": ")[1])
                i+=1
                number_DANN_samples = int(file_content_splitted[i].split(": ")[1])
                list_number_samples_ideal.append((number_SAE_samples, number_DANN_samples))
                i+=3
                selection_metrics = file_content_splitted[i]
                selection_metrics = selection_metrics.split(";")
                tp_selection = int(selection_metrics[0].split(":")[1])
                tn_selection = int(selection_metrics[1].split(":")[1])
                fp_selection = int(selection_metrics[2].split(":")[1])
                fn_selection = int(selection_metrics[3].split(":")[1])

                list_metrics_selection.append((tp_selection, tn_selection, fp_selection, fn_selection))

            if "db1=" in file_content_splitted[i]:
                print (file_content_splitted[i] + "->" + file_content_splitted[i+1]+" - " + file_content_splitted[i+2] + ")\n")
                i = i+2
        except:
            i = i+1

        if "Summary:" in file_content_splitted[i]:
            while(True):
                try:
                    if "Test-Correlation threshold:" in file_content_splitted[i]:
                        results = ""  
                        correlation_th = float(file_content_splitted[i].split(": ")[1])
                        i+=3
                        targetdomain = file_content_splitted[i]
                        i+=1
                        prec_target_test, recall_target_test, fscore_target_test, th_target_test = extractTestResultsLine(file_content_splitted[i])
                        i+=1
                        pseudo_prec_target_test, pseudo_recall_target_test, pseudo_fscore_target_test, pseudo_th_target_test = extractTestResultsLine(file_content_splitted[i])
                        i+=3
                        targetdomain_ideal = file_content_splitted[i]
                        assert(targetdomain_ideal == targetdomain)
                        i+=1
                        prec_target_test_ideal, recall_target_test_ideal, fscore_target_test_ideal, th_target_test_ideal = extractTestResultsLine(file_content_splitted[i])
                        i+=1
                        pseudo_prec_target_test_ideal, pseudo_recall_target_test_ideal, pseudo_fscore_target_test_ideal, pseudo_th_target_test_ideal = extractTestResultsLine(file_content_splitted[i])

                        serial_targets_test = serializeResults(separator, prec_target_test, recall_target_test, fscore_target_test, th_target_test, pseudo_prec_target_test, pseudo_recall_target_test, pseudo_fscore_target_test, pseudo_th_target_test)
                        serial_targets_test_ideal = serializeResults(separator, prec_target_test_ideal, recall_target_test_ideal, fscore_target_test_ideal, th_target_test_ideal, pseudo_prec_target_test_ideal, pseudo_recall_target_test_ideal, pseudo_fscore_target_test_ideal, pseudo_th_target_test_ideal)
                        
                        distribution_samples1 = list_number_samples.pop(0)
                        distribution_samples2 = list_number_samples.pop(0)

                        metrics_selection1 = list_metrics_selection.pop(0)
                        metrics_selection2 = list_metrics_selection.pop(0)

                        distribution_samples1_ideal = list_number_samples_ideal.pop(0)
                        distribution_samples2_ideal = list_number_samples_ideal.pop(0)
                        

                        results+= str(correlation_th) + separator
                        results+= serial_targets_test + separator + serial_targets_test_ideal \
                                    + str(distribution_samples1[0]+distribution_samples2[0]) + separator\
                                    + str(distribution_samples1[1]+distribution_samples2[1]) + separator\
                                    + str(distribution_samples1_ideal[0]+distribution_samples2_ideal[0]) + separator\
                                    + str(distribution_samples1_ideal[1]+distribution_samples2_ideal[1]) + separator\
                                    + str(metrics_selection1[0] + metrics_selection2[0]) + separator\
                                    + str(metrics_selection1[1] + metrics_selection2[1]) + separator\
                                    + str(metrics_selection1[2] + metrics_selection2[2]) + separator\
                                    + str(metrics_selection1[3] + metrics_selection2[3])
                        print(results)
                    i+=1
                except:
                    print('*'*80)
                    break
        i+=1


def extractResultsTestSAEDANN(config):
    file_content = readStringFile(config.path)

    file_content_splitted = file_content.split("\n")

    is_header = True
    within_summary = False
    epoch = 0

    separator = ";"

    for i in range(len(file_content_splitted)):

        if "Summary:" in file_content_splitted[i]:

            source_domain = file_content_splitted[i+4]
            target_domain = file_content_splitted[i+9]
            add_domain = file_content_splitted[i+14]

            results_normal_source_val = file_content_splitted[i+5]
            results_pseudo_source_val = file_content_splitted[i+6]

            results_normal_target_val = file_content_splitted[i+10]
            results_pseudo_target_val = file_content_splitted[i+11]
            
            results_normal_add_val = file_content_splitted[i+15]
            results_pseudo_add_val = file_content_splitted[i+16]


            results_normal_source_test = file_content_splitted[i+23]
            results_pseudo_source_test = file_content_splitted[i+24]

            results_normal_target_test = file_content_splitted[i+28]
            results_pseudo_target_test = file_content_splitted[i+29]
            
            results_normal_add_test = file_content_splitted[i+33]
            results_pseudo_add_test = file_content_splitted[i+34]

            prec_source_val, recall_source_val, fscore_source_val, th_source_val = extractTestResultsLine(results_normal_source_val)
            prec_target_val, recall_target_val, fscore_target_val, th_target_val = extractTestResultsLine(results_normal_target_val)
            prec_add_val, recall_add_val, fscore_add_val, th_add_val = extractTestResultsLine(results_normal_add_val)
            
            prec_source_test, recall_source_test, fscore_source_test, th_source_test = extractTestResultsLine(results_normal_source_test)
            prec_target_test, recall_target_test, fscore_target_test, th_target_test = extractTestResultsLine(results_normal_target_test)
            prec_add_test, recall_add_test, fscore_add_test, th_add_test = extractTestResultsLine(results_normal_add_test)
            

            pseudo_prec_source_val, pseudo_recall_source_val, pseudo_fscore_source_val, pseudo_th_source_val = extractTestResultsLine(results_pseudo_source_val)
            pseudo_prec_target_val, pseudo_recall_target_val, pseudo_fscore_target_val, pseudo_th_target_val = extractTestResultsLine(results_pseudo_target_val)
            pseudo_prec_add_val, pseudo_recall_add_val, pseudo_fscore_add_val, pseudo_th_add_val = extractTestResultsLine(results_pseudo_add_val)
            
            pseudo_prec_source_test, pseudo_recall_source_test, pseudo_fscore_source_test, pseudo_th_source_test = extractTestResultsLine(results_pseudo_source_test)
            pseudo_prec_target_test, pseudo_recall_target_test, pseudo_fscore_target_test, pseudo_th_target_test = extractTestResultsLine(results_pseudo_target_test)
            pseudo_prec_add_test, pseudo_recall_add_test, pseudo_fscore_add_test, pseudo_th_add_test = extractTestResultsLine(results_pseudo_add_test)
            

            
            serial_sources_val = serializeResults(separator, prec_source_val, recall_source_val, fscore_source_val, th_source_val, pseudo_prec_source_val, pseudo_recall_source_val, pseudo_fscore_source_val, pseudo_th_source_val)
            serial_targets_val = serializeResults(separator, prec_target_val, recall_target_val, fscore_target_val, th_target_val, pseudo_prec_target_val, pseudo_recall_target_val, pseudo_fscore_target_val, pseudo_th_target_val)
            serial_adds_val = serializeResults(separator, prec_add_val, recall_add_val, fscore_add_val, th_add_val, pseudo_prec_add_val, pseudo_recall_add_val, pseudo_fscore_add_val, pseudo_th_add_val)
            
            serial_sources_test = serializeResults(separator, prec_source_test, recall_source_test, fscore_source_test, th_source_test, pseudo_prec_source_test, pseudo_recall_source_test, pseudo_fscore_source_test, pseudo_th_source_test)
            serial_targets_test = serializeResults(separator, prec_target_test, recall_target_test, fscore_target_test, th_target_test, pseudo_prec_target_test, pseudo_recall_target_test, pseudo_fscore_target_test, pseudo_th_target_test)
            serial_adds_test = serializeResults(separator, prec_add_test, recall_add_test, fscore_add_test, th_add_test, pseudo_prec_add_test, pseudo_recall_add_test, pseudo_fscore_add_test, pseudo_th_add_test)
            
            first_db = "ein"
            second_db = "sal-rec"
            third_db = "b-59-850"

            first_results = ""
            second_results = ""
            third_results = ""

            if first_db in source_domain:
                first_domain = source_domain
                first_results = serial_sources_val + serial_sources_test
                if second_db in target_domain:
                    second_domain = target_domain
                    third_domain = add_domain
                    second_results = serial_targets_val +serial_targets_test
                    third_results = serial_adds_val +serial_adds_test
                else:
                    second_domain = add_domain
                    third_domain = target_domain
                    third_results = serial_targets_val +serial_targets_test
                    second_results = serial_adds_val +serial_adds_test

            elif second_db in source_domain:
                second_domain = source_domain
                second_results = serial_sources_val + serial_sources_test
                if first_db in target_domain:
                    first_domain = target_domain
                    third_domain = add_domain
                    first_results = serial_targets_val +serial_targets_test
                    third_results = serial_adds_val +serial_adds_test
                else:
                    first_domain = add_domain
                    third_domain = target_domain
                    third_results = serial_targets_val +serial_targets_test
                    first_results = serial_adds_val +serial_adds_test
            else:
                third_domain = source_domain
                third_results = serial_sources_val + serial_sources_test
                if first_db in target_domain:
                    first_domain = target_domain
                    second_domain = add_domain
                    first_results = serial_targets_val +serial_targets_test
                    second_results = serial_adds_val +serial_adds_test
                else:
                    second_domain = target_domain
                    first_domain = add_domain
                    second_results = serial_targets_val +serial_targets_test
                    first_results = serial_adds_val +serial_adds_test



            report = str(source_domain) + "->" + target_domain + "\n" +\
                        str(first_domain) + separator + str(second_domain) + separator + str(third_domain) + "\n" +\
                        first_results +\
                        second_results +\
                        third_results + "\n"
            #report = report.replace(".", ",")

            print(report)


def extractResultsSAEDANN(config):

    file_content = readStringFile(config.path)

    file_content_splitted = file_content.split("\n")


    is_header = True
    within_summary = False
    epoch = 0

    separator = ";"

    last_learning_rate_line = ""
    last_tensor_line = ""
    for i in range(len(file_content_splitted)):

        if "tf.Tensor" in file_content_splitted[i]:
            last_learning_rate_line = file_content_splitted[i-2]
            last_tensor_line = file_content_splitted[i]
        
        if "EPOCH SUMMARY..." in file_content_splitted[i]:
            epoch = extractEpoch(file_content_splitted[i])
            last_tensor_split = last_tensor_line.replace("tf.Tensor(", "").replace("shape=(), dtype=float32)", "").replace(", ", ":").replace(" Accuracy", ":Accuracy").replace(" ", "").split(":")
            #last_tensor_split = last_tensor_line.split("tf.Tensor(")
            loss_sae = float(last_tensor_split[1])
            loss_domain = float(last_tensor_split[3])

            lr_split = last_learning_rate_line.replace(" - ", ":").replace("/", ":").replace(" ", "").split(":")
            lr_sae = float(lr_split[2])
            lr_domain = float(lr_split[4])
            lda = float(lr_split[6])

            acc_domain_line = file_content_splitted[i-3]
            acc_domain_line_splitted = acc_domain_line.replace("Acc", ":").replace(" ", "").split(":")
            
            acc_domain_source = acc_domain_line_splitted[2]
            acc_domain_target = acc_domain_line_splitted[4]
            acc_domain_avg = acc_domain_line_splitted[6]
            

            dbname_sources, prec_sources, recall_sources, fscore_sources, threshold_sources, pseudo_prec_sources, pseudo_recall_sources, pseudo_fscore_sources, pseudo_threshold_sources, i = getResultsDB(file_content_splitted, i+1)
            dbname_targets, prec_targets, recall_targets, fscore_targets, threshold_targets, pseudo_prec_targets, pseudo_recall_targets, pseudo_fscore_targets, pseudo_threshold_targets, i = getResultsDB(file_content_splitted, i+1)
            dbname_adds, prec_adds, recall_adds, fscore_adds, threshold_adds, pseudo_prec_adds, pseudo_recall_adds, pseudo_fscore_adds, pseudo_threshold_adds, i = getResultsDB(file_content_splitted, i+1)

            serial_sources = serializeResults(separator, prec_sources, recall_sources, fscore_sources, threshold_sources, pseudo_prec_sources, pseudo_recall_sources, pseudo_fscore_sources, pseudo_threshold_sources)
            serial_targets = serializeResults(separator, prec_targets, recall_targets, fscore_targets, threshold_targets, pseudo_prec_targets, pseudo_recall_targets, pseudo_fscore_targets, pseudo_threshold_targets)
            serial_adds = serializeResults(separator, prec_adds, recall_adds, fscore_adds, threshold_adds, pseudo_prec_adds, pseudo_recall_adds, pseudo_fscore_adds, pseudo_threshold_adds)
            
            if is_header:
                report = serializeHeaders(separator, dbname_sources, dbname_targets, dbname_adds)
                print(report)
                is_header = False
            
            report = str(epoch) + separator +\
                        serial_sources +\
                        serial_targets +\
                        serial_adds +\
                        str(loss_sae) + separator +\
                        str(loss_domain) + separator +\
                        str(lr_sae) + separator +\
                        str(lr_domain) + separator +\
                        str(acc_domain_source) + separator +\
                        str(acc_domain_target) + separator +\
                        str(acc_domain_avg) + separator +\
                        str(lda)
            #report = report.replace(".", ",")

            print(report)


def extractResultsSAE(config):

    file_content = readStringFile(config.path)

    file_content_splitted = file_content.split("\n")


    is_header = True
    within_summary = False
    epoch = 0

    separator = ";"

    last_learning_rate_line = ""
    last_tensor_line = ""
    for i in range(len(file_content_splitted)):

        if "Loss:" in file_content_splitted[i]:
            last_loss = float(file_content_splitted[i].replace(" ", ":").split(":")[2])
        
        if "EPOCH SUMMARY..." in file_content_splitted[i]:
            epoch = extractEpoch(file_content_splitted[i])
            
            dbname_sources, prec_sources, recall_sources, fscore_sources, threshold_sources, pseudo_prec_sources, pseudo_recall_sources, pseudo_fscore_sources, pseudo_threshold_sources, i = getResultsDB(file_content_splitted, i+1)
            dbname_targets, prec_targets, recall_targets, fscore_targets, threshold_targets, pseudo_prec_targets, pseudo_recall_targets, pseudo_fscore_targets, pseudo_threshold_targets, i = getResultsDB(file_content_splitted, i+1)
            dbname_adds, prec_adds, recall_adds, fscore_adds, threshold_adds, pseudo_prec_adds, pseudo_recall_adds, pseudo_fscore_adds, pseudo_threshold_adds, i = getResultsDB(file_content_splitted, i+1)

            serial_sources = serializeResults(separator, prec_sources, recall_sources, fscore_sources, threshold_sources, pseudo_prec_sources, pseudo_recall_sources, pseudo_fscore_sources, pseudo_threshold_sources)
            serial_targets = serializeResults(separator, prec_targets, recall_targets, fscore_targets, threshold_targets, pseudo_prec_targets, pseudo_recall_targets, pseudo_fscore_targets, pseudo_threshold_targets)
            serial_adds = serializeResults(separator, prec_adds, recall_adds, fscore_adds, threshold_adds, pseudo_prec_adds, pseudo_recall_adds, pseudo_fscore_adds, pseudo_threshold_adds)
            
            if is_header:
                report = serializeHeadersSAE(separator, dbname_sources, dbname_targets, dbname_adds)
                print(report)
                is_header = False
            
            report = str(epoch) + separator +\
                        serial_sources +\
                        serial_targets +\
                        serial_adds +\
                        str(last_loss)
                        
            #report = report.replace(".", ",")

            print(report)


def extractResultsSAE2(config):
    file_content = readStringFile(config.path)

    file_content_splitted = file_content.split("\n")


    contador = 0
    within_summary = False
    epoch = 0

    separator = ";"

    report = \
            "Epoch" + separator + \
            "PrS" + separator + \
            "ReS"+ separator + \
            "F1S" + separator + \
            "ThS" + separator +\
            "pPrS" + separator +\
            "pReS" + separator +\
            "pF1S" + separator + \
            "pThS" + separator + \
            "PrT" + separator + \
            "ReT" + separator + \
            "F1T" + separator + \
            "ThT" + separator + \
            "pPrT" + separator + \
            "pReT" + separator + \
            "psF1T" + separator + \
            "psThT" + separator + \
            "LsS"
    print(report)

    for i in range(len(file_content_splitted)):
        if "Loss:" in file_content_splitted[i]:
            line_loss_splitted = file_content_splitted[i].split(" ")
            loss_source = float(line_loss_splitted[1])

        if "EPOCH SUMMARY..." in file_content_splitted[i]:
            
            epoch = extractEpoch(file_content_splitted[i])

            prec_source, recall_source, fscore_source, threshold_source = extractResultsLineSAE(file_content_splitted[i+2])
            pseudo_prec_source, pseudo_recall_source, pseudo_fscore_source, pseudo_threshold_source = extractResultsLineSAE(file_content_splitted[i+3])

            prec_target, recall_target, fscore_target, threshold_target = extractResultsLineSAE(file_content_splitted[i+5])
            pseudo_prec_target, pseudo_recall_target, pseudo_fscore_target, pseudo_threshold_target = extractResultsLineSAE(file_content_splitted[i+6])

            i += 6
            
            report = str(epoch) + separator + \
                    str(prec_source) + separator + str(recall_source) + separator + str(fscore_source) + separator + str(threshold_source) + separator +\
                    str(pseudo_prec_source) + separator + str(pseudo_recall_source) + separator + str(pseudo_fscore_source) + separator + str(pseudo_threshold_source) + separator +\
                    str(prec_target) + separator + str(recall_target) + separator + str(fscore_target) + separator + str(threshold_target) + separator +\
                    str(pseudo_prec_target) + separator + str(pseudo_recall_target) + separator + str(pseudo_fscore_target) + separator + str(pseudo_threshold_target) + separator + \
                    str(loss_source)
            #report = report.replace(".", ",")

            print(report)
            

def extractResultsLineSAE(line):
    line_splitted = line.split(" ")
    precision = line_splitted[1]
    recall = line_splitted[3]
    f1 = line_splitted[5]
    threshold = line_splitted[7]

    return precision, recall, f1, threshold


if __name__ == "__main__":
    config = menu()

    if config.type == "dann":
        extractResultsSAEDANN(config)
    elif config.type == "cnn":
        extractResultsSAE(config)
    elif config.type == "dann-test":
        extractResultsTestSAEDANN(config)
    elif config.type == "cnn-test":
        extractResultsTestSAEDANN(config)
    elif config.type == "autodann":
        extractResultsTestAutoDANN(config)
    else:
        pass

    