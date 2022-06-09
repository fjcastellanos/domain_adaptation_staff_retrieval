


class Results:

    def __init__(self, precision, recall, fscore, threshold, pseudo_precision, pseudo_recall, pseudo_fscore, pseudo_threshold):
        self.precision = precision
        self.recall = recall
        self.fscore = fscore
        self.threshold = threshold

        self.pseudo_precision = pseudo_precision
        self.pseudo_recall = pseudo_recall
        self.pseudo_fscore = pseudo_fscore
        self.pseudo_threshold = pseudo_threshold


    def getPseudoFscore(self):
        return self.pseudo_fscore

    def getPseudoThreshold(self):
        return self.pseudo_threshold

        
    def __repr__(self):
        print ("Precision: %.3f" % self.precision, "Recall: %.3f" % self.recall, "Fscore: %.3f" % self.fscore, "Threshold: %.3f" % self.threshold)
        print ("Pseudo-Precision: %.3f" % self.pseudo_precision, "Pseudo-Recall: %.3f" % self.pseudo_recall, "Pseudo-Fscore: %.3f" % self.pseudo_fscore, "Pseudo-Threshold: %.3f" % self.pseudo_threshold)
        
    def __str__(self):
        s = "Precision: %.3f;Recall: %.3f;Fscore: %.3f;Threshold: %.3f\nPseudo-Precision: %.3f;Pseudo-Recall: %.3f;Pseudo-Fscore: %.3f;Pseudo-Threshold: %.3f" % (self.precision, self.recall, self.fscore, self.threshold, self.pseudo_precision, self.pseudo_recall, self.pseudo_fscore, self.pseudo_threshold)
        return s