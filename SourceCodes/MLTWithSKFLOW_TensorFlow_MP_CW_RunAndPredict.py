import numpy as np
import sys
import warnings
from scipy import interp
import pylab as pl
import pandas as pd
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from sklearn import metrics
import multiprocessing
from sklearn.utils import class_weight
# from sklearn.externals import joblib
# import pickle
# import dill

# Not show unnecessary Warning filter
warnings.filterwarnings("always")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ResourceWarning)




class Execute_machine_learning():

    # this function will generate prediction(input) and outcome(output) variable for main dataset
    def data_generation(self,csv_path):

        # This is used to get the actual path of csv file
        self.csv_path = csv_path

        data_table = pd.read_csv(self.csv_path)

        last_index_column = len(data_table.columns)

        pred_data = data_table[data_table.columns[1: last_index_column]].values
        outcome_data = data_table[data_table.columns[0]].values

        print("Inside data generator........")

        return pred_data, outcome_data

    # this function will generate prediction(input) and outcome(output) variable for independent dataset
    def data_generation_independent(self,csv_path):

        # This is used to get the actual path of csv file
        self.csv_path = csv_path

        data_table = pd.read_csv(self.csv_path)

        last_index_column = len(data_table.columns)

        pred_data = data_table[data_table.columns[1: last_index_column]].values
        outcome_data = data_table[data_table.columns[0]].values

        print("Inside independent data generator........")

        return pred_data, outcome_data

    def parameter_tuning_LR(self, X, y, nfolds, cls_wgt):
        self.nfolds = nfolds
        parameter_LR = {'penalty': ['l1', 'l2'], 'C': [1, 10, 100, 1000]}

        lr_grid_search = GridSearchCV(estimator=LogisticRegression(random_state=42, class_weight=cls_wgt),
                                      cv=self.nfolds, param_grid=parameter_LR, n_jobs=-1)

        print(cls_wgt)
        lr_grid_search.fit(X, y)

        fileforLR = open("LR_BestParameterAndAccuracy.txt", "w+")

        fileforLR.write("Best accuracy score achieved = {} \n".format(lr_grid_search.best_score_))
        fileforLR.write("Best Parameters: {} ".format(lr_grid_search.best_params_))
        fileforLR.close()

        print("Inside LR parameter tuning........")

        return lr_grid_search.best_estimator_

    def parameter_tuning_KNN(self, X, y, nfolds):
        self.nfolds = nfolds
        parameter_knn = {'n_neighbors': [5, 10, 15, 20], 'weights': ['uniform', 'distance'], 'p': [1, 2]}

        knn_grid_search = GridSearchCV(estimator=KNeighborsClassifier(n_jobs=-1), cv=self.nfolds, param_grid=parameter_knn)

        knn_grid_search.fit(X,y)


        fileforKNN = open("KNN_BestParameterAndAccuracy.txt", "w+")

        fileforKNN.write("Best accuracy score achieved = {} \n".format(knn_grid_search.best_score_))
        fileforKNN.write("Best Parameters: {} ".format(knn_grid_search.best_params_))
        fileforKNN.close()

        print("Inside KNN parameter tuning........")

        return knn_grid_search.best_estimator_

    def parameter_tuning_DT(self, X, y, nfolds, cls_wgt):
        self.nfolds = nfolds
        parameter_dt = {'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 3, 4],
                        'min_samples_leaf': [1, 2, 3]}

        dt_grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42, class_weight=cls_wgt),
                                      cv=self.nfolds, param_grid=parameter_dt, n_jobs=-1)

        dt_grid_search.fit(X, y)

        fileforDT = open("DT_BestParameterAndAccuracy.txt", "w+")

        fileforDT.write("Best accuracy score achieved = {} \n".format(dt_grid_search.best_score_))
        fileforDT.write("Best Parameters: {} ".format(dt_grid_search.best_params_))
        fileforDT.close()

        print("Inside DT parameter tuning........")

        return dt_grid_search.best_estimator_

    def parameter_tuning_SVM(self, X, y, nfolds, cls_wgt):
        self.nfolds = nfolds
        parameter_svm = [
            # {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            # {'C': [1, 10, 100, 1000], 'kernel': ['poly']},
            {'C': [1, 10, 100, 1000], 'gamma': ["auto", 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
            # {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid']}
        ]

        svm_grid_search = GridSearchCV(estimator=SVC(probability=True, random_state=42, class_weight=cls_wgt),
                                       cv=self.nfolds, param_grid=parameter_svm, n_jobs=-1)

        svm_grid_search.fit(X, y)

        # print("Best score achieved:", svm_grid_search.best_score_)
        # print("Best Parameters:", svm_grid_search.best_params_)
        # print("Scores:", svm_grid_search.best_estimator_)

        fileforSVM = open("SVM_BestParameterAndAccuracy.txt", "w+")

        fileforSVM.write("Best accuracy score achieved = {} \n".format(svm_grid_search.best_score_))
        fileforSVM.write("Best Parameters: {} ".format(svm_grid_search.best_params_))
        fileforSVM.close()

        print("Inside SVM parameter tuning........")

        return svm_grid_search.best_estimator_

    def parameter_tuning_RF(self, X, y, nfolds, cls_wgt):
        self.nfolds = nfolds
        parameter_rf = {'bootstrap': [True, False],
                        'max_depth': [None, 90, 100],
                        'max_features': ["auto", 2, 3],
                        'min_samples_leaf': [1, 2, 3],
                        'min_samples_split': [2, 4, 8],
                        'n_estimators': [10, 100, 200, 300, 500]}

        rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42, class_weight=cls_wgt),
                                      cv=self.nfolds, param_grid=parameter_rf, n_jobs=-1)

        rf_grid_search.fit(X, y, class_weight=cls_wgt)

        fileforRF = open("RF_BestParameterAndAccuracy.txt", "w+")

        fileforRF.write("Best accuracy score achieved = {} \n".format(rf_grid_search.best_score_))
        fileforRF.write("Best Parameters: {} ".format(rf_grid_search.best_params_))
        fileforRF.close()

        print("Inside RF parameter tuning........")

        return rf_grid_search.best_estimator_

    def parameter_tuning_DNN(self, X, y, nfolds):
        self.nfolds = nfolds
        parameter_dnn = {"hidden_units": [[10, 20, 10], [20, 40, 20], [50, 100, 50], [10, 20, 20, 10]],
                         "learning_rate": [0.1, 0.2],
                         "batch_size": [10, 32]}

        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X)

        classifier_DNN = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                        hidden_units=[10, 20, 10], n_classes=2,
                                                        config=tf.contrib.learn.RunConfig(tf_random_seed=232))

        dnn_grid_search = GridSearchCV(classifier_DNN, cv=self.nfolds, scoring="accuracy",
                                       param_grid=parameter_dnn, fit_params={'steps': [200, 400]})

        model = dnn_grid_search.fit(X, y)

        return dnn_grid_search.best_estimator_

    def performance_measure(self, X, y, model, nfolds, modelIndex, independent_pred_var, independent_outcome_var):
        self.nfolds = nfolds
        self.modelIndex = modelIndex

        sen_list = []
        spec_list = []
        acc_list = []
        pre_list = []
        mcc_list = []
        f1_list = []

        tpr_list = []
        mean_fpr = np.linspace(0, 1, 100)
        # auc_list = []

        # this is for independent dataset
        sen_list_independent = []
        spec_list_independent = []
        acc_list_independent = []
        pre_list_independent = []
        mcc_list_independent = []
        f1_list_independent = []

        tpr_list_independent = []
        mean_fpr_independent = np.linspace(0, 1, 100)


        skf = StratifiedKFold(n_splits = self.nfolds, random_state=423)

        for train_index, test_index in skf.split(X,y):
            # print("Train:", train_index, "Test:", test_index)
            if modelIndex == 6:
                probability_model = model.fit(X[train_index], y[train_index], steps=2000).predict_proba(
                    X[test_index], as_iterable=False)
                prediction_model = model.fit(X[train_index], y[train_index], steps=2000).predict(
                    X[test_index], as_iterable=False)

                fpr, tpr, thresholds = metrics.roc_curve(y[test_index], probability_model[:, 1])

                tpr_list.append(interp(mean_fpr, fpr, tpr))
                tpr_list[-1][0] = 0.0

                conf_matrix = metrics.confusion_matrix(y[test_index], prediction_model)

                # this is use to predict independent dataset
                probability_model_independent = model.fit(X[train_index], y[train_index], steps=2000).predict_proba(
                    independent_pred_var, as_iterable=False)
                prediction_model_independent = model.fit(X[train_index], y[train_index], steps=2000).predict(
                    independent_pred_var, as_iterable=False)
                fpr_independent, tpr_independent, thresholds_independent = metrics.roc_curve(independent_outcome_var, probability_model_independent[:, 1])

                tpr_list_independent.append(interp(mean_fpr_independent, fpr_independent, tpr_independent))
                tpr_list_independent[-1][0] = 0.0

                conf_matrix_independent = metrics.confusion_matrix(independent_outcome_var, prediction_model_independent)


            else:
                probability_model = model.fit(X[train_index], y[train_index]).predict_proba(X[test_index])
                prediction_model = model.fit(X[train_index], y[train_index]).predict(X[test_index])

                # this use to predict independent dataset
                probability_model_independent = model.fit(X[train_index], y[train_index]).predict_proba(independent_pred_var)
                prediction_model_independent = model.fit(X[train_index], y[train_index]).predict(independent_pred_var)


                fpr, tpr, thresholds = metrics.roc_curve(y[test_index], probability_model[:, 1])

                tpr_list.append(interp(mean_fpr, fpr, tpr))
                tpr_list[-1][0] = 0.0

                conf_matrix = metrics.confusion_matrix(y[test_index], prediction_model)

                # this is for independent dataset
                fpr_independent, tpr_independent, thresholds_independent = metrics.roc_curve(independent_outcome_var, probability_model_independent[:, 1])

                tpr_list_independent.append(interp(mean_fpr_independent, fpr_independent, tpr_independent))
                tpr_list_independent[-1][0] = 0.0

                conf_matrix_independent = metrics.confusion_matrix(independent_outcome_var, prediction_model_independent)

            new_list_CM = []

            for i in conf_matrix:
                for j in i:
                    new_list_CM.append(j)

            TP = float(new_list_CM[0])
            FP = float(new_list_CM[1])
            FN = float(new_list_CM[2])
            TN = float(new_list_CM[3])

            # print("TP:", TP, "FP:", FP, "FN:", FN,"TN:", TN)
            try:
                sensitivity = round(float(TP / (TP + FN)), 2)
            except:
                print("Error in sensitivity")
                pass
            try:
                specificity = round(float(TN / (TN + FP)), 2)
            except:
                print("Error in specificity")
                pass
            try:
                accuracy = round(float((TP + TN) / (TP + FP + FN + TN)), 2)
            except:
                print("Error in accuracy")
                pass
            try:
                precision = round(float(TP / (TP + FP)), 2)
            except:
                print("Error in precision")
                pass
            try:
                mcc = round(metrics.matthews_corrcoef(y[test_index], prediction_model), 2)
            except:
                print("Error in mcc")
                pass
            try:
                # f1 = round(metrics.f1_score(y[test_index], prediction_model), 2)
                f1 = 2 * ((sensitivity * precision)/(sensitivity + precision))
            except:
                print("Error in f1")
                pass

            # store the value in list of performance measure
            sen_list.append(sensitivity)
            spec_list.append(specificity)
            acc_list.append(accuracy)
            pre_list.append(precision)
            mcc_list.append(mcc)
            f1_list.append(f1)

            # this is for independent dataset
            new_list_CM_independent = []
            for i in conf_matrix_independent:
                for j in i:
                    new_list_CM_independent.append(j)

            TP_independent = float(new_list_CM_independent[0])
            FP_independent = float(new_list_CM_independent[1])
            FN_independent = float(new_list_CM_independent[2])
            TN_independent = float(new_list_CM_independent[3])
            # print("TP_Independent:", TP_independent, "FP_Independent:", FP_independent, "FN_Independent:", FN_independent,"TN_Independent:", TN_independent)
            try:
                sensitivity_independent = round(float(TP_independent / (TP_independent + FN_independent)), 2)
            except:
                print("Error in sensitivity_independent")
                pass
            try:
                specificity_independent = round(float(TN_independent / (TN_independent + FP_independent)), 2)
            except:
                print("Error in specificity_independent")
                pass
            try:
                accuracy_independent = round(float((TP_independent + TN_independent) / (TP_independent + FP_independent + FN_independent + TN_independent)), 2)
            except:
                print("Error in accuracy_independent")
                pass
            try:
                precision_independent = round(float(TP_independent / (TP_independent + FP_independent)), 2)
            except:
                print("Error in precision_independent")
                pass
            try:
                mcc_independent = round(metrics.matthews_corrcoef(independent_outcome_var, prediction_model_independent), 2)
            except:
                print("Error in mcc_independent")
                pass
            try:
                # f1 = round(metrics.f1_score(y[test_index], prediction_model), 2)
                f1_independent = 2 * ((sensitivity_independent * precision_independent)/(sensitivity_independent + precision_independent))
            except:
                print("Error in f1_independent")
                pass

            # store the value in list of performance measure
            sen_list_independent.append(sensitivity_independent)
            spec_list_independent.append(specificity_independent)
            acc_list_independent.append(accuracy_independent)
            pre_list_independent.append(precision_independent)
            mcc_list_independent.append(mcc_independent)
            f1_list_independent.append(f1_independent)

        sen_mean = round(float(sum(sen_list))/float(len(sen_list)),3)
        spec_mean = round(float(sum(spec_list))/float(len(spec_list)),3)
        acc_mean = round(float(sum(acc_list))/float(len(acc_list)),3)
        pre_mean = round(float(sum(pre_list))/float(len(pre_list)),3)
        mcc_mean = round(float(sum(mcc_list))/float(len(mcc_list)),3)
        f1_mean = round(float(sum(f1_list))/float(len(f1_list)),3)


        mean_tpr = np.mean(tpr_list, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr,mean_tpr)

        # this is for independent dataset
        sen_mean_independent = round(float(sum(sen_list_independent)) / float(len(sen_list_independent)), 3)
        spec_mean_independent = round(float(sum(spec_list_independent)) / float(len(spec_list_independent)), 3)
        acc_mean_independent = round(float(sum(acc_list_independent)) / float(len(acc_list_independent)), 3)
        pre_mean_independent = round(float(sum(pre_list_independent)) / float(len(pre_list_independent)), 3)
        mcc_mean_independent = round(float(sum(mcc_list_independent)) / float(len(mcc_list_independent)), 3)
        f1_mean_independent = round(float(sum(f1_list_independent)) / float(len(f1_list_independent)), 3)

        mean_tpr_independent = np.mean(tpr_list_independent, axis=0)
        mean_tpr_independent[-1] = 1.0
        mean_auc_independent = metrics.auc(mean_fpr_independent, mean_tpr_independent)


        perf_header = ("sensitivity", "specificity", "accuracy", "precision", "mcc", "f1", "auc")
        perf_value = (sen_mean, spec_mean, acc_mean, pre_mean, mcc_mean, f1_mean, round(mean_auc,3))

        # this is for independent dataset
        perf_header_independent = ("sensitivity_independent", "specificity_independent", "accuracy_independent", "precision_independent", "mcc_independent", "f1_independent", "auc_independent")
        perf_value_independent = (sen_mean_independent, spec_mean_independent, acc_mean_independent, pre_mean_independent, mcc_mean_independent, f1_mean_independent, round(mean_auc_independent, 3))
        # print("Header:",perf_header, "Value:", perf_value)

        print("Inside performance measures........")
        # print(model_list)


        return perf_header, perf_value, mean_tpr, mean_fpr, perf_header_independent, perf_value_independent, mean_tpr_independent, mean_tpr_independent


    def main_program(self, input_file, no_fold, independent_file, input_hidden_Layer):
        self.input_file = input_file
        self.no_fold = no_fold
        self.independent_file = independent_file

        strNoOfHiddenLayers = "".join(str(input_hidden_Layer))

        # this will used to get prediction (input) and outcome (output) variable of main dataset
        pred_var, outcome_var = Execute_machine_learning().data_generation(self.input_file)

        # this will used to get prediction (input) and outcome (output) variable of independent dataset
        pred_var_independent, outcome_var_independent = Execute_machine_learning().data_generation_independent(self.independent_file)

        # print("Independent_Pred_var::",pred_var_independent)
        # print("Independent_Outcome_var::", outcome_var_independent)

        class_weight_values = class_weight.compute_class_weight('balanced', np.unique(outcome_var), outcome_var)
        # data into dictionary format..
        class_weights = dict(zip(np.unique(outcome_var), class_weight_values))

        perfromance_values = []

        tpr_list = []
        fpr_list = []

        # this is for independent dataset
        perfromance_values_independent = []

        tpr_list_independent = []
        fpr_list_independent = []



        LR = Execute_machine_learning().parameter_tuning_LR(pred_var, outcome_var, self.no_fold, class_weights)
        # GuassianNB does not accept parameters, except priors parameter.
        GNB = GaussianNB()
        KNN = Execute_machine_learning().parameter_tuning_KNN(pred_var, outcome_var, self.no_fold)
        DT = Execute_machine_learning().parameter_tuning_DT(pred_var, outcome_var, self.no_fold, class_weights)
        SVM = Execute_machine_learning().parameter_tuning_SVM(pred_var, outcome_var, self.no_fold, class_weights)
        RF = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weights)

        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(pred_var)

        DNN = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=input_hidden_Layer,
                                             n_classes=2, config=tf.contrib.learn.RunConfig(tf_random_seed=234))

        classifiers = (LR, GNB, KNN, DT, SVM, RF, DNN)

        for i, classifier in enumerate(classifiers):
            performance_header, performance_value, tpr, fpr, performance_header_independent, performance_value_independent, tpr_independent, fpr_independent = Execute_machine_learning().\
                performance_measure(pred_var, outcome_var, classifier, self.no_fold, i, pred_var_independent, outcome_var_independent)
            perfromance_values.append(performance_value)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

            # this for independent dataset
            perfromance_values_independent.append(performance_value_independent)
            tpr_list_independent.append(tpr_independent)
            fpr_list_independent.append(fpr_independent)


        performance_result = pd.DataFrame(perfromance_values, index=("LR", "GNB", "KNN", "DT", "SVM", "RF", "DNN"), columns=performance_header)

        performance_result.plot.bar()
        pl.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol = 4)
        pl.savefig("CW_Performance_summary_" + str(self.no_fold) + "foldCV_" + strNoOfHiddenLayers + "_" + str(input_file).replace(".csv", ".png"))

        # pl.savefig("TestPerforamnec_SKflow_summary.png")
        pl.close()

        performance_result.to_csv("CW_Performance_summary_" + str(self.no_fold) + "foldCV_" + strNoOfHiddenLayers + "_" + str(input_file))
        # performance_result.to_csv("TestPerforamnec_SKflow.csv")

        model_short_name = ("LR", "NB", "KNN", "DT", "SVM", "RF", "DNN")

        pl.plot([0, 1], [0, 1], '--', lw=2, label="Random(AUC = 0.5)")

        for i, tpr_value in enumerate(tpr_list):
            mean_auc = metrics.auc(fpr_list[i], tpr_value)
            pl.plot(fpr_list[i], tpr_value, '-', label=model_short_name[i] + '(AUC = %0.2f)' % mean_auc, lw=2)

        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC Curve')
        pl.legend(loc="lower right")
        pl.savefig("CW_CombineRocCurve_" + strNoOfHiddenLayers + "_" + str(input_file).replace(".csv", ".png"))
        # pl.savefig("Test_SKflowRocCurve.png")
        pl.close()

        # this is for independent dataset
        performance_result_independent = pd.DataFrame(perfromance_values_independent, index=("LR", "GNB", "KNN", "DT", "SVM", "RF", "DNN"),
                                          columns=performance_header_independent)

        performance_result_independent.plot.bar()
        pl.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
        pl.savefig("CW_Performance_summary_" + str(self.no_fold) + "foldCV_" + strNoOfHiddenLayers + "_" + str(independent_file).replace(".csv", ".png"))
        pl.close()

        performance_result_independent.to_csv("CW_Performance_summary_" + str(self.no_fold) + "foldCV_" + strNoOfHiddenLayers + "_" + str(independent_file))

        model_short_name_independent = ("LR", "NB", "KNN", "DT", "SVM", "RF", "DNN")

        pl.plot([0, 1], [0, 1], '--', lw=2, label="Random(AUC = 0.5)")

        for i, tpr_value_independent in enumerate(tpr_list):
            mean_auc_independent = metrics.auc(fpr_list_independent[i], tpr_value_independent)
            pl.plot(fpr_list_independent[i], tpr_value_independent, '-', label=model_short_name_independent[i] + '(AUC = %0.2f)' % mean_auc_independent, lw=2)

        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC Curve')
        pl.legend(loc="lower right")
        pl.savefig("CW_CombineRocCurve_" + strNoOfHiddenLayers + "_" + str(independent_file).replace(".csv", ".png"))
        pl.close()

if __name__=="__main__":
    import argparse

    multiprocessing.set_start_method('forkserver', force=True)
    parser = argparse.ArgumentParser()

    # get arguments from command line
    parser.add_argument("-f", "--filepath",
                        required=True,
                        default=None,
                        help="Path to target CSV file")
    parser.add_argument("-n", "--n_folds",
                        required=None,
                         default=5,
                        help="n_folds for Cross Validation")
    parser.add_argument("-i", "--independent_file",
                        required=None,
                        default=None,
                        help="Path to target CSV file")
    parser.add_argument("-l", "--list",
                        required=True,
                        nargs="*",
                        type=int,
                        default=[10, 20, 10],
                        help="Hidden Layer for DNN")

    # parse the argument from command line
    args = parser.parse_args()
    # Create instance of class
    objEML = Execute_machine_learning()
    # pass the command line argument to the method
    objEML.main_program(args.filepath, int(args.n_folds), args.independent_file, args.list)

    exit()
