# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:48:49 2019

@author: User Ambev
"""
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tqdm


class eval_multiclass():
    
    def __init__(self,model):
        self.model = model
    
    def get_preds(self, X_test):
        self.preds = self.model.predict(X_test).flatten()
        self.prob_preds = self.model.predict_proba(X_test)
        return {'preds':self.preds, 'prob_preds':self.prob_preds}
        
    def classification_report(self, y_test):
        class_report = classification_report(y_test, self.preds, output_dict = True)
        class_report_df = pd.DataFrame(class_report).transpose()
        return class_report_df
    
    def check_df(self, X_test, y_test):
        self.y_test = y_test
        self.X_test = X_test
        if not all([attr in dir(self) for attr in ['preds','prob_preds']]):
            self.get_preds(X_test)
        
        def arg_val_max(arr, n):
            z = arr.argsort()[-n:][::-1]
            return tuple(zip(z,arr[z]))

        #classes = y_test).flatten()
        class_prob = [arg_val_max(self.prob_preds[row], 3) for row in range(self.prob_preds.shape[0])]
        
        max_probs = [class_prob[i][0][1] for i in range(self.prob_preds.shape[0])]
        self.check = check = self.preds == np.array(y_test).flatten()
        ground_truth = np.array(y_test).flatten()

        check_prob_df = pd.DataFrame([max_probs,check, ground_truth, self.preds]).T
        check_prob_df.columns = ['max_prob','got_right','ground_truth','preds']
        self.check_prob_df = check_prob_df
        return
            
    def iterate_over_threshold(self, n_points = 301):

        
        self.threshold_arr = threshold_arr = np.linspace(0,1,n_points)
        thresh_dict = {}
        for threshold in tqdm.tqdm(threshold_arr):

            thresh_msk = self.prob_preds.max(axis = 1) >= threshold
            preds = self.preds[thresh_msk]
            y_test = self.y_test[thresh_msk]
            if thresh_msk.sum() > 0:
                thresh_dict[threshold] = classification_report(preds,y_test, output_dict = True, zero_division = 1) 
        
        self.thresh_dict = thresh_dict
        return

    def fit(self, X_test, y_test, n_points = 301):
        self.check_df(X_test, y_test)
        self.iterate_over_threshold()
        return self
#plots
###################### volume x cutting threshold and precision vs cutting threshold
    def get_arrays(self, metric = 'recall', label = 'weighted avg'):
        
        metrics = np.array([self.thresh_dict[i][label][metric] for i in self.thresh_dict])
        support = np.array([self.thresh_dict[i][label]['support'] for i in self.thresh_dict])
        thresh_arr = np.array([i for i in self.thresh_dict])
        
        return {'thresh_arr':thresh_arr, 'metrics':metrics, 'support':support }
        
    def plt_cutting_threshold(self, label = 'weighted avg', metric = 'recall', scaled = True):
        metrics_dict = self.get_arrays(label = label,metric = metric)
        threshold_arr = metrics_dict['thresh_arr']
        metrics = metrics_dict['metrics']
        volume = metrics_dict['support']
        if scaled:
            volume = volume/volume.max()
        
        fig = plt.figure()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        ax1.plot(threshold_arr, metrics, color = 'g', label = metric)
        ax2.plot(threshold_arr, volume, label = 'Automation ratio')
        ax1.set_xlabel('cutting_threshold')
        
        plt.ylim(0,1)
        ax1.set_ylabel(metric)
        ax2.set_ylabel('automation_ratio')
        
        fig.legend()
        fig.show()
        return
############################### voume x precision
# bining + calibration
    def plt_calibration(self, n_digits = 2, label = 'all'):
        
        check_prob_df = self.check_prob_df
        
        check_prob_df['bin'] = check_prob_df['max_prob'].apply(lambda x: round(x,n_digits))
        if label == 'all':
            mean_df = check_prob_df.groupby(['bin'])[['got_right']].apply(lambda x: x.astype(int).mean()).reset_index()
            count_df = check_prob_df.groupby(['bin'])[['got_right']].apply(lambda x: x.astype(int).count()).reset_index()
            plt.clf()
            plt.plot(
                mean_df['bin'],
                mean_df['got_right']
                )
            plt.plot(
                count_df['bin'],
                count_df['got_right']/count_df['got_right'].sum()
                )
        else:
            check_prob_df = check_prob_df[check_prob_df['preds'] == label]
            mean_df = check_prob_df.groupby(['bin'])[['got_right']].apply(lambda x: x.astype(int).mean()).reset_index()
            count_df = check_prob_df.groupby(['bin'])[['got_right']].apply(lambda x: x.astype(int).count()).reset_index()
            plt.clf()
            plt.plot(
                mean_df['bin'],
                mean_df['got_right']
                )
            plt.plot(
                count_df['bin'],
                count_df['got_right']/count_df['got_right'].sum()
                )
            
            
        plt.plot([0,1],[0,1])
        plt.ylim([0,1])
        plt.xlim([0,1])
        plt.title('Calibration curve')
        return
        ##### precision vs automation ratio
    def plt_precision_automation(self, label = 'weighted avg', metric = 'recall', scaled = True):
        metrics_dict = self.get_arrays(label = label,metric = metric)
        threshold_arr, metrics, volume = metrics_dict['thresh_arr'],metrics_dict['metrics'],metrics_dict['support']
        if scaled:
            volume = volume/volume.max()    
        
        plt.clf()
        fig2,ax1_2 = plt.subplots()
        ax1_2.scatter(volume, metrics)
        ax1_2.set_xlabel('Automation ratio')
        ax1_2.set_ylabel(metric)
        fig2.suptitle('IVA Classifier performance')        
        fig2.show()
        return
###############################
#alterados = pd.DataFrame(np.array(max_probs)[np.array(total_data.loc[dummy_df.index][total_data['Alterado'] == 1]['new_index'])-1])
#inalterados =pd.DataFrame(np.array(max_probs)[list(total_data[total_data['Alterado'] == 0]['new_index'])])
#alterados.hist(cumulative=True, density=1, bins=100)
#inalterados.hist(cumulative=True, density=1, bins=100)

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.hist(alterados.values, density = 1,cumulative=False, bins=100, label = 'Alterados', alpha = 0.3)
#ax.hist(inalterados.values,density = 1,cumulative=False, bins=100, color='red', alpha=.3,label = 'Inalterados')
#plt.xlabel('model prediction')
#plt.ylabel('cumulative sum')
#plt.legend()

#ax1 = ax.twinx()
#alterados.plot(kind='kde', ax=ax1)
#inalterados.plot(kind='kde', ax=ax1, color='red')

#plt.show()
#################################

#class_metrics = {}
#class_metrics_df_dict = {}
#classes = classifier.classes_
#
#
#for class_index,class_name in tqdm.tqdm(enumerate(classes)):
#    class_true_mask = (y_test == class_name).flatten()
#    class_pred_mask = preds == class_name
#    
#   class_prob_preds = prob_preds[class_pred_mask]
#    class_preds = preds[class_pred_mask]
#    class_prob_preds_max = class_prob_preds.max(axis = 1)
#    class_true_positive_mask = y_test[class_pred_mask].flatten() == preds[class_pred_mask]
#    n_samples = y_test.shape[0]
#    n_class_pred = class_preds.shape[0]
#    class_metrics[class_name] = {}
#    for threshold in threshold_arr:        
#        threshold_mask = class_prob_preds_max >= threshold
#        preds_threshold = class_preds[threshold_mask]
#        tp = class_prob_preds[threshold_mask & class_true_positive_mask]
#        n_tp = tp.shape[0]
#        n_preds_threshold = preds_threshold.shape[0]
#        
#        try:
#            class_weight = n_class_pred/n_samples
#        except:
#            pass
#        
#        try:
#            volume = n_preds_threshold/n_class_pred
#        except:
#            pass
#        try:
#            precision = n_tp/n_preds_threshold
#        except:
#            pass
#        class_metrics[class_name][threshold] = {}
#        class_metrics[class_name][threshold]['class_weight'] = class_weight
#        class_metrics[class_name][threshold]['precision'] = precision
#        class_metrics[class_name][threshold]['volume'] = volume      
#    class_metrics_df_dict[class_name] = pd.DataFrame(class_metrics[class_name]).T
#
#index = 1
#class_name = classes[index]
#
#plt.clf()
#fig2,ax1_2 = plt.subplots()
#ax1_2.scatter(
#        class_metrics_df_dict[class_name]['volume'],
#        class_metrics_df_dict[class_name]['precision']
#        )
#ax1_2.set_xlabel('Automation ratio')
#ax1_2.set_ylabel('Precision')
#ax1_2.set_label('IVA {}'.format(class_name))
#fig2.suptitle('IVA Classifier performance')
#
#fig2.show()
###########by ranking
    def plt_ranking(self, label = 'all'):
        
        if label == 'all':    
            classes = self.model.classes_
            y_test = self.y_test
            prob_preds = self.prob_preds
            
        else:
            label_msk = y_test.isin(label)
            classes = self.model.classes_[label_msk]
            y_test = self.y_test[label_msk]
            prob_preds = self.prob_preds[label_msk]
        
        i = 0
        for class_index,class_name in tqdm.tqdm(enumerate(classes)):
            class_true_mask = (y_test == class_name).flatten()
            class_prob = prob_preds[:,class_index]
            if i == 0:
                ranking_tp = np.array([(prob_preds[i] >= class_prob[i]).astype(int).sum() for i in range(len(prob_preds))])[class_true_mask]
                
            else:
                ranking_tp = np.concatenate(
                        [ranking_tp,
                         np.array([(prob_preds[i] >= class_prob[i]).astype(int).sum() for i in range(len(prob_preds))])[class_true_mask]
                         ],
                        axis = 0
                        )
            i+=1
        
        cdf_model = pd.Series(ranking_tp).value_counts().sort_index().cumsum()/pd.Series(ranking_tp).value_counts().sum()
        pdf_model = pd.Series(ranking_tp).value_counts().sort_index()/pd.Series(ranking_tp).value_counts().sum()
        
        plt.clf()
        fig, ax = plt.subplots()
        ax = fig.gca()
        ax.bar(pdf_model.index,pdf_model)
        ax.set_xlabel('model prediction ranking')
        ax.set_ylabel('recall')
        ax.set_xticks(cdf_model.index)
        ax1 = ax.twinx()
        ax.set_ylim([0,1])
        ax1.set_ylim([0,1])
        ax1.plot(pdf_model.index,cdf_model, color = 'orange')
        
        fig.show()
        return

    @classmethod
    def feature_importance_sparse_tree(
    	cls,
    	feature_importance,
    	random_vars,
    	categorical_nested_vars,
    	): 
    
	    if not len(random_vars) == feature_importance.shape[-1]:
	        print('len(random_vars) ({}) must match num of columns in feature_importances ({}).'.format(len(random_vars), feature_importance.shape[-1]))
	        raise AssertionError
	        
	    ft_importance = dict(zip(random_vars,feature_importance))  
	    
	    for cat in categorical_nested_vars.keys():
	        ft_importance[cat] = 0
	        for key in categorical_nested_vars[cat]:
	            ft_importance[cat] += ft_importance[key]
	            ft_importance.pop(key, None)
	    
	    imp_sum = sum(ft_importance.values())
	    print('sum of importanes: {}'.format(imp_sum))
	    ft_importance = {k: v for k, v in sorted(ft_importance.items(), key=lambda item: item[1], reverse = True)}
	    
	    plt.bar(range(len(ft_importance)), list(ft_importance.values()), align='center')
	    plt.xticks(range(len(ft_importance)), list(ft_importance.keys()))
	    plt.show()
	    
	    return ft_importance
