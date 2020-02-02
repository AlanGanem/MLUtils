# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:30:32 2019

@author: User Ambev
"""
import numpy as np
import tqdm
import joblib

cat_model = joblib.load('catboost_model.sav')
rf_classifier = joblib.load('rf_model.sav')


ensemble = classifier_ensemble(
        **{'RF':rf_classifier,
         'CAT':cat_model
            }
        )


fitargs = {
        'RF':{'X':rf_X,
              'y':rf_y,
                },
                
        'CAT':{'X':train_dataset,
               'eval_set':eval_dataset
               }
        }

ensemble.fit(**fitargs)

predargs = {
        'RF':{'X':X_test},
        'CAT':{'data':eval_data}
        }

preds_ = ensemble.predict(method = 'max',**predargs)


proba_preds = ensemble.predict_proba(**predargs)


weights = {
        'RF':3,
        'CAT':3
        }

ensemble_proba_preds = ensemble.aggregate_preds(proba_preds,method = 'mean', weights = weights)
ensemble_preds = ensemble.predict_ensemble(proba_preds_ensemble= ensemble_proba_preds ,method = 'mean', weights = weights)


from sklearn.metrics import classification_report
print(classification_report(preds_['CAT'],eval_label))
print(classification_report(preds_['RF'],y_test))
print(classification_report(ensemble_preds,y_test))



class ClassifierEnsemble():
    
    @classmethod
    def load(cls, loading_path, **joblibargs):        
        return joblib.load(loading_path, **joblibargs)
    
    def save(self, saving_path, **joblibargs):        
        joblib.dump(self, saving_path, **joblibargs)
    
    def __init__(self, **models):
        self.models = models
        
        try:
            for method in ['fit','predict']:
                for model in self.models:
                    dir(self.models[model]).index(method)
        except:
            raise AttributeError(
                    'all models should contain "fit" and "predict" methods and "predict_proba" optionaly',
                    '"{}" does not contain the "{}" method'.format(model,method)
                    )
        
        return
    
    def fit(
            self,
            **fitargs
            ):
        
        
        assert all([i in fitargs.keys() for i in self.models.keys()])
        
        for model in tqdm.tqdm(self.models.keys()):
            self.models[model].fit(**fitargs[model])
        
        return self.models
    
    def cat_mapper(self):
        self.cat_map = {name:index for index,name in enumerate(self.models[list(self.models)[0]].classes_)}
        self.inv_cat_map = {index:name for index,name in enumerate(self.models[list(self.models)[0]].classes_)}
        self.classes_ = list(cat_map)
        return
    
    def predict(
            self,
            **predargs
            ):
        
        self.predargs = predargs
        assert all([i in predargs.keys() for i in self.models.keys()])
        preds = {}
        for model in tqdm.tqdm(self.models.keys()):
            preds[model] = self.models[model].predict(**self.predargs[model]).flatten()
        
        
        return preds
    
    def predict_proba(
            self,
            **probapredargs
            ):
        
        self.probapredargs = probapredargs
        assert all([i in probapredargs.keys() for i in self.models.keys()])
        proba_preds = {}
        for model in tqdm.tqdm(self.models.keys()):
            proba_preds[model] = self.models[model].predict_proba(**self.probapredargs[model])
        
        return proba_preds
    
    def predict_proba_ensemble(
            self,
            method = 'max',
            weights = None,
            custom_method = None,
            **probapredargs
            ):
        
        '''
        predicts from probabilities arrays aggregation.
        method defines how aggregation is done (get the class of max proba or get the max of proba arrays mean)
        
        '''
        
        assert method in ['max','mean','custom']
        
        proba_preds = self.predict_proba(**probapredargs)    
        
        proba_preds_ensemble = self.aggregate_preds(
                proba_preds,
                method = method,
                weights=weights,
                custom_method = custom_method
                )
        
        return proba_preds_ensemble
    
    def aggregate_preds(
            self,
            proba_preds,
            method = 'max',
            weights = None,
            custom_method = None):
        
        if method == 'custom':
            proba_preds_ensemble = custom_method(proba_preds)
        if method == 'mean':
            proba_preds_ensemble = sum(weights[model]*proba_preds[model] for model in self.models.keys())/sum(weights[model] for model in self.models.keys())
        if method == 'max':
            model_preds_max = np.concatenate(
                    [proba_preds[model].max(axis = 1).reshape(-1,1) for model in self.models.keys()],
                    axis = 1)
            
            proba_preds_ensemble_concat = np.concatenate(
                    [
                        proba_preds[model].reshape(
                            proba_preds[model].shape[0],
                            proba_preds[model].shape[1],
                            1
                        )
                        for model in self.models.keys()
                    ],
                    
                    axis = 2
                    )
            enum = dict(enumerate(np.argmax(model_preds_max,axis = 1)))
            proba_preds_ensemble = np.array([proba_preds_ensemble_concat[i,:,j] for i,j in enum.items()])
    
        return proba_preds_ensemble
    
    
    def predict_ensemble(
            self,
            method = 'max',
            weights = {},
            custom_method = None,
            proba_preds_ensemble = None,
            **probapredargs
            ):
        
        self.cat_mapper()
        
        if not (type(proba_preds_ensemble) == type(None)):
            
            proba_preds_ensemble = proba_preds_ensemble
            
        else:
            
            proba_preds_ensemble = self.predict_proba_ensemble(
                    method = method,
                    weights = weights,
                    custom_method = custom_method,
                    **probapredargs
                    )
        
        preds_ensemble = np.argmax(proba_preds_ensemble, axis = 1).flatten()
        preds_ensemble = np.array([self.inv_cat_map[pred] for pred in preds_ensemble])
        
        return preds_ensemble
        