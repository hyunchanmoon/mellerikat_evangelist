#asset_input.py
 
# -*- coding: utf-8 -*-
import os
import sys
from alolib.asset import Asset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  
from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


#--------------------------------------------------------------------------------------------------------------------------
#    CLASS
#--------------------------------------------------------------------------------------------------------------------------
class UserAsset(Asset):
    def __init__(self, asset_structure):
        super().__init__(asset_structure)        
        self.args       = self.asset.load_args()
        self.config     = self.asset.load_config()

    @Asset.decorator_run
    def run(self):        
                            
        """ Data load """        
        data = pd.read_csv(os.path.join(self.asset.get_input_path(), f"{self.args['file_name']}.csv")) # ALO API 사용
        # data = pd.read_csv(os.path.join(self.args['data_dir'], f"{self.args['file_name']}.csv")) # custom directory 사용
        
        
        """ train/validset define """
        # 독립변수/종속변수 정의
        X = data.drop([self.args['target']], axis=1)
        y = data[self.args['target']]

        # 학습(train)/검증(valid) 데이터셋 정의
        X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                              y,
                                                              test_size=(1-self.args['train_ratio']),
                                                              shuffle=False,
                                                              random_state=self.args['random_state'])
        
        
        """ model define & training """
        # 램덤포레스트 모델 정의
        model = RandomForestClassifier(n_estimators=100, max_depth=None, max_leaf_nodes=None)

        # 랜덤포레스트 모델 학습
        model.fit(X_train, y_train)        
        
        
        """ model evaulation """
        # prediction
        y_pred = model.predict(X_valid)

        # evaluation by metrics
        result = {'acc': accuracy_score(y_valid, y_pred),
                  'f1': f1_score(y_valid, y_pred, average='macro')}
        
        
        """ model save """
        joblib.dump(model, os.path.join(self.asset.get_model_path(), "best_model.joblib"))
        
        
        """ save configuration """
        self.config['data_dir'] = self.args['data_dir']
        self.config['model_file'] = 'best_model.joblib'
        
        
        """ requirement API """
        self.asset.save_data(result) # to next asset
        self.asset.save_config(self.config) # to next asset
                
        
#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ua = UserAsset(envs={}, argv={}, data={}, config={})
    ua.run()
