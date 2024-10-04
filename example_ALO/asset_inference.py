#asset_[step_name].py
 
# -*- coding: utf-8 -*-
import os
import sys
from alolib.asset import Asset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd 


import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
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

        """ data load """        
        ## ALO API 사용할 경우,
        df_path_list = glob('{}/*/*.csv'.format(self.asset.get_input_path()))
        data = pd.read_csv(df_path_list[-1])

        ## 직접 데이터를 업로드하고 로드하는 경우,
        # data = pd.read_csv(os.path.join(self.args['data_dir'], f"{self.args['file_name']}.csv"))
        
        
        """ model load """
        model = joblib.load(os.path.join(self.asset.get_model_path(), self.args['model_file']))
        
        
        """ model inference """
        prediction = model.predict(data)
        predict_probability = model.predict_proba(data)
        class_probability = np.max(predict_probability, axis=1)
        
        data['pred_class'] = prediction
        data['pred_proba'] = class_probability
        
        
        """ result save """
        output_path = self.asset.get_output_path() # needed: .csv only 1 / .jpg only 1 / .only, .jpg each 1
        data.to_csv(output_path + 'output.csv')
        
        summary = {}
        summary['result'] = 'OK'
        summary['score'] = 0.98
        summary['note'] = "The score represents the probability value of the model's prediction result."        
        self.asset.save_summary(result=summary['result'], score=summary['score'], note=summary['note'])
        
        
        """ requirement API """
        self.asset.save_data(self.args)
        self.asset.save_config(self.config)
        
 
 
#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    envs, argv, data, config = {}, {}, {}, {}
    ua = UserAsset(envs, argv, data, config)
    ua.run()
