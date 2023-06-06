import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            print("here")
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            print("After Loading")
            print(features)
            data_scaled=preprocessor.transform(features)
            ## error here
            preds=model.predict(data_scaled)
            print(preds)
            
            # its seems like predicted outcome is reversed, so for now just reverting the outputs while 
            #giving the answer until the problem is identified
            print(np.hsplit(preds, 2))
    
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    

class CustomData:
    def __init__(self,followers:int):
        self.followers = followers


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Followers": [self.followers],

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

