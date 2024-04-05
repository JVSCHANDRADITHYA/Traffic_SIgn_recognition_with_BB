import numpy as np 
import pandas as pd 


labels_path = 'F:\\road_sign_detect\\dataset\\labels.csv'
classes = pd.read_csv(labels_path)
class_names = list(classes['Name'])
print(class_names)