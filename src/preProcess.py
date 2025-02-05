import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mainDF = pd.read_csv(r'C:\Users\raadr\OneDrive\Desktop\AflAnalysis-\data\raw\rawData12_24Complete.csv')

# Removing stadiums which have low capacity or low usage
stadiums_to_remove = [
    'Bellerive Oval', 'Manuka Oval', 'Stadium Australia', 'Marrara Oval', 
    "Cazaly's Stadium", 'Eureka Stadium', 'Traeger Park', 'Wellington', 
    'Jiangwan Stadium', 'Norwood Oval', 'Blacktown', 'Riverway Stadium', 
    'Summit Sports Park'
]


mainDF = mainDF[~mainDF['Venue'].isin(stadiums_to_remove)]

#Removing Years affected by COVID-19
mainDF = mainDF[(mainDF.Year != 2020)& (mainDF.Year != 2021)]

#Rain?