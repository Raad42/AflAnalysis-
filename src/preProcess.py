import pandas as pd 
import numpy as np
import os

file_path = os.path.join('data', 'raw', 'rawData12_24.csv')

mainDF = pd.read_csv(file_path)

mainDF['Attendance'] = mainDF['Attendance'].str.replace(',', '', regex=True)  # Remove commas
mainDF['Attendance'] = pd.to_numeric(mainDF['Attendance'], errors='coerce') 

#Removing Years affected by COVID-19
mainDF = mainDF[(mainDF.Year != 2020)& (mainDF.Year != 2021)]

#Rain?

output_file = os.path.join('data', 'interim', 'fullSesData.csv')

mainDF.to_csv(output_file, index=False)

