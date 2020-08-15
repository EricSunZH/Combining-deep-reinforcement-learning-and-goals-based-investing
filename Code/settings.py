import datetime
import numpy as np

START_DATE = '2000-01-03'
END_DATE = '2019-12-31'
DATE_FORMAT = '%Y-%m-%d'
EPS = 1e-8
RHO = 0.2
GAMMA_r = 1.01
STEP_LENGTH = 20   
ADD_INVEST = np.repeat(0.1, 120)  

DATA_DIR = 'marketdata.csv'
CSV_DIR = 'portfolio-management.csv'
START_DATETIME = datetime.datetime.strptime(START_DATE, DATE_FORMAT)  

# ------------------------------------------------------------------
# reward function settings
GOAL = 18
ALPHA = 0.95
BETA = 0.5
GAMMA = 0.9
DELTA = 1
THETA = 1  
MU = 0.1

