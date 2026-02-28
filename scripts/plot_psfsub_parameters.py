import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from glob import glob

foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_paper_draft/PSFSub_Docs/L1448MM1/'
filenames = glob(foldername + '*_options.csv')