import numpy as np
from ifu_analysis.jdfitting import read_optool
import matplotlib.pyplot as plt 

input_foldername = '/home/vorsteja/Documents/JOYS/JDust/optool_opacities/'

op_fn = input_foldername + 'pyrmg80.dat'

header,lam,kabs,ksca,g = read_optool(op_fn)

plt.plot(lam,kabs,label='abs')
plt.plot(lam,ksca,label='scat')
plt.plot(lam,kabs + ksca,label='ext')
plt.legend()
plt.yscale('log')
plt.show()
