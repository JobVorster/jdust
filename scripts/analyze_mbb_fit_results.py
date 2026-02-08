import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os
from ifu_analysis.jdspextract import read_aperture_ini
import astropy.units as u
from astropy.coordinates import SkyCoord

pd.set_option('display.max_columns', None)

source_name = 'L1448MM1'
d = 293 #pc
extinction_correction = True


#Define aperture filename (for aperture names and radii).
aperture_filename = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/input-files/aperture-ini-%s.txt'%(source_name)

if os.path.isfile(aperture_filename):
	aper_names,aper_sizes,coord_list = read_aperture_ini(aperture_filename)
	skycoord_arr = [SkyCoord(ra,dec,frame='icrs') for (ra,dec) in coord_list]
	skycoord_sep_arr = [skycoord_arr[0].separation(skycoord_arr[i]).to(u.arcsec) for i in range(len(coord_list))]

if extinction_correction:
	input_foldername = '/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/spectral_extraction/spectra_bb/Extinction_Corrected/'
else:
	input_foldername ='/home/vorsteja/Documents/JOYS/JDust/ifu_analysis/output-files/L1448MM1_post_processing/spectral_extraction/spectra_bb/'

results_filename = input_foldername + 'fit_results.csv'

chi2_red_ceiling = 500

df_mbbresults = pd.read_csv(results_filename)

do_boxplot = False
if do_boxplot:
	model_arr = ['2BB','2BB-SP','2BB-SO','2BB-SPO','2BB-SPO-WW','2BB-SPO-WH','2BB-SPO-WWH']
	for psfsub in [True, False]:
		#Plots of goodness of fit statistics per model.
		chi2_2D_arr = []
		for model in model_arr:
			df_mbbresults = pd.read_csv(results_filename)

			#PSFSub filtering
			inds_psfsub = np.where(df_mbbresults['PSF Subtracted (T/F)'].values==psfsub)[0]
			df_mbbresults = df_mbbresults.iloc[inds_psfsub]


			#Remove values where the fit did not work.
			inds_badfits = np.where(df_mbbresults['Fit Successful (T/F)'].values!=False)[0]
			df_mbbresults = df_mbbresults.iloc[inds_badfits]

			#Model filtering
			inds_model = np.where(df_mbbresults['Model'].values==model)[0]
			df_mbbresults = df_mbbresults.iloc[inds_model]

			chi2_2D_arr.append(df_mbbresults['chi2_reduced'].values)

		plt.figure(figsize = (10,4))
		plt.boxplot(chi2_2D_arr,tick_labels=model_arr)
		plt.gca().tick_params(axis='x', rotation=45)

		plt.ylabel('chi2_reduced')
		plt.suptitle('Source: %s, PSFSUB: %s, Av Corrected: %s'%(source_name,psfsub, extinction_correction)+'\n Codes Legend- SO: Olivine, SP: Pyroxene, WW: 15 K H2O, WH: 150 K H2O')
		plt.show()

psfsub = True 
GREEN = '\033[92m'
YELLOW = '\033[93m'
ORANGE = '\033[38;5;208m'
RESET = '\033[0m'
BOLD = '\033[1m'

COL_WIDTH = 12
SIG_FIGS = 3

model_params = {
    'BB': 2,              # temp, scaling
    'BB-SP': 3,           # temp, scaling, pyroxene_scaling
    'BB-SO': 3,           # temp, scaling, olivine_scaling
    'BB-SPO': 4,          # temp, scaling, pyroxene_scaling, olivine_scaling
    'BB-SPO-WW': 5,       # BB-SPO + h2o15K_scaling (note: "WW" seems to be just one water component)
    'BB-SPO-WH': 5,       # BB-SPO + h2o150K_scaling
    'BB-SPO-WWH': 6,      # BB-SPO + h2o15K_scaling + h2o150K_scaling
    '2BB': 4,             # temp1, scaling1, temp2, scaling2
    '2BB-SP': 5,          # 2BB + pyroxene_scaling
    '2BB-SO': 5,          # 2BB + olivine_scaling
    '2BB-SPO': 6,         # 2BB + pyroxene_scaling + olivine_scaling
    '2BB-SPO-WW': 7,      # 2BB-SPO + h2o15K_scaling
    '2BB-SPO-WH': 7,      # 2BB-SPO + h2o150K_scaling
    '2BB-SPO-WWH': 8      # 2BB-SPO + h2o15K_scaling + h2o150K_scaling
}


for aperture in aper_names:
	df_mbbresults = pd.read_csv(results_filename)
	#Remove values where the fit did not work.
	inds_badfits = np.where(df_mbbresults['Fit Successful (T/F)'].values!=False)[0]
	df_mbbresults = df_mbbresults.iloc[inds_badfits]

	#Values without psf subtraction.
	inds_psfsub = np.where(df_mbbresults['PSF Subtracted (T/F)'].values==psfsub)[0]
	df_mbbresults = df_mbbresults.iloc[inds_psfsub]

	inds_aperture = np.where(df_mbbresults['Aperture'].values==aperture)[0]
	df_mbbresults = df_mbbresults.iloc[inds_aperture]
	print('#'*30)
	print('#'*9 + ' '*5 + aperture + ' '*5 + '#'*9 )
	print('#'*30)

	sep_str = ' '*5

	cols = ['Model', 'BB1T','BB1S','BB2T','BB2S','Sil Oliv','Sil Pyro','H2O 15K','H2O 150K','BIC','chi2_reduced']
	print_str = ''
	for col in cols[:-1]:
		print_str += f'{col:<{COL_WIDTH}}'
	print_str += f'{cols[-1]:<{COL_WIDTH}}'
	print(print_str)

	sort_by = 'chi2_reduced'

	chi2_vals = df_mbbresults[sort_by].values
	if len(chi2_vals) > 0:
		sorted_indices = sorted(range(len(chi2_vals)), key=lambda i: chi2_vals[i])
		lowest_3 = set(sorted_indices[:3])
		colors = {sorted_indices[0]: GREEN, sorted_indices[1]: YELLOW, sorted_indices[2]: ORANGE}


	for i in range(len(df_mbbresults['Aperture'].values)):
		print_vals = [df_mbbresults[x].values[i] for x in cols]
		print_str = ''

		for val,col in zip(print_vals[:-1],cols[:-1]):
			if col in ['BB1T','BB2T', 'chi2_reduced','BIC']:
				FORMATTER = 'f'
				SIG_FIGS = 0
			else:
				FORMATTER = 'E'
				SIG_FIGS = 2

			if type(val) == str:
				print_str += f'{val:<{COL_WIDTH}}'
			else:
				print_str += f'{val:<{COL_WIDTH}.{SIG_FIGS}{FORMATTER}}'
		val = print_vals[-1]

		#Want to see the Chi2 as a integer.
		FORMATTER = 'f'
		SIG_FIGS = 2
		formatted = f'{val:<{COL_WIDTH}}' if isinstance(val, str) else f'{val:<{COL_WIDTH}.{SIG_FIGS}{FORMATTER}}'
		print_str += formatted
		

		# Add color to entire row if this row is in top 3 lowest chi2
		if i in lowest_3:
			print_str =  BOLD + colors[i] + print_str + RESET
		print(print_str)
			
	#print(df_mbbresults[['Model','BB1T','BB1S','BB2T','BB2S','chi2_reduced']])

#Plot best fit T1 and S1

aperture_plots = {}

for aperture in aper_names:
	df_mbbresults = pd.read_csv(results_filename)
	#Remove values where the fit did not work.
	inds_badfits = np.where(df_mbbresults['Fit Successful (T/F)'].values!=False)[0]
	df_mbbresults = df_mbbresults.iloc[inds_badfits]

	#Values without psf subtraction.
	inds_psfsub = np.where(df_mbbresults['PSF Subtracted (T/F)'].values==psfsub)[0]
	df_mbbresults = df_mbbresults.iloc[inds_psfsub]

	inds_aperture = np.where(df_mbbresults['Aperture'].values==aperture)[0]
	df_mbbresults = df_mbbresults.iloc[inds_aperture]
	if len(df_mbbresults['Aperture']) > 0:
		sort_by = 'chi2_reduced'
		chi2_vals = df_mbbresults[sort_by].values
		if len(chi2_vals) > 0:
			sorted_indices = sorted(range(len(chi2_vals)), key=lambda i: chi2_vals[i])
			lowest_3 = set(sorted_indices[:3])
		ind_select = sorted_indices[0]

		aper_ind = np.where(aper_names==aperture)[0][0]

		angular_sep = skycoord_sep_arr[aper_ind].value #arcsec
		aperture_plots['%s:Angular_Sep'%(aperture)] = angular_sep
		aperture_plots['%s:Temp1'%(aperture)] = df_mbbresults['BB1T'].values[ind_select]
		aperture_plots['%s:Scale1'%(aperture)] = df_mbbresults['BB1S'].values[ind_select]
		aperture_plots['%s:SO'%(aperture)] = df_mbbresults['Sil Oliv'].values[ind_select]
		aperture_plots['%s:SP'%(aperture)] = df_mbbresults['Sil Pyro'].values[ind_select]


plt.figure(figsize = (10,10))
for ap_series in ['A','B','C']:
	
	#Grab dictionary keys.
	sep_cols = []
	for col in aperture_plots.keys():
		if ('Angular_Sep' in col) and (col[0] == ap_series):
			sep_cols.append(col)

	t_cols = []
	for col in aperture_plots.keys():
		if ('Temp1' in col) and (col[0] == ap_series):
			t_cols.append(col)

	scale_cols = []
	for col in aperture_plots.keys():
		if ('Scale1' in col) and (col[0] == ap_series):
			scale_cols.append(col)

	SO_cols = []
	for col in aperture_plots.keys():
		if ('SO' in col) and (col[0] == ap_series):
			SO_cols.append(col)

	SP_cols = []
	for col in aperture_plots.keys():
		if ('SP' in col) and (col[0] == ap_series):
			SP_cols.append(col)

	#Create the plot arrays.
	sep_arr = []
	for col in sep_cols:
		sep_arr.append(aperture_plots[col])

	t_arr = []
	for col in t_cols:
		t_arr.append(aperture_plots[col])

	scale_arr = []
	for col in scale_cols:
		scale_arr.append(aperture_plots[col])

	SO_arr = []
	for col in SO_cols:
		SO_arr.append(aperture_plots[col])

	SP_arr = []
	for col in SP_cols:
		SP_arr.append(aperture_plots[col])

	sep_arr = np.array(sep_arr)
	sep_arr*= d
	do_mass = True
	if do_mass:
		SO_arr = np.array(SO_arr)
		SP_arr = np.array(SP_arr)
		aperture_radius = 0.7 #arcsec
		aperture_radius *= d #au
		aperture_area = np.pi*aperture_radius**2 #au^2
		aperture_area *= 1.496e+13**2 #cm^2
		SO_arr *= aperture_area #g
		SO_arr *= 5.267e-31 #Jupiter masses

		SP_arr *= aperture_area #g
		SP_arr *= 5.267e-31 #Jupter masses


	plt.subplot(221)
	plt.plot(sep_arr,t_arr,label = ap_series,marker='o',markeredgecolor='black')
	plt.xlabel('Distance from Star+Disk (au, d=293 pc)')
	plt.ylabel('Hot Blackbody Temperature (K)')
	plt.subplot(222)
	plt.plot(sep_arr,scale_arr,label = ap_series,marker='o',markeredgecolor='black')
	plt.xlabel('Distance from Star+Disk (au, d=293 pc)')
	plt.ylabel('Hot Blackbody Scale (aperture solid angle)')
	plt.subplot(223)
	plt.plot(sep_arr,SO_arr,label = ap_series,marker='o',markeredgecolor='black')
	plt.xlabel('Distance from Star+Disk (au, d=293 pc)')
	if not do_mass:
		plt.ylabel('Olivine surface density (g cm-2)')
	else:
		plt.ylabel(r'Olivine Mass (M$_{\rm Jup}$)')
	plt.subplot(224)
	plt.plot(sep_arr,SP_arr,label = ap_series,marker='o',markeredgecolor='black')
	plt.xlabel('Distance from Star+Disk (au, d=293 pc)')
	if not do_mass:
		plt.ylabel('Pyroxene surface density (g cm-2)')
	else:
		plt.ylabel(r'Pyroxene Mass (M$_{\rm Jup}$)')
plt.subplot(221)
plt.legend()
plt.subplot(222)
plt.legend()
plt.subplot(223)
plt.legend()
plt.subplot(224)
plt.legend()
if not do_mass:
	plt.savefig('Fit_overview.png',dpi=200,bbox_inches='tight')
else:
	plt.savefig('Fit_overview_mass.png',dpi=200,bbox_inches='tight')
for ap_series in ['A','B','C']:
	#Grab dictionary keys.
	sep_cols = []
	for col in aperture_plots.keys():
		if ('Angular_Sep' in col) and (col[0] == ap_series):
			sep_cols.append(col)

	SO_cols = []
	for col in aperture_plots.keys():
		if ('SO' in col) and (col[0] == ap_series):
			SO_cols.append(col)

	SP_cols = []
	for col in aperture_plots.keys():
		if ('SP' in col) and (col[0] == ap_series):
			SP_cols.append(col)

	#Create the plot arrays.
	sep_arr = []
	for col in sep_cols:
		sep_arr.append(aperture_plots[col])

	SO_arr = []
	for col in SO_cols:
		SO_arr.append(aperture_plots[col])

	SP_arr = []
	for col in scale_cols:
		SP_arr.append(aperture_plots[col])
	print(ap_series)


	sep_arr = np.array(sep_arr)
	SO_arr = np.array(SO_arr)
	SP_arr = np.array(SP_arr)

	print(SO_arr)
	print(SP_arr)

	SO_arr[np.isnan(SO_arr)] = 0
	SP_arr[np.isnan(SP_arr)] = 0
	SPO_arr = SO_arr + SP_arr #g cm-2

	sep_arr*= d
	aperture_radius = 0.7 #arcsec
	aperture_radius *= d #au
	aperture_area = np.pi*aperture_radius**2 #au^2
	aperture_area *= 1.496e+13**2 #cm^2
	mass_arr = SPO_arr*aperture_area #g
	mass_arr *= 1.67442e-28 #earth masses


	plt.figure(figsize = (7,7))
	plt.scatter(sep_arr,mass_arr,label = ap_series)
	plt.xlabel('Distance from Star+Disk (au, d=293 pc)')
	plt.ylabel(r'Dust mass (M$_\oplus$)')
plt.show()





model_arr = ['2BB-SPO','2BB-SPO-WWH']
#Plots of quantities' distance dependence.
for model in model_arr:
	df_mbbresults = pd.read_csv(results_filename)
	#Index(['Source Name', 'Aperture', 'PSF Subtracted (T/F)', 'Model',
	#       'Fit Successful (T/F)', 'BB1T', 'lsqunc BB1T', 'BB1S', 'lsqunc BB1S',
	#       'BB2T', 'lsqunc BB2T', 'BB2S', 'lsqunc BB2S', 'Sil Oliv',
	#       'lsqunc Sil Oilv', 'Sil Pyro', 'lsqunc Sil Pyro', 'H2O 15K',
	#       'lsqunc H2O 15K', 'H2O 150K', 'lsqunc H2O 150K', 'chi2', 'chi2_reduced',
	#       'AIC', 'BIC']


	#Remove values where the fit did not work.
	inds_badfits = np.where(df_mbbresults['Fit Successful (T/F)'].values!=False)[0]
	df_mbbresults = df_mbbresults.iloc[inds_badfits]

	#Remove fits with too high chi_red:
	inds_chi2red = np.where(df_mbbresults['chi2_reduced'].values< chi2_red_ceiling)[0]
	df_mbbresults = df_mbbresults.iloc[inds_chi2red]

	#Remove the non-psfsubbed indices.
	psfsub = True
	inds_psfsub = np.where(df_mbbresults['PSF Subtracted (T/F)'].values==psfsub)[0]
	df_mbbresults = df_mbbresults.iloc[inds_psfsub]

	inds_model = np.where(df_mbbresults['Model'].values==model)[0]
	df_mbbresults = df_mbbresults.iloc[inds_model]

	#Lets only look at aperture A2
	#aperture = 'A3'
	#inds_aperture = np.where(df_mbbresults['Aperture'].values==aperture)[0]
	#df_mbbresults = df_mbbresults.iloc[inds_aperture]

	col_arr =['Aperture','BB1T','Sil Oliv','BB2T','Sil Pyro']

	plot_results = {}
	for c in ['A','B','C']:
		plot_results['%s:Angular_Sep'%(c)] = []
		plot_results['%s:%s'%(c,col_arr[1])] = []
		plot_results['%s:%s'%(c,col_arr[2])] = []
		plot_results['%s:%s'%(c,col_arr[3])] = []
		plot_results['%s:%s'%(c,col_arr[4])] = []
		
	dfaperture, dfbb1t, dfbb1s, dfbb2t, dfbb2s=[df_mbbresults[col].values for col in col_arr]
	for aperture, bb1t, bb1s, bb2t, bb2s in zip(dfaperture, dfbb1t, dfbb1s, dfbb2t, dfbb2s) :
		if aperture =='PS':
			continue

		if (bb1t > 1e4) or (bb2t > 1e4):
			continue

		c = aperture[0]
		aper_ind = np.where(aper_names==aperture)[0][0]

		angular_sep = skycoord_sep_arr[aper_ind].value #arcsec
		
		plot_results['%s:Angular_Sep'%(c)].append(angular_sep)
		plot_results['%s:%s'%(c,col_arr[1])].append(bb1t)
		plot_results['%s:%s'%(c,col_arr[2])].append(bb1s)
		plot_results['%s:%s'%(c,col_arr[3])].append(bb2t)
		plot_results['%s:%s'%(c,col_arr[4])].append(bb2s)

	for c in ['A','B','C']:
		plotcol_arr = ['%s:Angular_Sep'%(c),'%s:%s'%(c,col_arr[1]),'%s:%s'%(c,col_arr[2]),'%s:%s'%(c,col_arr[3]),'%s:%s'%(c,col_arr[4])]
		angular_sep_arr,bb1t_arr,bb1s_arr,bb2t_arr,bb2s_arr = [plot_results[col] for col in plotcol_arr]

		plt.subplot(221)
		plt.scatter(angular_sep_arr,bb1t_arr,label='%s'%(c))
		#plt.ylim(500,1500)
		plt.legend()
		plt.xlabel('Angular Separation (arcsec)')
		plt.ylabel('%s'%(col_arr[1]))
		plt.subplot(222)
		plt.scatter(angular_sep_arr,bb1s_arr,label='%s'%(c))
		plt.legend()
		plt.xlabel('Angular Separation (arcsec)')
		plt.ylabel('%s'%(col_arr[2]))
		plt.subplot(223)
		plt.scatter(angular_sep_arr,bb2t_arr,label='%s'%(c))
		plt.legend()
		plt.xlabel('Angular Separation (arcsec)')
		#plt.ylim(200,500)
		plt.ylabel('%s'%(col_arr[3]))
		plt.subplot(224)
		plt.scatter(angular_sep_arr,bb2s_arr,label='%s'%(c))
		plt.legend()
		plt.xlabel('Angular Separation (arcsec)')
		plt.ylabel('%s'%(col_arr[4]))
	plt.suptitle('Source: %s, PSFSUB: %s, Model: %s'%(source_name,psfsub,model)+'\n SO: Olivine, SP: Pyroxene, WW: 15 K H2O, WH: 150 K H2O')
	plt.show()




	
print(df_mbbresults[['Aperture','PSF Subtracted (T/F)', 'Model','BB1T','BB1S','BB2T','BB2S','chi2_reduced']])