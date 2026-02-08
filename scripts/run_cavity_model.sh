#!/bin/bash

for dustmodel in \
	''
do
    echo "Running cavity_model.py with $dustmodel"
    python3 cavity_model.py "$dustmodel"
done


#DIANA.dat
	#astrosil030_ol070_small.dat \
	#astrosil070_ol030_small.dat
	#astrosil050_ol050_small.dat 
	#astrosil080_cz020_small.dat \
	#astrosil_small.dat \
	#astrosil090_cz010_small.dat \
	#astrosil070_cz020_pyrcmg96010_small.dat \
	#astrosil070_cz020_h2ow010_small.dat \
	#'' 
