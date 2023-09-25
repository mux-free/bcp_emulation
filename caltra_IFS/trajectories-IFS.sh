#!/bin/bash

module load dyn_tools

# these are 48 hours from mature stage
t=-48
odir=$(pwd)
#cdir=$(pwd | cut -b 37-39)



#trastart_dir="/net/thermo/atmosdyn2/ascherrmann/010-IFS/ctraj/ETA"
trastart_dir="/net/helium/atmosdyn/freimax/data_msc/IFS-18/IFS-traj/CYC_validation/DEC17/ctraj"


for k in $(ls -rt ${trastart_dir}/trastart-mature-*.txt)
do
	start=$(ls $k | cut -b 98-108)
	
	
	name=$(ls $k | cut -b 98-118)
	

	#mon=$(ls ${trastart_dir}/traend-$name*.txt | cut -b 111-115)
        mon=$(ls ${trastart_dir}/trastart-mature-$name*.txt | cut -b 70-74)

	#nd="/net/helium/atmosdyn/freimax/data_msc/IFS-18/IFS-traj/traj-CYCcentre_randforest/trajectories-mature-$name.txt"
	M=$mon

	echo Month: $M and this is the start date: $start, the name is this: $name


	#cp $k /atmosdyn2/ascherrmann/010-IFS/data/$M/
	cp $k /net/helium/atmosdyn/freimax/msc_thesis/scripts/caltra_IFS/prog/IFS-1yr/$M/cdf/
	
        #cd /atmosdyn2/ascherrmann/010-IFS/data/$M/
	cd /net/helium/atmosdyn/freimax/msc_thesis/scripts/caltra_IFS/prog/IFS-1yr/$M/cdf/


        #/atmosdyn2/ascherrmann/scripts/caltra-IFSORO/prog/caltra $k $t $nd -i 60 -o 60 -ts 5 -ref $start 
	#rm $k
        #cd $odir
done
