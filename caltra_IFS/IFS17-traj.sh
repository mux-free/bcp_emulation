#!/bin/bash

module load dyn_tools

# these are 48 hours from mature stage
t=-48
odir=$(pwd)


trastart_dir="/net/helium/atmosdyn/freimax/data_msc/IFS-17/ctraj"

#for k in $(ls -rt ${trastart_dir}/trastart-mature-*.txt)
k=${trastart_dir}/trastart-mature-*.txt

start=$(ls $k | cut -b 98-108)


name=$(ls $k | cut -b 98-118)


mon=$(ls ${trastart_dir}/trastart-mature-$name*.txt | cut -b 70-74)

        #nd="/net/helium/atmosdyn/freimax/data_msc/IFS-18/IFS-traj/traj-CYCcentre_randforest/trajectories-mature-$name.txt"
M=$mon

echo Month: $M and this is the start date: $start, the name is this: $name


#cp $k /atmosdyn2/ascherrmann/010-IFS/data/$M/
cp $k /net/helium/atmosdyn/freimax/msc_thesis/scripts/caltra_IFS/prog/IFS-1yr/$M/cdf/

#cd /atmosdyn2/ascherrmann/010-IFS/data/$M/
cd /net/helium/atmosdyn/freimax/msc_thesis/scripts/caltra_IFS/prog/IFS-1yr/$M/cdf/


