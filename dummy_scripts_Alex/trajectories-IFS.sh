#!/bin/bash
# these are 48 hours from mature stage
t=-48
odir=$(pwd)
cdir=$(pwd | cut -b 37-39)
for k in $(ls -rt trastart-mature-*.txt | tail -378)
do
	start=$(ls $k | cut -b 17-27)
	name=$(ls $k | cut -b 17-37)
	mon=$(ls traend-$name*.txt | cut -b 30-34)
        nd="/home/ascherrmann/TT/trajectories-mature-$name.txt"
	M=$mon

	cp $k /home/ascherrmann/010-IFS/data/$M/
	cd /home/ascherrmann/010-IFS/data/$M/
	
	#	caltra trajectory-file timerange (-X for backward, X for forward) outfile -i 60 -o 60 -ts 5 -ref startdate (as a reference for time)
        /home/ascherrmann/scripts/caltra-IFSORO/prog/caltra $k $t $nd -i 60 -o 60 -ts 5 -ref $start
	sleep 600s
	rm $k
        cd $odir
done
