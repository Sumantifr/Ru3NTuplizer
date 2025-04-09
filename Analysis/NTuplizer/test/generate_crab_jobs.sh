#!/bin/bash

# Input file with sample names
#SAMPLES_FILE=${1:-"MC_Hadronic_Run3_2022.txt"}
SAMPLES_FILE=${1:="xx.txt"}

#input parameters
config_file=Run_MINIAOD_Run3_cfg.py
publication=True
site=T2_DE_DESY
DBS=global

#other constatnt inputs
YEAR=${2:-"2022"}
#"2022"
IsMC=${3:-"true"}
ERA=${4:-"C"}
Signal=${5:-"true"}

IsDATA=0
if [[ "$IsMC" == "false" ]]; then
	IsDATA=1
fi

echo "IsMC "${IsMC}

#this results in
identifier=${YEAR}
if [[ "$IsMC" == "true" ]]
then
	identifier+="_MC"
	if ${Signal}
	then
		identifier+="_SIGNAL"
	else
		identifier+="_Bkg"
	fi
else
	identifier+="_DATA_"${ERA}
fi

fil_list=crab_submit_${identifier}
mon_list=crab_monitor_${identifier}

truncate -s 0 ${fil_list}.sh
echo "#!/bin/bash" | cat >>${fil_list}.sh
truncate -s 0 ${mon_list}.sh

if [[ ! -f "$SAMPLES_FILE" ]]; then
  echo "Error: Samples file '$SAMPLES_FILE' not found."
  exit 1
fi

mapfile -t sample_data < "$SAMPLES_FILE"

# Read sample names from the input file

for sample in "${sample_data[@]}"; do

	sample=$(echo "$sample" | tr -d '\r')  # Remove carriage return if present
	label=$(echo "$sample" | cut -d'/' -f2)

	if [[ "$IsMC" == "false" ]]; then
		era_label=$(echo "$sample" | cut -d '/' -f 3 | cut -d '-' -f 1)
		era_label2=$(echo "$sample" | cut -d '/' -f 3 | cut -d '-' -f 2)
		label="${label}_${era_label}"
		#special condition for 2023 due to odd dataset naming
		if [[ "$YEAR" == "2023" ]] || [[ "$YEAR" == "2023BPiX"  ]]; then
			label="${label}_${era_label}_${era_label2}"
		fi

	fi
  	#dataset="${sample_data[$i]}"            # Use the sample as the dataset (can be modified)
  	dataset=$(echo "$sample" | tr -d '\r')  # Remove carriage return if present

  	echo ${sample}  

	./write_crab_config.sh $label $identifier $config_file $dataset $publication $site $DBS $YEAR $IsDATA $ERA
	echo "Python file ${label}.py created."
	
	#submit & monitor
	echo "crab submit -c crabfile_${YEAR}_${label}.py" | cat >>${fil_list}.sh
	echo "crab status -d crab_${identifier}/crab_crab_${label}/ #--maxmemory=4700 --maxjobruntime=2800" | cat >>${mon_list}.sh

done

echo "All Python files created successfully."

chmod 744 ${fil_list}.sh
chmod 744 ${mon_list}.sh
