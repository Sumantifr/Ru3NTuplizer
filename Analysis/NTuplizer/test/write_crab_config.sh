#!/bin/sh
production_tag=$1
workarea=$2
config=$3
Dataset=$4
publication=$5
site=$6
DBS=$7
YEAR=$8
IsDATA=$9
ERA=${10}

temp=crabfile_${YEAR}_${1}.py

echo "IsDATA" ${IsDATA}

golden_json=""
if [ $YEAR == "2022" ] || [ $YEAR == "2022EE" ]; then
	golden_json='/eos/user/c/cmsdqm/www/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json'
elif [ $YEAR == "2023" ] || [ $YEAR == "2023BPiX" ]; then
	golden_json='/eos/user/c/cmsdqm/www/CAF/certification/Collisions23/Cert_Collisions2023_366442_370790_Golden.json'
else
	golden_json='/eos/user/c/cmsdqm/www/CAF/certification/Collisions24/Cert_Collisions2024_378981_386951_Golden.json'
fi

truncate -s 0 $temp

echo "from CRABClient.UserUtilities import config
config = config()

config.General.requestName = 'crab_${production_tag}'
config.General.workArea = 'crab_${workarea}'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '${config}'
config.JobType.inputFiles = ['JECfiles','JERfiles', 'JetVetoMaps','roccor.Run2.v5']" | cat >>$temp
if [[ "$IsDATA" ==  "1" ]]; then
	echo "config.JobType.pyCfgParams = ['--YEAR','${YEAR}','--ERA','${ERA}','--IsDATA','--IsRun3','1','--ReadJEC','1']" | cat >>$temp
else
	echo "config.JobType.pyCfgParams = ['--YEAR','${YEAR}','--ERA','${ERA}','--IsRun3','1','--ReadJEC','1']" | cat >>$temp
fi
#echo "config.JobType.pyCfgParams = ['--YEAR','${YEAR}','--ERA','${ERA}','--IsDATA','$IsDATA','--IsRun3','1','--ReadJEC','1']
echo "config.JobType.disableAutomaticOutputCollection = True
config.JobType.outputFiles = ['hist.root','rootuple.root']
config.JobType.maxJobRuntimeMin = 2700
config.JobType.maxMemoryMB = 4000
config.JobType.allowUndistributedCMSSW = True

config.Data.inputDataset = '$Dataset'
config.Data.inputDBS = '$DBS'" | cat >>$temp
if [[ "$IsDATA" ==  "1" ]]; then
	echo "config.Data.splitting = 'LumiBased'
config.Data.lumiMask = '${golden_json}'" | cat >>$temp
else
	echo "config.Data.splitting = 'FileBased'" | cat >>$temp
fi
echo "config.Data.unitsPerJob = 1
#config.Data.totalUnits = 1 #for test job
config.Data.outLFNDirBase = '/store/user/chatterj/XtoYHto4b/'
config.Data.publication = $publication
config.Data.outputDatasetTag = '${production_tag}'
config.Data.publishDBS = 'phys03'

config.Site.storageSite = '$site'" | cat >>$temp
