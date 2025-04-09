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

temp=crabfile_${1}.py

truncate -s 0 $temp

echo "from CRABClient.UserUtilities import config
config = config()

config.General.requestName = 'crab_${production_tag}'
config.General.workArea = 'crab_${workarea}'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '${config}'
config.JobType.inputFiles = ['JECfiles','JERfiles', 'JetVetoMaps','roccor.Run2.v5']
config.JobType.pyCfgParams = ['--YEAR','${YEAR}','--ERA','${ERA}','--IsDATA','$IsDATA','--IsRun3','1','--ReadJEC','1']
config.JobType.disableAutomaticOutputCollection = True
config.JobType.outputFiles = ['hist.root','rootuple.root']
config.JobType.maxJobRuntimeMin = 2700
config.JobType.maxMemoryMB = 4000
config.JobType.allowUndistributedCMSSW = True

config.Data.inputDataset = '$Dataset'
config.Data.inputDBS = '$DBS'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
#config.Data.totalUnits = 1 #for test job
config.Data.outLFNDirBase = '/store/user/chatterj/XtoYHto4b/'
config.Data.publication = $publication
config.Data.outputDatasetTag = '${production_tag}'
config.Data.publishDBS = 'phys03'

config.Site.storageSite = '$site'" | cat >>$temp
