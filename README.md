# XtoYH

Framework for ntuple production in Run 3 (from MINIAOD samples)

- Log in to your lxplus account

- cd work/private

- mkdir XtoYH

- cd XtoYH

- cmsrel CMSSW_14_2_1 <br/>

- cd CMSSW_14_2_1/src

- git clone https://github.com/Sumantifr/XtoYH4b.git . <br/>
  *(Don't forget '.')*

- scram b -j10 

## For a test run: 

- cd CMSSSW_14_2_1/src

- cmsenv

- cd $CMSSW_BASE/src/Analysis/NTuplizer/test/

- voms-proxy-init -rfc -voms cms -valid 48:00

- cmsRun Run_MINIAOD_Run3_cfg.py

Enjoy!

## For submitting crab jobs (MC):

- cd CMSSSW_14_2_1/src

- cmsenv

- cd $CMSSW_BASE/src/Analysis/NTuplizer/test/

- voms-proxy-init -rfc -voms cms -valid 48:00

- sh submit_job.sh <br/>
  *This will create all the files necessary to submit jobs, but the command will not submit jobs!!*
   *Need to update instructions better* 

- ./crab_submit.sh
