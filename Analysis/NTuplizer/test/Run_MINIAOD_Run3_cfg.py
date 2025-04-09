import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
import os, sys

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--YEAR',             action='store',      default='2022',   type=str,      help="Which year? Options: 2022, 2022EE, 2023, 2023BPiX")
argParser.add_argument('--ERA',             action='store',      default='F',   type=str,      help="Which era?")
#argParser.add_argument('--IsDATA',             action='store',      default=False,   type=bool,      help="Is it DATA? Default:NO")
argParser.add_argument('--IsDATA', action='store_true', help="Is it DATA? Default:NO")
argParser.add_argument('--IsRun3',             action='store',      default=True,   type=bool,      help="Is Run3? Default:YES")
argParser.add_argument('--ReadJEC',             action='store',      default=True,   type=bool,      help="Read JEC from sqlite? Default:YES")
args = argParser.parse_args()

IsDATA = args.IsDATA  #bool(False)
#year
#YEAR = "2022"
YEAR = args.YEAR
#options:
#2022, 2022EE, 2023, 2023BPiX
#era
#ERA = "F" 
ERA = args.ERA
#options:
#2022: C, D
#2022EE: E, F, G
#2023: Cv1, Cv2, Cv3, Cv4
#2023BPiX: D
IsRun3 = args.IsRun3 #bool(True)
##object-specific booleans
ReadJEC = args.ReadJEC #bool(True)
ReclusterAK8Jets = bool(False)

from RecoJets.Configuration.RecoPFJets_cff import *
from RecoJets.Configuration.RecoGenJets_cff import ak4GenJets, ak8GenJets
#from RecoJets.JetProducers.SubJetParameters_cfi import SubJetParameters
#from RecoJets.JetProducers.PFJetParameters_cfi import *
#from RecoJets.JetProducers.GenJetParameters_cfi import *
#from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff import *
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets
from PhysicsTools.PatAlgos.tools.jetTools import *
from PhysicsTools.PatAlgos.patSequences_cff import *
from PhysicsTools.PatAlgos.patTemplate_cfg import *
from PhysicsTools.PatAlgos.tools.jetTools import *
from PhysicsTools.PatAlgos.slimming.metFilterPaths_cff import *

## Modified version of jetToolBox from https://github.com/cms-jet/jetToolbox
## Options for PUMethod: Puppi, CS, SK, CHS

# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

process = cms.Process("Combined")
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Conditions
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
#process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
# For Jets
process.load('RecoJets.Configuration.GenJetParticles_cff')
process.load('RecoJets.Configuration.RecoGenJets_cff')
process.load('RecoJets.JetProducers.TrackJetParameters_cfi')
process.load('RecoJets.JetProducers.PileupJetIDParams_cfi')
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

from RecoJets.Configuration.GenJetParticles_cff import *

if IsDATA:
    if YEAR=="2022":
        process.GlobalTag.globaltag = "130X_dataRun3_v2"
        JEC_tag = "Summer22_22Sep2023_RunCD_V2_DATA"
        JER_tag = 'Summer22_22Sep2023_JRV1_MC'
        JetVeto_tag = 'Summer22_23Sep2023_RunCD_v1'
    elif YEAR=="2022EE":
        process.GlobalTag.globaltag = "130X_dataRun3_PromptAnalysis_v1"
        JEC_tag = "Summer22EE_22Sep2023_Run"+ERA+"_V2_DATA" 
        JER_tag = 'Summer22EE_22Sep2023_JRV1_MC'
        JetVeto_tag = 'Summer22EE_23Sep2023_RunEFG_v1'
    elif YEAR=="2023":
        process.GlobalTag.globaltag = "130X_dataRun3_PromptAnalysis_v1"
        if ERA == "Cv4":
            JEC_tag = "Summer23Prompt23_RunCv4_V1_DATA"
        else:
            JEC_tag = "Summer23Prompt23_RunCv123_V1_DATA"
        JER_tag = 'Summer23Prompt23_RunCv123_JRV1_MC'       
        JetVeto_tag = "Summer23Prompt23_RunC_v1"
    elif YEAR=="2023BPiX":
        process.GlobalTag.globaltag = "130X_dataRun3_PromptAnalysis_v1"
        JEC_tag = "Summer23BPixPrompt23_RunD_V1_DATA" 
        JER_tag = 'Summer23BPixPrompt23_RunD_JRV1_MC'       
        JetVeto_tag = "Summer23BPixPrompt23_RunD_v1"
else:
    if YEAR=="2022":
        process.GlobalTag.globaltag = "130X_mcRun3_2022_realistic_v5"
        JEC_tag = 'Summer22_22Sep2023_V2_MC'
        JER_tag = 'Summer22_22Sep2023_JRV1_MC'
        JetVeto_tag = 'Summer22_23Sep2023_RunCD_v1'
    elif YEAR=="2022EE":
        process.GlobalTag.globaltag = "130X_mcRun3_2022_realistic_postEE_v6"
        JEC_tag = 'Summer22EE_22Sep2023_V2_MC'
        JER_tag = 'Summer22EE_22Sep2023_JRV1_MC'
        JetVeto_tag = 'Summer22EE_23Sep2023_RunEFG_v1'
    elif YEAR=="2023":
        process.GlobalTag.globaltag = "130X_mcRun3_2023_realistic_v14"
        JEC_tag = "Summer23Prompt23_V1_MC"
        if ERA=="Cv4":
            JER_tag = 'Summer23Prompt23_RunCv4_JRV1_MC'
        else:
            JER_tag = 'Summer23Prompt23_RunCv123_JRV1_MC'   
        JetVeto_tag = "Summer23Prompt23_RunC_v1"
    elif YEAR=="2023BPiX":
        process.GlobalTag.globaltag = "130X_mcRun3_2023_realistic_postBPix_v2"
        JEC_tag = "Summer23BPixPrompt23_V1_MC"
        JER_tag = 'Summer23BPixPrompt23_RunD_JRV1_MC'     
        JetVeto_tag = "Summer23BPixPrompt23_RunD_v1"
    else:
        process.GlobalTag.globaltag = "130X_mcRun3_2022_realistic_v5"
        JEC_tag = "Summer22_22Sep2023_RunCD_V2_DATA" 
        JER_tag = 'Summer22_22Sep2023_JRV1_MC'       
        

#JEC_tag = 'Summer22_22Sep2023_V2_MC'
#JER_tag = 'Summer22_22Sep2023_JRV1_MC'
#JetVeto_tag = 'Summer22_23Sep2023_RunCD_v1'
#process.GlobalTag.globaltag = "130X_mcRun3_2022_realistic_v5"

print("Configuration: YEAR",YEAR,"Era:",ERA,"Is Data?",IsDATA,"Is Run3?",IsRun3)
print("JEC tag:",JEC_tag)
print("JER tag:",JER_tag)
print("JetVeto tag:",JetVeto_tag)
print("globaltag",process.GlobalTag.globaltag)

##-------------------- Import the JEC services -----------------------
#process.load('JetMETCorrections.Configuration.DefaultJEC_cff')

from PhysicsTools.PatAlgos.tools.coreTools import *
process.load("PhysicsTools.PatAlgos.patSequences_cff")

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Input
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

inFiles = cms.untracked.vstring(
#'root://cms-xrd-global.cern.ch//store/mc/Run3Summer22MiniAODv4/ZZ_TuneCP5_13p6TeV_pythia8/MINIAODSIM/130X_mcRun3_2022_realistic_v5-v2/2530000/e319e433-9397-4985-aa9e-a30d46e29f24.root'
#'root://cms-xrd-global.cern.ch//store/mc/Run3Summer23BPixMiniAODv4/TTto2L2Nu_HT-500_NJet-7_TuneCP5_13p6TeV_powheg-pythia8/MINIAODSIM/130X_mcRun3_2023_realistic_postBPix_v2-v3/2820000/0115e762-15b7-40e7-949b-4b60b2770b76.root'
'root://cms-xrd-global.cern.ch//store/mc/Run3Summer22MiniAODv4/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/MINIAODSIM/130X_mcRun3_2022_realistic_v5-v2/2520000/00274cff-39eb-442e-bc3d-ae099a760f39.root'
#'root://xrootd-cms.infn.it//store/mc/RunIISummer20UL18MiniAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/04A0B676-D63A-6D41-B47F-F4CF8CBE7DB8.root'
   )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(3000))

#process.firstEvent = cms.untracked.PSet(input = cms.untracked.int32(5000))
process.source = cms.Source("PoolSource", fileNames = inFiles )

FastSIM = bool(False)

process.p = cms.Path()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Services
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.load('CommonTools.UtilAlgos.TFileService_cfi')

process.TFileService = cms.Service("TFileService",
fileName = cms.string('hist.root')             #largest data till April5,2016 
)

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *  
#needed for electron and photon ID mapping


from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask
def _addProcessAndTask(proc, label, module):
    task = getPatAlgosToolsTask(proc)
    addToProcessAndTask(label, module, proc, task)

#New PF-collection (if needed):   
def producePF(process) :
    from CommonTools.PileupAlgos.Puppi_cff import puppi
    puppi.useExistingWeights = True
    puppi.candName = "packedPFCandidates"
    puppi.vertexName = "offlineSlimmedPrimaryVertices"
    from LeptonLessPFProducer_cff import leptonLessPFProducer
    _addProcessAndTask(process,"leptonLessPFProducer",leptonLessPFProducer.clone())
    _addProcessAndTask(process,"leptonLesspuppi",puppi.clone(candName = cms.InputTag('leptonLessPFProducer')))

#Jets

process.patJets.addTagInfos = True


#JEC from sqlite file

if ReadJEC:
    from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
    #if options.jecDBFileRelPath:
    #    options.jecDBFile = options.jecDBFile;
    #else:
    cmssw_base_dir = os.getenv("CMSSW_BASE");
    #jecDBFile = cmssw_base_dir+"/src/Analysis/NTuplizer/data/JEC/"+(JEC_tag)+".db";
    jecDBFile = "JECfiles/SQLite/"+(JEC_tag)+".db"
    print("jecDBFile",jecDBFile)
    #print("Run with local DB file ",options.jecDBFile," correction tag ",options.jecDBRecord)

    jecDBRecord = (JEC_tag)+"_AK4PFPuppi"

    process.jec = cms.ESSource('PoolDBESSource',
            CondDBSetup,
            connect = cms.string('sqlite_file:'+jecDBFile),
            toGet = cms.VPSet(
                cms.PSet(
                    record = cms.string('JetCorrectionsRecord'),
                    tag    = cms.string("JetCorrectorParametersCollection_"+jecDBRecord),
                    label  = cms.untracked.string('AK4PFPuppi')
                ),
                cms.PSet(
                    record = cms.string('JetCorrectionsRecord'),
                    tag    = cms.string("JetCorrectorParametersCollection_"+jecDBRecord.replace("AK4","AK8")),
                    label  = cms.untracked.string('AK8PFPuppi')
                ),
            )
    )
    process.es_prefer_jec = cms.ESPrefer('PoolDBESSource', 'jec')

#JEC levels

jec_level = cms.vstring([])
if not IsDATA:
    jec_level = cms.vstring('L1FastJet','L2Relative','L3Absolute')
else:
    jec_level = cms.vstring('L1FastJet','L2Relative','L3Absolute','L2L3Residual');

# Jet tagging

# AK8 jets

deep_discriminators = [ "pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:TvsQCD",
                        "pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:WvsQCD",
                        "pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZvsQCD",
		                "pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHbbvsQCD",
		                "pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:bbvsLight",
		                "pfMassDecorrelatedParticleNetJetTags:probXbb",
		                "pfMassDecorrelatedParticleNetJetTags:probQCDbb",
		                "pfMassDecorrelatedParticleNetJetTags:probQCDcc",
		                "pfMassDecorrelatedParticleNetJetTags:probQCDb",
		                "pfMassDecorrelatedParticleNetJetTags:probQCDc",
		                "pfMassDecorrelatedParticleNetJetTags:probQCDothers",
		                "pfMassDecorrelatedParticleNetDiscriminatorsJetTags:XbbvsQCD",
		                "pfMassDecorrelatedParticleNetDiscriminatorsJetTags:XccvsQCD",
                        "pfMassDecorrelatedParticleNetDiscriminatorsJetTags:XqqvsQCD",
		                "pfParticleNetDiscriminatorsJetTags:TvsQCD",
		                "pfParticleNetDiscriminatorsJetTags:WvsQCD",
		                "pfParticleNetDiscriminatorsJetTags:ZvsQCD",
                        "pfParticleNetDiscriminatorsJetTags:HbbvsQCD",
                        "pfParticleNetDiscriminatorsJetTags:HccvsQCD",
                        "pfParticleNetDiscriminatorsJetTags:H4qvsQCD",
]

#Adding ParticleNet scores 
#from RecoBTag.MXNet.pfParticleNet_cff import _pfParticleNetJetTagsAll as pfParticleNetJetTagsAll  #Run2
from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsAll as pfParticleNetJetTagsAll #Run3
deep_discriminators += pfParticleNetJetTagsAll
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK8_cff import _pfParticleNetFromMiniAODAK8JetTagsAll as pfParticleNetFromMiniAODAK8JetTagsAll
deep_discriminators += pfParticleNetFromMiniAODAK8JetTagsAll

#Adding Global ParticleTransformer scores:
from RecoBTag.ONNXRuntime.pfGlobalParticleTransformerAK8_cff import _pfGlobalParticleTransformerAK8JetTagsAll as pfGlobalParticleTransformerAK8JetTagsAll
deep_discriminators += pfGlobalParticleTransformerAK8JetTagsAll

from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection

if ReclusterAK8Jets:

    def ak8JetSequences(process):

        from jetToolbox_cff import jetToolbox
        #jetToolbox( process, 'ak8', 'ak8JetSubs', 'noOutput', PUMethod='Puppi', dataTier="miniAOD", runOnMC=False, postFix='', newPFCollection=True, nameNewPFCollection='leptonLesspuppi', JETCorrPayload='AK8PFPuppi', JETCorrLevels = jec_level, addSoftDrop=True, addSoftDropSubjets=True, subJETCorrPayload='AK4PFPuppi', subJETCorrLevels=JETCorrLevels, addNsub=True, bTagDiscriminators=['None'])
        jetToolbox( process, 'ak8', 'ak8JetSubs', 'noOutput', PUMethod='Puppi', dataTier="miniAOD", runOnMC=False, postFix='', newPFCollection=True, nameNewPFCollection='leptonLesspuppi', JETCorrPayload='AK8PFPuppi', addSoftDrop=True, addSoftDropSubjets=True, subJETCorrPayload='AK4PFPuppi', bTagDiscriminators=['None'], subjetBTagDiscriminators=['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb'])

        updateJetCollection(
            process,
            jetSource = cms.InputTag('packedPatJetsAK8PFPuppiSoftDrop'),
            pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
            svSource = cms.InputTag('slimmedSecondaryVertices'),
            rParam = 0.8,
            jetCorrections = ('AK8PFPuppi', jec_level, 'None'),
            btagDiscriminators = deep_discriminators,
            postfix = 'SlimmedJetsAK8',
            printWarning = False
        )

    def defaultJetSequences(process):
        producePF(process)
        ak8JetSequences(process)

    defaultJetSequences(process)
    #-> gives jet collection from jetToolbox+updateJetCollection

else: 

    updateJetCollection(
        process,
        jetSource = cms.InputTag('slimmedJetsAK8'),
        pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
        svSource = cms.InputTag('slimmedSecondaryVertices'),
        rParam = 0.8,
        labelName = 'SlimmedJetsAK8',
        jetCorrections = ('AK8PFPuppi', jec_level, 'None'),
        btagDiscriminators = deep_discriminators 
    )   

# AK4 jets

#For QG likelihood 
from RecoJets.JetProducers.QGTagger_cfi import  QGTagger
process.qgtagger = QGTagger.clone(
    srcJets = cms.InputTag("slimmedJetsPuppi"),
    srcVertexCollection="offlineSlimmedPrimaryVertices"
)

#For pileup jet ID 
from RecoJets.JetProducers.PileupJetID_cfi import pileupJetId
process.pileupJetID= pileupJetId.clone(
    jets = cms.InputTag('slimmedJetsPuppi'),
    inputIsCorrected = False,
    applyJec = False,
    vertexes = cms.InputTag("offlineSlimmedPrimaryVertices"),
)

#Creating a new AK4 jet collection adding QG likelihood and pileup jet ID values
process.slimmedJetsPuppiWithInfo = cms.EDProducer("PATJetUserDataEmbedder",
    src = cms.InputTag("slimmedJetsPuppi"),
    userFloats = cms.PSet(
        qgLikelihood = cms.InputTag('qgtagger:qgLikelihood'),
        pileupJetId_fullDiscriminant = cms.InputTag('pileupJetID:fullDiscriminant'),
    )
)

#AAdding ParticleNet scores
pnetDiscriminators = []
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll as pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll as pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll
pnetDiscriminators += pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll
pnetDiscriminators += pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll
#Adding Robust ParticleTransformer scores
from RecoBTag.ONNXRuntime.pfParticleTransformerAK4_cff import _pfParticleTransformerAK4JetTagsAll as pfParticleTransformerAK4JetTagsAll
pnetDiscriminators += pfParticleTransformerAK4JetTagsAll

#updating AK4 jet collection with tagger scores
updateJetCollection(
    process,
    #jetSource = cms.InputTag('slimmedJetsPuppi'),
    jetSource = cms.InputTag('slimmedJetsPuppiWithInfo'),
    btagDiscriminators = pnetDiscriminators,
    #labelName = 'SlimmedJetsPuppi',
    postfix = 'WithPNetInfo',
    jetCorrections = ('AK4PFPuppi', jec_level,'None')
)

#from RecoJets.JetProducers.QGTagger_cfi import  QGTagger
#process.qgtagger = QGTagger.clone(
#    srcJets = "selectedUpdatedPatJetsWithPNetInfo",
#    srcVertexCollection="offlineSlimmedPrimaryVertices"
#    )

# DeltaR cleaning of AK4 jets (not used in the framework)
from PhysicsTools.PatAlgos.cleaningLayer1.jetCleaner_cfi import cleanPatJets
process.cleanJets = cms.EDProducer("PATJetCleaner",
    src = cms.InputTag("selectedUpdatedPatJetsWithPNetInfo"),
    preselection = cms.string(''),
    checkOverlaps = cms.PSet(
        muons = cms.PSet(
            src          = cms.InputTag("slimmedMuons"),
            algorithm    = cms.string("byDeltaR"),
            preselection = cms.string(""),
            deltaR       = cms.double(0.4),
            checkRecoComponents = cms.bool(False),
            pairCut             = cms.string(""),
            requireNoOverlaps   = cms.bool(True)
        ),
        electrons = cms.PSet(
            src          = cms.InputTag("slimmedElectrons"),
            algorithm    = cms.string("byDeltaR"),
            preselection = cms.string(""),
            deltaR       = cms.double(0.4),
            checkRecoComponents = cms.bool(False),
            pairCut             = cms.string(""),
            requireNoOverlaps   = cms.bool(True)
        )
    ),
    finalCut = cms.string('')
)

# MET 
# For CHS MET
from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
runMetCorAndUncFromMiniAOD(
    process,
    isData  = IsDATA,
    postfix = "Updated"
)
from PhysicsTools.PatAlgos.slimming.puppiForMET_cff import makePuppiesFromMiniAOD
makePuppiesFromMiniAOD( process, True )
# For Puppi
runMetCorAndUncFromMiniAOD(process,
                           isData=IsDATA,
                           metType="Puppi",
                           postfix="Puppi",
                           jetFlavor="AK4PFPuppi",
                           )
process.puppi.useExistingWeights = True

# Create a GEN collection with neutrinos

from RecoJets.Configuration.GenJetParticles_cff import genParticlesForJets
process.genParticlesForJets = genParticlesForJets.clone(
        src = cms.InputTag("packedGenParticles")
)
from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
process.ak4GenJetsWithNu = ak4GenJets.clone(
        src = "genParticlesForJets"
)

# For prefire correction #

from PhysicsTools.PatUtils.l1PrefiringWeightProducer_cfi import l1PrefiringWeightProducer
process.prefiringweight = l1PrefiringWeightProducer.clone(
	TheJets = cms.InputTag("slimmedJets"), #"updatedPatJetsUpdatedJEC"), #this should be the slimmedJets collection with up to date JECs 
	DataEraECAL = cms.string("None"), 
	DataEraMuon = cms.string("20172018"), 
	UseJetEMPt = cms.bool(False),
	PrefiringRateSystematicUnctyECAL = cms.double(0.2),
	PrefiringRateSystematicUnctyMuon = cms.double(0.2)
)

# Trigger (was needed for adding prescales in Run 3, but gives other problems, so not used for now) #

#from HLTrigger.HLTcore.hltEventAnalyzerAODDefault_cfi import hltEventAnalyzerAODDefault as _hltEventAnalyzerAODDefault
#hltEventAnalyzerAOD = _hltEventAnalyzerAODDefault.clone()
#from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
#stage2L1Trigger.toModify(hltEventAnalyzerAOD, stageL1Trigger = 2)

# Adding more information to muons

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import patMuons #.miniIsoParams #as _miniIsoParams

process.slimmedMuonsUpdated = cms.EDProducer("PATMuonUpdater",
    src = cms.InputTag("slimmedMuons"),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    computeMiniIso = cms.bool(False),
    fixDxySign = cms.bool(True),
    pfCandsForMiniIso = cms.InputTag("packedPFCandidates"),
    miniIsoParams = patMuons.miniIsoParams, #PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi.patMuons.miniIsoParams, # so they're in sync
    recomputeMuonBasicSelectors = cms.bool(False),
)

process.isoForMu = cms.EDProducer("MuonIsoValueMapProducer",
    src = cms.InputTag("slimmedMuonsUpdated"),
    relative = cms.bool(False),
    rho_MiniIso = cms.InputTag("fixedGridRhoFastjetAll"),
    EAFile_MiniIso = cms.FileInPath("PhysicsTools/NanoAOD/data/effAreaMuons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
)

process.ptRatioRelForMu = cms.EDProducer("MuonJetVarProducer",
    srcJet = cms.InputTag("slimmedJetsPuppi"),
    srcLep = cms.InputTag("slimmedMuonsUpdated"),
    srcVtx = cms.InputTag("offlineSlimmedPrimaryVertices"),
)

process.muonMVAID = cms.EDProducer("EvaluateMuonMVAID",
    src = cms.InputTag("slimmedMuonsUpdated"),
    weightFile =  cms.FileInPath("RecoMuon/MuonIdentification/data/mvaID.onnx"),
    backend = cms.string('ONNX'),
    name = cms.string("muonMVAID"),
    outputTensorName= cms.string("probabilities"),
    inputTensorName= cms.string("float_input"),
    outputNames = cms.vstring(["probGOOD", "wpMedium", "wpTight"]),
    batch_eval =cms.bool(True),
    outputFormulas = cms.vstring(["at(1)", "? at(1) > 0.08 ? 1 : 0", "? at(1) > 0.20 ? 1 : 0"]),
    variables = cms.VPSet(
        cms.PSet( name = cms.string("LepGood_global_muon"), expr = cms.string("isGlobalMuon")),
        cms.PSet( name = cms.string("LepGood_validFraction"), expr = cms.string("?innerTrack.isNonnull?innerTrack().validFraction:-99")),
        cms.PSet( name = cms.string("Muon_norm_chi2_extended")),
        cms.PSet( name = cms.string("LepGood_local_chi2"), expr = cms.string("combinedQuality().chi2LocalPosition")),
        cms.PSet( name = cms.string("LepGood_kink"), expr = cms.string("combinedQuality().trkKink")),
        cms.PSet( name = cms.string("LepGood_segmentComp"), expr = cms.string("segmentCompatibility")),
        cms.PSet( name = cms.string("Muon_n_Valid_hits_extended")),
        cms.PSet( name = cms.string("LepGood_n_MatchedStations"), expr = cms.string("numberOfMatchedStations()")),
        cms.PSet( name = cms.string("LepGood_Valid_pixel"), expr = cms.string("?innerTrack.isNonnull()?innerTrack().hitPattern().numberOfValidPixelHits():-99")),
        cms.PSet( name = cms.string("LepGood_tracker_layers"), expr = cms.string("?innerTrack.isNonnull()?innerTrack().hitPattern().trackerLayersWithMeasurement():-99")),
        cms.PSet( name = cms.string("LepGood_pt"), expr = cms.string("pt")),
        cms.PSet( name = cms.string("LepGood_eta"), expr = cms.string("eta")),
    )
)

process.slimmedMuonsWithUserData = cms.EDProducer("PATMuonUserDataEmbedder",
     src = cms.InputTag("slimmedMuonsUpdated"),
     userFloats = cms.PSet(
        miniIsoChg = cms.InputTag("isoForMu:miniIsoChg"),
        miniIsoAll = cms.InputTag("isoForMu:miniIsoAll"),
        #ptRatio = cms.InputTag("ptRatioRelForMu:ptRatio"),
        #ptRel = cms.InputTag("ptRatioRelForMu:ptRel"),
        #jetNDauChargedMVASel = cms.InputTag("ptRatioRelForMu:jetNDauChargedMVASel"),
        mvaIDMuon_wpMedium = cms.InputTag("muonMVAID:wpMedium"),
        mvaIDMuon_wpTight = cms.InputTag("muonMVAID:wpTight"),
        mvaIDMuon = cms.InputTag("muonMVAID:probGOOD")
     ),
     userCands = cms.PSet(
        #jetForLepJetVar = cms.InputTag("ptRatioRelForMu:jetForLepJetVar") # warning: Ptr is null if no match is found
     ),
)

# Adding more information to electrons

# Electron IDs for AOD/MINIAOD
switchOnVIDElectronIdProducer(process, DataFormat.MiniAOD)

# define which IDs to produce
el_id_modules = [
##    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff"
    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff",
    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff",
    "RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V2_cff",
    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_iso_V1_cff",
    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_noIso_V1_cff",
    "RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Winter22_122X_V1_cff",
##    "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff"
]
# Add them to the VID producer
for iModule in el_id_modules:

    setupAllVIDIdsInModule(process, iModule, setupVIDElectronSelection)

##PhysicsTools/NanoAOD/python/EleIsoValueMapProducer_cfi.py
process.isoForEle = cms.EDProducer("EleIsoValueMapProducer",
    src = cms.InputTag("slimmedElectrons"),
    relative = cms.bool(False),
    rho_MiniIso = cms.InputTag("fixedGridRhoFastjetAll"),
    rho_PFIso = cms.InputTag("fixedGridRhoFastjetAll"),
    EAFile_MiniIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Run3_Winter22/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_122X.txt"),
    EAFile_PFIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Run3_Winter22/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_122X.txt"),
)

process.isoForEleFall17V2 = process.isoForEle.clone(
    EAFile_MiniIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
    EAFile_PFIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
)

process.ptRatioRelForEle = cms.EDProducer("ElectronJetVarProducer",
    srcJet = cms.InputTag("slimmedJetsPuppi"),
    srcLep = cms.InputTag("slimmedElectrons"),
    srcVtx = cms.InputTag("offlineSlimmedPrimaryVertices"),
)

electron_id_modules_WorkingPoints_nanoAOD = cms.PSet(
    modules = cms.vstring(
        'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff',
        # HZZ ID
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer16UL_ID_ISO_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer17UL_ID_ISO_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer18UL_ID_ISO_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Winter22_HZZ_V1_cff',
        # Fall17: need to include the modules too to make sure they are run
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V2_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff',
        # Run3Winter22:
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Winter22_122X_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_iso_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_noIso_V1_cff',
    ),
    WorkingPoints = cms.vstring(
        "egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-veto",
        "egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-loose",
        "egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-medium",
        "egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-tight",
    )
)

electron_id_modules_WorkingPoints_nanoAOD_Run2 = cms.PSet(
    modules = cms.vstring(
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V2_cff',
        'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff',
        # HZZ ID
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer16UL_ID_ISO_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer17UL_ID_ISO_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer18UL_ID_ISO_cff',
    ),
    WorkingPoints = cms.vstring(
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-veto",
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-loose",
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-medium",
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-tight",
    )
)

process.bitmapVIDForEle = cms.EDProducer("EleVIDNestedWPBitmapProducer",
    src = cms.InputTag("slimmedElectrons"),
    srcForID = cms.InputTag("reducedEgamma","reducedGedGsfElectrons"),
    WorkingPoints = electron_id_modules_WorkingPoints_nanoAOD.WorkingPoints,
)

process.bitmapVIDForEleFall17V2 = process.bitmapVIDForEle.clone(
    WorkingPoints = electron_id_modules_WorkingPoints_nanoAOD_Run2.WorkingPoints
)

# Load the producer for MVA IDs. Make sure it is also added to the sequence!
process.load("RecoEgamma.ElectronIdentification.ElectronMVAValueMapProducer_cfi")

process.slimmedElectronsWithUserData = cms.EDProducer("PATElectronUserDataEmbedder",
    src = cms.InputTag("slimmedElectrons"),
    parentSrcs = cms.VInputTag("reducedEgamma:reducedGedGsfElectrons"),
    userFloats = cms.PSet(
        
        mvaIso_Fall17V2 = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17IsoV2Values"),
        mvaNoIso_Fall17V2 = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17NoIsoV2Values"),
        mvaIso = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2RunIIIWinter22IsoV1Values"),
        mvaNoIso = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2RunIIIWinter22NoIsoV1Values"),
        mvaHZZIso = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Winter22HZZV1Values"),

        miniIsoChg = cms.InputTag("isoForEle:miniIsoChg"),
        miniIsoAll = cms.InputTag("isoForEle:miniIsoAll"),
        PFIsoChg = cms.InputTag("isoForEle:PFIsoChg"),
        PFIsoAll = cms.InputTag("isoForEle:PFIsoAll"),
        PFIsoAll04 = cms.InputTag("isoForEle:PFIsoAll04"),

        miniIsoChg_Fall17V2 = cms.InputTag("isoForEleFall17V2:miniIsoChg"),
        miniIsoAll_Fall17V2 = cms.InputTag("isoForEleFall17V2:miniIsoAll"),
        PFIsoChg_Fall17V2 = cms.InputTag("isoForEleFall17V2:PFIsoChg"),
        PFIsoAll_Fall17V2 = cms.InputTag("isoForEleFall17V2:PFIsoAll"),
        PFIsoAll04_Fall17V2 = cms.InputTag("isoForEleFall17V2:PFIsoAll04"),

        #ptRatio = cms.InputTag("ptRatioRelForEle:ptRatio"),
        #ptRel = cms.InputTag("ptRatioRelForEle:ptRel"),
        #jetNDauChargedMVASel = cms.InputTag("ptRatioRelForEle:jetNDauChargedMVASel"),
    ),
    userIntFromBools = cms.PSet(
        mvaIso_Fall17V2_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wp90"),
        mvaIso_Fall17V2_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wp80"),
        mvaIso_Fall17V2_WPL = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wpLoose"),
        mvaIso_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-RunIIIWinter22-iso-V1-wp90"),
        mvaIso_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-RunIIIWinter22-iso-V1-wp80"),
        mvaNoIso_Fall17V2_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wp90"),
        mvaNoIso_Fall17V2_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wp80"),
        mvaNoIso_Fall17V2_WPL = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wpLoose"),
        mvaNoIso_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-RunIIIWinter22-noIso-V1-wp90"),
        mvaNoIso_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-RunIIIWinter22-noIso-V1-wp80"),
        #mvaIso_WPHZZ = cms.InputTag("egmGsfElectronIDs:mvaEleID-Winter22-HZZ-V1"),
        
        cutBasedID_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-veto"),
        cutBasedID_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-loose"),
        cutBasedID_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-medium"),
        cutBasedID_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-tight"),
        cutBasedID_Fall17V2_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-veto"),
        cutBasedID_Fall17V2_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-loose"),
        cutBasedID_Fall17V2_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-medium"),
        cutBasedID_Fall17V2_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-tight"),
        #cutBasedID_HEEP = cms.InputTag("egmGsfElectronIDs:heepElectronID-HEEPV70"),

    ),
    userInts = cms.PSet(
        VIDNestedWPBitmap = cms.InputTag("bitmapVIDForEle"),
        VIDNestedWPBitmap_Fall17V2 = cms.InputTag("bitmapVIDForEleFall17V2"),
        #VIDNestedWPBitmapHEEP = cms.InputTag("bitmapVIDForEleHEEP"),
        #seedGain = cms.InputTag("seedGainEle"),
    ),
    userCands = cms.PSet(
        #jetForLepJetVar = cms.InputTag("ptRatioRelForEle:jetForLepJetVar") # warning: Ptr is null if no match is found
    ),
)

# Adding photon IDs

switchOnVIDPhotonIdProducer(process, DataFormat.MiniAOD)

pho_id_modules = [
    "RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff",
    "RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff",
    'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Winter22_122X_V1_cff',
    'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_RunIIIWinter22_122X_V1_cff',
]

for iModule in pho_id_modules:

    setupAllVIDIdsInModule(process, iModule, setupVIDPhotonSelection)


# E-Gamma scale and smearing corrections 
from Analysis.NTuplizer.EgammaPostRecoTools import setupEgammaPostRecoSeq
#setupEgammaPostRecoSeq(process,era='2018-Prompt')  
'''
setupEgammaPostRecoSeq(process,
    runEnergyCorrections=False, #Only set it to True if you want to pick the latest Scale&Smearing corrections.
    runVID=True,
    era='2022-Prompt',
    eleIDModules=[  'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_iso_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_noIso_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Winter22_122X_V1_cff'
                ]
)
'''

# Analyzer #

process.mcjets =  cms.EDAnalyzer('Leptop',
    #basic things (year, data/MC, Run2 or Run3)
    Data =  cms.untracked.bool(IsDATA),
	MonteCarlo =  cms.untracked.bool(not IsDATA),
	FastSIM =  cms.untracked.bool(False),
    YEAR = cms.untracked.string(YEAR),
    UltraLegacy =  cms.untracked.bool(False), 
    isRun3 =  cms.untracked.bool(IsRun3),
    #output filename
    RootFileName = cms.untracked.string('rootuple.root'),
    #analysis specific
	SoftDrop_ON =  cms.untracked.bool(True),
	add_prefireweights =  cms.untracked.bool(False),
#	PFJetsAK8 = cms.InputTag("slimmedJetsAK8"),
#   PFJetsAK8 = cms.InputTag("selectedPatJetsAK8PFPuppiSoftDropPacked","SubJets","Combined"),
#   PFJetsAK8 = cms.InputTag("selectedPatJetsAK8PFPuppi","","Combined"),+ssDecorrelatedParticleNetJetTagsSlimmedJetsAK8
#	PFJetsAK8 = cms.InputTag("updatedPatJetsTransientCorrectedSlimmedJetsAK8"),#"","PAT"),
    PFJetsAK8 = cms.InputTag("selectedUpdatedPatJetsSlimmedJetsAK8"),
#    PFJetsAK8 = cms.InputTag("slimmedJetsAK8WithGloParT"),
	minAK8JetPt = cms.untracked.double(180.),
    Subtract_Lepton_fromAK8 = cms.untracked.bool(True),
    store_fatjet_constituents = cms.untracked.bool(False),
	softdropmass  = cms.untracked.string("ak8PFJetsSoftDropMass"),#ak8PFJetsPuppiSoftDropMass"),#('ak8PFJetsPuppiSoftDropMass'),
	tau1  = cms.untracked.string("NjettinessAK8Puppi:tau1"),#'NjettinessAK8Puppi:tau1'),
	tau2  = cms.untracked.string("NjettinessAK8Puppi:tau2"),#'NjettinessAK8Puppi:tau2'),
	tau3  = cms.untracked.string("NjettinessAK8Puppi:tau3"),#'NjettinessAK8Puppi:tau3'),
	subjets  = cms.untracked.string('SoftDropPuppi'),#("SoftDrop"),#'SoftDropPuppi'),
#    subjets  = cms.untracked.string('SoftDrop'),
#	subjets = cms.untracked.string("slimmedJetsAK8PFPuppiSoftDropPacked"),
	toptagger_DAK8 = cms.untracked.string("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:TvsQCD"),
	Wtagger_DAK8 = cms.untracked.string("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:WvsQCD"),
	Ztagger_DAK8 = cms.untracked.string("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZvsQCD"),
	Htagger_DAK8 = cms.untracked.string("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHbbvsQCD"),
	bbtagger_DAK8 = cms.untracked.string("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:bbvsLight"),	
	toptagger_PNet = cms.untracked.string("pfParticleNetDiscriminatorsJetTags:TvsQCD"),
	Wtagger_PNet = cms.untracked.string("pfParticleNetDiscriminatorsJetTags:WvsQCD"),
	Ztagger_PNet = cms.untracked.string("pfParticleNetDiscriminatorsJetTags:ZvsQCD"),
    Hbbtagger_PNet = cms.untracked.string("pfParticleNetDiscriminatorsJetTags:HbbvsQCD"),
    Hcctagger_PNet = cms.untracked.string("pfParticleNetDiscriminatorsJetTags:HccvsQCD"),
    H4qtagger_PNet = cms.untracked.string("pfParticleNetDiscriminatorsJetTags:H4qvsQCD"),
	#Xbbtagger_PNet = cms.untracked.string("pfMassDecorrelatedParticleNetDiscriminatorsJetTags:XbbvsQCD"),
	#Xcctagger_PNet = cms.untracked.string("pfMassDecorrelatedParticleNetDiscriminatorsJetTags:XccvsQCD"),
	#Xqqtagger_PNet = cms.untracked.string("pfMassDecorrelatedParticleNetDiscriminatorsJetTags:XqqvsQCD"),
    Xbbtagger_PNet = cms.untracked.string("pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HbbvsQCD"),
    Xcctagger_PNet = cms.untracked.string("pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HccvsQCD"),
    Xqqtagger_PNet = cms.untracked.string("pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HqqvsQCD"),
    Xggtagger_PNet = cms.untracked.string("pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HggvsQCD"),
    Xtetagger_PNet = cms.untracked.string("pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HtevsQCD"),
    Xtmtagger_PNet = cms.untracked.string("pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HtmvsQCD"),
    Xtttagger_PNet = cms.untracked.string("pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HttvsQCD"),
	QCDtagger_PNet = cms.untracked.string("pfMassDecorrelatedParticleNetDiscriminatorsJetTags:QCD"),
    QCD0HFtagger_PNet = cms.untracked.string("pfParticleNetFromMiniAODAK8JetTags:probQCD0hf"),
    QCD1HFtagger_PNet = cms.untracked.string("pfParticleNetFromMiniAODAK8JetTags:probQCD1hf"),
    QCD2HFtagger_PNet = cms.untracked.string("pfParticleNetFromMiniAODAK8JetTags:probQCD2hf"),
    Xbbtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probXbb"),
    Xcctagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probXcc"),
    Xcstagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probXcs"),
    Xqqtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probXqq"),
    TopbWqqtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probTopbWqq"),
    TopbWqtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probTopbWq"),
    TopbWevtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probTopbWev"),
    TopbWmvtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probTopbWmv"),
    TopbWtauvtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probTopbWtauhv"),
    QCDtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probQCD"),
    XWW4qtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probXWW4q"),
    XWW3qtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probXWW3q"),
    XWWqqevtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probXWWqqev"),
    XWWqqmvtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probXWWqqmv"),
    TvsQCDtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probWithMassTopvsQCD"),
    WvsQCDtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probWithMassWvsQCD"),
    ZvsQCDtagger_PartT = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:probWithMassZvsQCD"),
    mass_cor_PNet = cms.untracked.string("pfParticleNetFromMiniAODAK8JetTags:masscorr"),
    mass_cor_PartT_genertic = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:massCorrGeneric"),
    mass_cor_PartT_twoprong = cms.untracked.string("pfGlobalParticleTransformerAK8JetTags:massCorrX2p"),
	#PFJetsAK4 = cms.InputTag("slimmedJetsPuppi"), 
	PFJetsAK4 = cms.InputTag("selectedUpdatedPatJetsWithPNetInfo"),
    #PFJetsAK4 = cms.InputTag("cleanJets"),
    minJetPt = cms.untracked.double(25.),
	maxEta = cms.untracked.double(3.),
    Subtract_Lepton_fromAK4 = cms.untracked.bool(True),
    Read_btagging_SF = cms.untracked.bool(False),
    #GENJets
	GENJetAK8 = cms.InputTag("slimmedGenJetsAK8"),
	GENJetAK4 = cms.InputTag("slimmedGenJets"),
    GENJetAK4wNu = cms.InputTag("ak4GenJetsWithNu"),
    minGenJetPt = cms.untracked.double(15.), 
    minGenAK8JetPt = cms.untracked.double(150.),
	maxGenJetEta = cms.untracked.double(5.),
	#Muons = cms.InputTag("slimmedMuons"),
    Muons = cms.InputTag("slimmedMuonsWithUserData"),
	minMuonPt  = cms.untracked.double(10.),
	EAFile_MuonMiniIso = cms.FileInPath("PhysicsTools/NanoAOD/data/effAreaMuons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
	#Electrons = cms.InputTag("slimmedElectrons"),
	Electrons = cms.InputTag("slimmedElectronsWithUserData"),
    minElectronPt   = cms.untracked.double(10.),
    electronID_cutbased_loose   = cms.string('cutBasedElectronID-RunIIIWinter22-V1-loose'),
    electronID_cutbased_medium   = cms.string('cutBasedElectronID-RunIIIWinter22-V1-medium'),
    electronID_cutbased_tight   = cms.string('cutBasedElectronID-RunIIIWinter22-V1-tight'),
	electronID_isowp90        = cms.string('mvaEleID-RunIIIWinter22-iso-V1-wp90'),
    electronID_noisowp90      = cms.string('mvaEleID-RunIIIWinter22-noIso-V1-wp90'),
    electronID_isowp80        = cms.string('mvaEleID-RunIIIWinter22-iso-V1-wp80'),
    electronID_noisowp80      = cms.string('mvaEleID-RunIIIWinter22-noIso-V1-wp80'),
    electronID_isowploose        = cms.string('mvaEleID-RunIIIWinter22-iso-V1-wpLoose'),
    electronID_noisowploose      = cms.string('mvaEleID-RunIIIWinter22-noIso-V1-wpLoose'),
	electronID_isowp90_Fall17        = cms.string('mvaEleID-Fall17-iso-V2-wp90'),
    electronID_noisowp90_Fall17      = cms.string('mvaEleID-Fall17-noIso-V2-wp90'),
    electronID_isowp80_Fall17        = cms.string('mvaEleID-Fall17-iso-V2-wp80'),
    electronID_noisowp80_Fall17      = cms.string('mvaEleID-Fall17-noIso-V2-wp80'),
    electronID_isowploose_Fall17      = cms.string('mvaEleID-Fall17-iso-V2-wpLoose'),
    electronID_noisowploose_Fall17      = cms.string('mvaEleID-Fall17-noIso-V2-wpLoose'),
    #Run 2
    #EAFile_EleMiniIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
	#EAFile_ElePFIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
	#Run 3
    EAFile_EleMiniIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Run3_Winter22/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_122X.txt"),
    EAFile_ElePFIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Run3_Winter22/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_122X.txt"),
    #Photon
    Photons = cms.InputTag("slimmedPhotons"),
	minPhotonPt  = cms.untracked.double(20.),
    PhoID_RunIIIWinter22V1_WP90 = cms.string("mvaPhoID-RunIIIWinter22-v1-wp90"),
    PhoID_RunIIIWinter22V1_WP80 = cms.string("mvaPhoID-RunIIIWinter22-v1-wp80"),
	PhoID_FallV2_WP90 = cms.string("mvaPhoID-RunIIFall17-v2-wp90"),
    PhoID_FallV2_WP80 = cms.string("mvaPhoID-RunIIFall17-v2-wp80"),
    PhoID_SpringV1_WP90 = cms.string("mvaPhoID-Spring16-nonTrig-V1-wp90"),
    PhoID_SpringV1_WP80 = cms.string("mvaPhoID-Spring16-nonTrig-V1-wp80"),
	label_mvaPhoID_FallV2_Value = cms.InputTag("photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v2Values"),
	#label_mvaPhoID_FallV2_WP90 = cms.InputTag("egmPhotonIDs:mvaPhoID-RunIIFall17-v2-wp90"),
    #label_mvaPhoID_FallV2_WP80 = cms.InputTag("egmPhotonIDs:mvaPhoID-RunIIFall17-v2-wp80"),
	Taus = cms.InputTag("slimmedTaus"),#Updated"),
	minTauPt = cms.untracked.double(25.),
    maxTauEta = cms.untracked.double(2.3),
    #MET
	PFMet = cms.InputTag("slimmedMETsUpdated"),          #"Updated" comes from postfix in runMetCorAndUncFromMiniAOD
    PuppiMet = cms.InputTag("slimmedMETsPuppi"),  #"Updated" comes from postfix in runMetCorAndUncFromMiniAOD
    GENMet  = cms.InputTag("genMetTrue","","SIM"),
    #GEN Particles & Flavors
	GenParticles = cms.InputTag("prunedGenParticles"),#("prunedGenParticles"),#("packedGenParticles"),
	jetFlavourInfos = cms.InputTag("slimmedGenJetsFlavourInfos"),
	#jetFlavourInfos = cms.InputTag("genJetAK8FlavourAssociation"),
    #PF candidates
	pfCands = cms.InputTag("packedPFCandidates"),
    #Beam spot, vertices, pileup, energy density
    Beamspot = cms.InputTag("offlineBeamSpot"),
    PrimaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    SecondaryVertices = cms.InputTag("slimmedSecondaryVertices"),
	slimmedAddPileupInfo = cms.InputTag("slimmedAddPileupInfo"),
    PFRho = cms.InputTag("fixedGridRhoFastjetAll"),
    #Trigger
    bits = cms.InputTag("TriggerResults","","HLT"),
    prescales = cms.InputTag("patTrigger","","RECO"),
    TriggerObjects = cms.InputTag("slimmedPatTrigger"),
    L1_GtHandle = cms.InputTag("gtStage2Digis"),
    #MET Filter
    MET_Filters = cms.InputTag("TriggerResults::PAT"),
    #GEN info
	Generator = cms.InputTag("generator"),
    LHEEventProductInputTag = cms.InputTag('externalLHEProducer'),
    GenEventProductInputTag = cms.InputTag('generator'),
	nPDFsets = cms.untracked.uint32(103),
    #JEC 
	jecL1FastFileAK4          = cms.string('JECfiles/'+JEC_tag+'/'+JEC_tag+'_L1FastJet_AK4PFPuppi.txt'),
    jecL1FastFileAK8          = cms.string('JECfiles/'+JEC_tag+'/'+JEC_tag+'_L1FastJet_AK8PFPuppi.txt'),
    jecL2RelativeFileAK4      = cms.string('JECfiles/'+JEC_tag+'/'+JEC_tag+'_L2Relative_AK4PFPuppi.txt'),
    jecL2RelativeFileAK8      = cms.string('JECfiles/'+JEC_tag+'/'+JEC_tag+'_L2Relative_AK8PFPuppi.txt'),
    jecL3AbsoluteFileAK4      = cms.string('JECfiles/'+JEC_tag+'/'+JEC_tag+'_L3Absolute_AK4PFPuppi.txt'),
    jecL3AbsoluteFileAK8      = cms.string('JECfiles/'+JEC_tag+'/'+JEC_tag+'_L3Absolute_AK8PFPuppi.txt'),
    jecL2L3ResidualFileAK4    = cms.string('JECfiles/'+JEC_tag+'/'+JEC_tag+'_L2L3Residual_AK4PFPuppi.txt'),
    jecL2L3ResidualFileAK8    = cms.string('JECfiles/'+JEC_tag+'/'+JEC_tag+'_L2L3Residual_AK8PFPuppi.txt'),
    JECUncFileAK4 = cms.string('JECfiles/'+JEC_tag+'/'+JEC_tag+'_UncertaintySources_AK4PFPuppi.txt'),
	JECUncFileAK8 = cms.string('JECfiles/'+JEC_tag+'/'+JEC_tag+'_UncertaintySources_AK8PFPuppi.txt'),
    #JER
    PtResoFileAK4  = cms.string('JERfiles/'+JER_tag+'/'+JER_tag+'_PtResolution_AK4PFPuppi.txt'),
    PtResoFileAK8  = cms.string('JERfiles/'+JER_tag+'/'+JER_tag+'_PtResolution_AK8PFPuppi.txt'),
    PtSFFileAK4 = cms.string('JERfiles/'+JER_tag+'/'+JER_tag+'_SF_AK4PFPuppi.txt'),
    PtSFFileAK8 = cms.string('JERfiles/'+JER_tag+'/'+JER_tag+'_SF_AK8PFPuppi.txt'),
    #Jet Veto Map
    JetVetoMap = cms.string('JetVetoMaps/'+JetVeto_tag+'.root'),
    #Btag SF file
	BtagSFFile_DeepCSV = cms.string("BtagRecommendation106XUL18/DeepCSV_106XUL18SF_V1p1.csv"),
	BtagSFFile_DeepFlav = cms.string("BtagRecommendation106XUL18/DeepJet_106XUL18SF_V1p1.csv"),
	RochcorFolder = cms.string("roccor.Run2.v5/"),
    #Boolean for storing objects & variables
    store_electrons = cms.untracked.bool(True),
    store_muons = cms.untracked.bool(True),
    store_photons = cms.untracked.bool(False),
    store_ak4jets = cms.untracked.bool(True),
    store_ak8jets = cms.untracked.bool(True),
    store_taus = cms.untracked.bool(True),
    store_CHS_met = cms.untracked.bool(False),
    store_PUPPI_met = cms.untracked.bool(True),
    store_jet_id_variables = cms.untracked.bool(False),
    store_muon_id_variables = cms.untracked.bool(False),
    store_additional_muon_id_variables = cms.untracked.bool(False),
    store_electron_id_variables = cms.untracked.bool(False),
    store_additional_electron_id_variables = cms.untracked.bool(False),
    store_electron_scalnsmear =  cms.untracked.bool(False),
    store_photon_id_variables = cms.untracked.bool(False),
    store_tau_id_variables = cms.untracked.bool(False),
)

#===== MET Filters ==

process.load('RecoMET.METFilters.primaryVertexFilter_cfi')
process.primaryVertexFilter.vertexCollection = cms.InputTag("offlineSlimmedPrimaryVertices")

if not FastSIM:
	process.load('RecoMET.METFilters.globalSuperTightHalo2016Filter_cfi')

process.load('CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi')
process.load('CommonTools.RecoAlgos.HBHENoiseFilter_cfi')
process.HBHENoiseFilterResultProducerNoMinZ = process.HBHENoiseFilterResultProducer.clone(minZeros = cms.int32(99999))

process.load('RecoMET.METFilters.EcalDeadCellTriggerPrimitiveFilter_cfi')
process.EcalDeadCellTriggerPrimitiveFilter.tpDigiCollection = cms.InputTag("ecalTPSkimNA")

process.load('RecoMET.METFilters.BadPFMuonFilter_cfi')
process.BadPFMuonFilter.muons = cms.InputTag("slimmedMuons")
process.BadPFMuonFilter.PFCandidates = cms.InputTag("packedPFCandidates")
process.BadPFMuonFilter.vtx = cms.InputTag("offlineSlimmedPrimaryVertices") 
process.BadPFMuonFilter.taggingMode = cms.bool(True)

process.load('RecoMET.METFilters.BadPFMuonDzFilter_cfi')
process.BadPFMuonDzFilter.muons = cms.InputTag("slimmedMuons")
process.BadPFMuonDzFilter.PFCandidates = cms.InputTag("packedPFCandidates")
process.BadPFMuonDzFilter.vtx = cms.InputTag("offlineSlimmedPrimaryVertices")
process.BadPFMuonDzFilter.taggingMode = cms.bool(True)

process.load('RecoMET.METFilters.BadChargedCandidateFilter_cfi')
process.BadChargedCandidateFilter.muons = cms.InputTag("slimmedMuons")
process.BadChargedCandidateFilter.PFCandidates = cms.InputTag("packedPFCandidates")

process.load('RecoMET.METFilters.eeBadScFilter_cfi')
#process.eeBadScFilter.EERecHitSource = cms.InputTag('reducedEgamma','reducedEERecHits')

process.load('RecoMET.METFilters.ecalBadCalibFilter_cfi')

process.load('RecoMET.METFilters.hfNoisyHitsFilter_cfi')

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('eventoutput.root'),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

#from RecoMET.METFilters.metFilters_cff import EcalDeadCellTriggerPrimitiveFilter, eeBadScFilter, ecalLaserCorrFilter, EcalDeadCellBoundaryEnergyFilter, ecalBadCalibFilter

if FastSIM:
	process.allMetFilterPaths=cms.Sequence(process.primaryVertexFilter*
	#process.EcalDeadCellTriggerPrimitiveFilter*
	process.BadPFMuonFilter*
	process.BadPFMuonDzFilter#*
	#process.ecalBadCalibFilter
        )
else:
	process.allMetFilterPaths=cms.Sequence(
    process.primaryVertexFilter
	*process.globalSuperTightHalo2016Filter
	#*process.EcalDeadCellTriggerPrimitiveFilter
	*process.BadPFMuonFilter
	*process.BadPFMuonDzFilter
	#*process.eeBadScFilter
	#*process.hfNoisyHitsFilter
        )

if ReclusterAK8Jets:

    process.jetSeq=cms.Sequence(
        process.patJetCorrFactorsWithPNetInfo
        +process.updatedPatJetsWithPNetInfo
        +process.selectedUpdatedPatJetsWithPNetInfo
    )

else:

    process.jetSeq=cms.Sequence(
        process.patJetCorrFactorsSlimmedJetsAK8
        +process.updatedPatJetsSlimmedJetsAK8
        +process.selectedUpdatedPatJetsSlimmedJetsAK8
        #+process.cleanJets
        +process.patJetCorrFactorsWithPNetInfo
        +process.updatedPatJetsWithPNetInfo
        +process.selectedUpdatedPatJetsWithPNetInfo
        #+process.qgtagger
    )

import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
updatedTauName = "slimmedTausUpdated"
'''
tauIdEmbedder = tauIdConfig.TauIDEmbedder(process, cms, debug = True, updatedTauName = updatedTauName,
	toKeep = [ "2017v2",
        #       "dR0p32017v2", 
        #       "newDM2017v2", 
        "deepTau2017v2p1",
        "againstEle2018"\
        ]
)
tauIdEmbedder.runTauID()
'''

process.edTask = cms.Task()
for key in process.__dict__.keys():
    if(type(getattr(process,key)).__name__=='EDProducer' or type(getattr(process,key)).__name__=='EDFilter') :
        process.edTask.add(getattr(process,key))

process.p = cms.Path(
              process.egmPhotonIDSequence 
#		     *process.allMetFilterPaths    #Not needed since storing booleans in ntuple
#		     *process.egmGsfElectronIDSequence*
             *process.electronMVAValueMapProducer
#		     *process.rerunMvaIsolationSequence*getattr(process,updatedTauName) #not needed anymore
#		     *process.slimmedTausUpdated  # this also works for tauID
#		     *process.egammaPostRecoSeq  # throws errors, switched off for now
		     #*process.prefiringweight
#             *process.qgtagger
		     *process.jetSeq
		     *process.mcjets
             ,process.edTask
		     )

process.schedule = cms.Schedule(process.p)

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )
process.options.allowUnscheduled = cms.untracked.bool(True)

from PhysicsTools.PatAlgos.tools.helpers  import getPatAlgosToolsTask
process.patAlgosToolsTask = getPatAlgosToolsTask(process)
process.pathRunPatAlgos = cms.Path(process.patAlgosToolsTask)

#process.options.numberOfThreads=cms.untracked.uint32(2)
#process.options.numberOfStreams=cms.untracked.uint32(0)
