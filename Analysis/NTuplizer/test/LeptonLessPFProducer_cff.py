# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

# Producer #

leptonLessPFProducer =  cms.EDProducer('LeptonLessPFProducer',

	 maxEta = cms.untracked.double(2.5),

	 Muons = cms.InputTag("slimmedMuons"),
	 minmuPt  = cms.untracked.double(10.),

	 Electrons = cms.InputTag("slimmedElectrons"),
	 minePt   = cms.untracked.double(10.),
	 electronID_isowp90        = cms.string('mvaEleID-Fall17-iso-V2-wp90'),
         electronID_noisowp90      = cms.string('mvaEleID-Fall17-noIso-V2-wp90'),
         electronID_isowp80        = cms.string('mvaEleID-Fall17-iso-V2-wp80'),
         electronID_noisowp80      = cms.string('mvaEleID-Fall17-noIso-V2-wp80'),

	 pfCands = cms.InputTag("packedPFCandidates"),
         PrimaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
         SecondaryVertices = cms.InputTag("slimmedSecondaryVertices"),
	 slimmedAddPileupInfo = cms.InputTag("slimmedAddPileupInfo")
)
