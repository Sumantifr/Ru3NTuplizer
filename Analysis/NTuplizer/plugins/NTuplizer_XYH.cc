// -*- C++ -*-
//
// Package:    Analysis/NTuplizer
// Class:      NTuplizer
// 
/**\class NTuplizer_XYH NTuplizer_XYH.cc 
   
   Description: [one line class summary]
   
   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Suman Chatterjee
//         Created:  Fri, 1 Oct 2021 16:22:44 GMT
//

//Twikis used:
/*
Prefiring weights: https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1PrefiringWeightRecipe#2018_UL 
Electron MVA ID: https://twiki.cern.ch/twiki/bin/view/CMS/MultivariateElectronIdentificationRun2
Photon MVA ID: https://twiki.cern.ch/twiki/bin/view/CMS/MultivariatePhotonIdentificationRun2 
Main EGamma: https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaIDRecipesRun2#MVA_based_electron_Identificatio
JEC: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JECUncertaintySources
DeepAKX: https://twiki.cern.ch/twiki/bin/viewauth/CMS/DeepAKXTagging
Btag SF (recipe): https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagCalibration
Btag SF (2018UL): https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18
Rochester correction: https://gitlab.cern.ch/akhukhun/roccor
*/

// system include files
#include <memory>
// user include files
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
//#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"//Run 2
#include "CommonTools/Egamma/interface/EffectiveAreas.h" //Run 3
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetMatching/interface/JetFlavourInfo.h"
#include "DataFormats/JetMatching/interface/JetFlavourInfoMatching.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/JetID.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"
#include "TAxis.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "TRandom.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
//#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
//#include "GeneratorInterface/Pythia8Interface/plugins/ReweightUserHooks.h"
#include "fastjet/contrib/SoftDrop.hh"
#include "fastjet/contrib/Nsubjettiness.hh"
#include <string>
#include <iostream>
#include <fstream>
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
//#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
//#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include  "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2.h"
//#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReportEntry.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReport.h"
//#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
//L1
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "JetMETCorrections/Modules/interface/JetResolution.h"
#include "CondFormats/JetMETObjects/interface/JetResolutionObject.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Utilities/interface/typelookup.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "CondFormats/BTauObjects/interface/BTagCalibration.h"
#include "CondFormats/BTauObjects/interface/BTagEntry.h"
#include "CondFormats/BTauObjects/interface/BTagCalibration.h"
#include "CondTools/BTau/interface/BTagCalibrationReader.h"
#include "fastjet/Selector.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include <fastjet/GhostedAreaSpec.hh>
#include "fastjet/GhostedAreaSpec.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/tools/Pruner.hh"
#include "fastjet/tools/MassDropTagger.hh"
#include "fastjet/tools/JetMedianBackgroundEstimator.hh"
#include "fastjet/tools/GridMedianBackgroundEstimator.hh"
#include "fastjet/tools/Subtractor.hh"
#include "fastjet/Selector.hh"
#include "fastjet/tools/Subtractor.hh"

// Rochester correction for muons //
#include "RoccoR.h"

//for storing vectors in tree//

# include <vector>

#include <ctime>
#include <sys/resource.h>

#include <bitset>

#ifdef __MAKECINT__
    
    #pragma link C++ class std::vector+;
    #pragma link C++ class std::vector<float>+;
    #pragma link C++ class std::vector<std::vector<float> >+;
    
#endif


using namespace std;
using namespace edm;
using namespace reco;  
using namespace CLHEP;
using namespace trigger;
using namespace math;
using namespace fastjet;
using namespace fastjet::contrib;

const float mu_mass = 0.105658;
const float el_mass = 0.000511;
const float pival = acos(-1.);

static const int nsrc = 24;
const char* jecsrcnames[nsrc] = {
	 "AbsoluteStat", "AbsoluteScale","AbsoluteMPFBias", 
	 "FlavorQCD", "Fragmentation", 
	 "PileUpDataMC",  "PileUpPtBB", "PileUpPtEC1", "PileUpPtEC2", //"PileUpPtHF",
	 "PileUpPtRef",
	 "RelativeFSR", "RelativeJEREC1", "RelativeJEREC2", //"RelativeJERHF",
	 "RelativePtBB", "RelativePtEC1", "RelativePtEC2", //"RelativePtHF", 
	 "RelativeBal", "RelativeSample", "RelativeStatEC", "RelativeStatFSR", //"RelativeStatHF", 
	 "SinglePionECAL", "SinglePionHCAL","TimePtEta",
	 "Total"
	};
const int njecmcmx = 2*nsrc + 1 ;

struct triggervar{
  TLorentzVector  trg4v;
  bool		  both;
  bool            level1;
  bool            highl;
  int             ihlt;
  int             prescl;
  int             pdgId;
  int			  type;
  string		  hltname;
};

void logMemoryUsage(const std::string& message) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    edm::LogInfo("MemoryUsage") << message
                                << " | Memory: " << usage.ru_maxrss << " KB";
}

int getbinid(double val, int nbmx, double* array) {
  if (val<array[0]) return -2;
  for (int ix=0; ix<=nbmx; ix++) {
    if (val < array[ix]) return ix-1;
  }
  return -3;
}

double theta_to_eta(double theta) { return -log(tan(theta/2.)); }

double PhiInRange(const double& phi) {
  double phiout = phi;
  if( phiout > 2*M_PI || phiout < -2*M_PI) {
    phiout = fmod( phiout, 2*M_PI);
  }
  if (phiout <= -M_PI) phiout += 2*M_PI;
  else if (phiout >  M_PI) phiout -= 2*M_PI;
  return phiout;
}

double delta2R(double eta1, double phi1, double eta2, double phi2) {
  return sqrt(pow(eta1 - eta2,2) +pow(PhiInRange(phi1 - phi2),2));
}

double diff_func(double f1, double f2){
  double ff = pow(f1-f2,2)*1./pow(f1+f2,2);
  return ff;
}


TLorentzVector productX(TLorentzVector X, TLorentzVector Y, float pro1, float pro2)
{
  float b1, b2, b3;
  float c1, c2, c3;
  
  b1 = X.Px();
  b2 = X.Py();
  b3 = X.Pz();
  
  c1 = Y.Px();
  c2 = Y.Py();
  c3 = Y.Pz();
  
  float d1, d2, e1, e2, X1, X2;
  
  X1 = pro1;
  X2 = pro2;
  
  d1 = (c2*X1 - b2*X2)*1./(b1*c2 - b2*c1);
  d2 = (c1*X1 - b1*X2)*1./(b2*c1 - b1*c2);
  e1 = (b2*c3 - b3*c2)*1./(b1*c2 - b2*c1);
  e2 = (b1*c3 - b3*c1)*1./(b2*c1 - b1*c2);
  
  float A, B, C;
  A = (e1*e1 + e2*e2+ 1);
  B = 2*(d1*e1 + d2*e2);
  C = d1*d1 + d2*d2 - 1;
  
  float sol;
  
  if((pow(B,2) - (4*A*C)) < 0){
    sol = -1*B/(2*A);
    
    float A1, A2, A3;
    A3 = sol;
    A1 = d1 + e1*A3;
    A2 = d2 + e2*A3;
    
    TLorentzVector vec4;
    vec4.SetPxPyPzE(A1,A2,A3,0);
    return vec4;
  }
  else{
    float sol1 = (-1*B+sqrt((pow(B,2) - (4*A*C))))*1./(2*A);
    float sol2 =  (-1*B-sqrt((pow(B,2) - (4*A*C))))*1./(2*A);
    (sol1>sol2)?sol=sol1:sol=sol2;
    
    float A1, A2, A3;
    A3 = sol;
    A1 = d1 + e1*A3;
    A2 = d2 + e2*A3;
    
    TLorentzVector vec4;
    vec4.SetPxPyPzE(A1,A2,A3,0);
    return vec4;;
  }
}

struct JetIDVars
{
  float NHF, NEMF, MUF, CHF, CEMF;
  int NumConst, NumNeutralParticle, CHM;
};

bool getJetID(JetIDVars vars, string jettype="CHS", string year="2018", double eta=0, bool tightLepVeto=true, bool UltraLegacy=false, bool isRun3=false){
  
  //https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID
  
  if (jettype!="CHS" && jettype!="PUPPI"){
    cout<<"Don't know your jet type! I know only CHS & PUPPI :D"<<endl;
    return false;
  }
  
  float NHF, NEMF, MUF, CHF, CEMF;
  int NumConst, NumNeutralParticle, CHM;
  
  NHF = vars.NHF; 
  NEMF = vars.NEMF;
  MUF = vars.MUF;
  CHF = vars.CHF;
  CEMF = vars.CEMF;
  NumConst = vars.NumConst;
  NumNeutralParticle = vars.NumNeutralParticle;
  CHM = vars.CHM;
  
  bool JetID = false;
  
  if(isRun3){
    /*
    if(year=="2022" && jettype=="CHS"){
      
      JetID = ( (fabs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0.01 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)<=2.6 && CHM>0 && CHF>0.01 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.6 && fabs(eta)<=2.7 && CEMF<0.8 && CHM>0 && NEMF<0.99 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)>2.6 && fabs(eta)<=2.7 && CHM>0 && NEMF<0.99 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.7 && fabs(eta)<=3.0 && NEMF<0.99 && NumNeutralParticle>1) || (fabs(eta)>3.0 && NEMF<0.90 && NumNeutralParticle>10));
    }
    
    if(year=="2022" && jettype=="PUPPI"){
      
      JetID = ( (fabs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0.01 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto) || (fabs(eta)<=2.6 && CHM>0 && CHF>0.01 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.6 && abs(eta)<=2.7 && CEMF<0.8 && NEMF<0.99 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)>2.6 && fabs(eta)<=2.7 && NEMF<0.99 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.7 && fabs(eta)<=3.0 && NHF<0.99) || (fabs(eta)>3.0 && NEMF<0.90 && NumNeutralParticle>2));
    }
    */
    if(jettype=="CHS"){
      
      JetID = ( (fabs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0.01 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.99 && tightLepVeto ) || (fabs(eta)<=2.6 && CHM>0 && CHF>0.01 && NumConst>1 && NEMF<0.9 && NHF < 0.99 && !tightLepVeto ) || 
		        (fabs(eta)>2.6 && fabs(eta)<=2.7 && CEMF<0.8 && CHM>0 && NEMF<0.99 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)>2.6 && fabs(eta)<=2.7 && CHM>0 && NEMF<0.99 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.7 && fabs(eta)<=3.0 && NEMF<0.99 && NEMF<0.99 && NumNeutralParticle>1) || (fabs(eta)>3.0 && NEMF<0.40 && NumNeutralParticle>10));
    }
    
    if(jettype=="PUPPI"){
      
      JetID = ( (fabs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0.01 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.99 && tightLepVeto ) || (fabs(eta)<=2.6 && CHM>0 && CHF>0.01 && NumConst>1 && NEMF<0.9 && NHF < 0.99 && !tightLepVeto ) ||
			    (fabs(eta)>2.6 && fabs(eta)<=2.7 && CEMF<0.8 && NEMF<0.99 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)>2.6 && fabs(eta)<=2.7 && NEMF<0.99 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.7 && fabs(eta)<=3.0 && NHF<0.99) || (fabs(eta)>3.0 && NEMF<0.40 && NumNeutralParticle>=2));
    }

  } //Run3
  
  else{
  
  if(!UltraLegacy){
    
    if(year=="2018" && jettype=="CHS"){
      
      JetID = ( (fabs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)<=2.6 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.6 && fabs(eta)<=2.7 && CEMF<0.8 && CHM>0 && NEMF<0.99 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)>2.6 && fabs(eta)<=2.7 && CHM>0 && NEMF<0.99 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.7 && fabs(eta)<=3.0 && NEMF>0.02 && NEMF<0.99 && NumNeutralParticle>2) || (fabs(eta)>3.0 && NEMF<0.90 && NHF>0.2 && NumNeutralParticle>10));
    }
    
    if(year=="2018" && jettype=="PUPPI"){
      
      JetID = ( (fabs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)<=2.6 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.6 && fabs(eta)<=2.7 && CEMF<0.8 && NEMF<0.99 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)>2.6 && fabs(eta)<=2.7 && NEMF<0.99 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)<=3.0 && NHF<0.99) || (fabs(eta)>3.0 && NEMF<0.90 && NHF>0.02 && NumNeutralParticle>2 && NumNeutralParticle<15));
    }
    
    if(year=="2017" && jettype=="CHS"){
      
      JetID = ( (fabs(eta)<=2.4 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)<=2.4 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.4 && fabs(eta)<=2.7 &&  NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)>2.4 && fabs(eta)<=2.7 &&  NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.7 && fabs(eta)<=3.0 && NEMF>0.02 && NEMF<0.99 && NumNeutralParticle>2) || (fabs(eta)>3.0 && NEMF<0.90 && NHF>0.02 && NumNeutralParticle>10));
    }
    
    if(year=="2017" && jettype=="PUPPI"){
      
      JetID = ( (fabs(eta)<=2.4 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)<=2.4 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) ||
 (fabs(eta)>2.4 && fabs(eta)<=2.7 &&  NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)>2.4 && fabs(eta)<=2.7 &&  NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.7 && fabs(eta)<=3.0 && NHF<0.99) || (fabs(eta)>3.0 && NEMF<0.90 && NHF>0.02 && NumNeutralParticle>2 && NumNeutralParticle<15));
    }

    if(year=="2016" && jettype=="CHS"){
      
      JetID = ( (fabs(eta)<=2.4 && CEMF<0.90 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)<=2.4 && CEMF<0.99 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9  && !tightLepVeto ) || (fabs(eta)>2.4 && fabs(eta)<=2.7 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)>2.4 && fabs(eta)<=2.7 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.7 && fabs(eta)<=3.0 && NEMF>0.01 && NHF<0.98 && NumNeutralParticle>2) || (fabs(eta)>3.0 && NEMF<0.90 && NumNeutralParticle>10));
	}
    
    if(year=="2016" && jettype=="PUPPI"){
      
      JetID = ( (fabs(eta)<=2.4 && CEMF<0.9 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)<=2.4 && CEMF<0.99 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || (fabs(eta)>2.4 && fabs(eta)<=2.7 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || (fabs(eta)>2.4 && fabs(eta)<=2.7 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ));
      if(fabs(eta)>2.7) { JetID = false; }
	}
  }
  
  else {
    
    if(year=="2017"||year=="2018"){
      
      if(jettype=="CHS"){
	
	JetID = ( fabs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || ( fabs(eta)<=2.6 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || ( fabs(eta)>2.6 && fabs(eta)<=2.7 && CEMF<0.8 && CHM>0 && NEMF<0.99 && MUF <0.8 && NHF < 0.9 ) || ( fabs(eta)>2.6 && fabs(eta)<=2.7 && CHM>0 && NEMF<0.99 && NHF < 0.9 ) || ( fabs(eta)>2.7 && fabs(eta)<=3.0 && NEMF>0.01 && NEMF<0.99 && NumNeutralParticle>1 ) || ( fabs(eta)>3.0 && NEMF<0.90 && NHF>0.2 && NumNeutralParticle>10) ;
      }
      
      if(jettype=="PUPPI"){
	
	JetID =  ( fabs(eta)<=2.6 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || ( fabs(eta)<=2.6 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || ( fabs(eta)>2.6 && fabs(eta)<=2.7 && CEMF<0.8 && NEMF<0.99 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || ( fabs(eta)>2.6 && fabs(eta)<=2.7 && NEMF<0.99 && NHF < 0.9 && !tightLepVeto ) || ( fabs(eta)>2.7 && fabs(eta)<=3.0 && NHF<0.9999 ) ||( fabs(eta)>3.0 && NEMF<0.90 && NumNeutralParticle>2 ) ;
      }
      // there is a inconsistency between table & lines in https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL
      // table is chosen as it is consistent with the slides https://indico.cern.ch/event/937597/contributions/3940302/attachments/2073315/3481068/ULJetID_UL17_UL18_AK4PUPPI.pdf 
    }
    
    if(year=="2016"){
      
      if(jettype=="CHS"){
	
	JetID =  ( fabs(eta)<=2.4 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || ( fabs(eta)<=2.4 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || ( fabs(eta)>2.4 && fabs(eta)<=2.7 && NEMF<0.99 && NHF < 0.9 ) || ( fabs(eta)>2.7 && fabs(eta)<=3.0 && NEMF>0.0 && NEMF<0.99 && NHF<0.9 && NumNeutralParticle>1 ) || ( fabs(eta)>3.0 && NEMF<0.90 && NHF>0.2 && NumNeutralParticle>10) ;
	
      }
      
      if(jettype=="PUPPI"){
	
	JetID = ( fabs(eta)<=2.4 && CEMF<0.8 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && MUF <0.8 && NHF < 0.9 && tightLepVeto ) || ( fabs(eta)<=2.4 && CHM>0 && CHF>0 && NumConst>1 && NEMF<0.9 && NHF < 0.9 && !tightLepVeto ) || ( fabs(eta)>2.4 && fabs(eta)<=2.7 && NEMF<0.99 && NHF < 0.98 ) || ( fabs(eta)>2.7 && fabs(eta)<=3.0 && NumNeutralParticle>=1 ) || ( fabs(eta)>3.0 && NEMF<0.90 && NumNeutralParticle>2  ) ;
      }
    }	
  }
  
 }//else of Run3
  
  return JetID;
  
}

bool Muon_Tight_ID(bool muonisGL,bool muonisPF, float muonchi, float muonhit, float muonmst, float muontrkvtx, float muondz, float muonpixhit, float muontrklay){
    //https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2#Tight_Muon
	bool tightid = false;
	if(muonisGL && muonisPF){
		if(muonchi<10 && muonhit>0 && muonmst>1){
			if(fabs(muontrkvtx)<0.2 && fabs(muondz)<0.5){
				if(muonpixhit>0 && muontrklay>5){
					tightid = true;
				}
			}
		}
	}
	return tightid;
}

bool StoreMuon(pat::Muon muon1, float ptcut, float etacut){
	
	if (((muon1.isTrackerMuon() || muon1.isGlobalMuon()) && (muon1.isPFMuon())) && (muon1.pt()>=ptcut) && (fabs(muon1.eta())<=etacut)) {                                                                
			return true;
	}
	else{
			return false;
		}
}

bool StoreElectron(pat::Electron electron1, float ptcut, float etacut){
	
	GsfTrackRef gsftrk1 = electron1.gsfTrack();                                                                                                      
    if ((!gsftrk1.isNull()) && (electron1.pt()>=ptcut) && (fabs(electron1.eta())<=etacut) && (gsftrk1->ndof()>=9)) {
			return true;
		}
    else{
			return false;
		}
}

TLorentzVector LeptonJet_subtraction(vector<auto> leps, pat::Jet jet, TLorentzVector jet4v){
	
	TLorentzVector newjet4v;
	newjet4v = jet4v;
	
	if (leps.size()>0) {                                                                                           
		for (unsigned int ilep = 0; ilep<leps.size(); ilep++) {
	  
			bool lepmember = false;
			
			for(unsigned int jd = 0 ; jd < leps[ilep].numberOfSourceCandidatePtrs() ; ++jd) {
				
				if(leps[ilep].sourceCandidatePtr(jd).isNonnull() && leps[ilep].sourceCandidatePtr(jd).isAvailable()){
					const reco::Candidate* jcand = leps[ilep].sourceCandidatePtr(jd).get();
				
					for(unsigned int ic = 0 ; ic < jet.numberOfSourceCandidatePtrs() ; ++ic) {  
					
						if(jet.sourceCandidatePtr(ic).isNonnull() && jet.sourceCandidatePtr(ic).isAvailable()){
							const reco::Candidate* icand = jet.sourceCandidatePtr(ic).get();
							if (delta2R(jcand->eta(),jcand->phi(),icand->eta(),icand->phi()) < 0.00001)    
							{
								TLorentzVector tmpvec(jcand->px(),jcand->py(),jcand->pz(),jcand->energy());
								newjet4v = jet4v - tmpvec;
								lepmember = true; 
								break;
							}
						}
					}		
				
				}
								
			}//jd
			
			if(lepmember) break;
			
		}//ilep
	}
    
    return newjet4v;
}

void Read_JEC(double &total_JEC,  double &tmprecpt, 
			  TLorentzVector jetp4, 
			  double Rho, 
			  bool isData,
			  pat::Jet jet,
			  FactorizedJetCorrector *jecL1Fast, FactorizedJetCorrector *jecL2Relative, FactorizedJetCorrector *jecL3Absolute, FactorizedJetCorrector*jecL2L3Residual)
{
	
	double jeteta = jetp4.Eta();
	double jetphi = jetp4.Phi();
	
    double total_cor =1;
      
    jecL1Fast->setJetPt(tmprecpt); jecL1Fast->setJetA(jet.jetArea()); jecL1Fast->setRho(Rho);jecL1Fast->setJetEta(jeteta); 
    double corFactorL1Fast = jecL1Fast->getCorrection();
    total_cor *= corFactorL1Fast;
    tmprecpt = tmprecpt * corFactorL1Fast;
      
    jecL2Relative->setJetPt(tmprecpt); jecL2Relative->setJetEta(jeteta); jecL2Relative->setJetPhi(jetphi);
    double corFactorL2Relative = jecL2Relative->getCorrection();
    total_cor *= corFactorL2Relative ;
    tmprecpt = tmprecpt * corFactorL2Relative;
      
    jecL3Absolute->setJetPt(tmprecpt); jecL3Absolute->setJetEta(jeteta);
    double corFactorL3Absolute = jecL3Absolute->getCorrection();
    total_cor *= corFactorL3Absolute ;
    tmprecpt = tmprecpt * corFactorL3Absolute;
      
    double corFactorL2L3Residual=1.;
      
    if(isData){
		jecL2L3Residual->setJetPt(tmprecpt); jecL2L3Residual->setJetEta(jeteta);
		corFactorL2L3Residual = jecL2L3Residual->getCorrection();
		total_cor*= corFactorL2L3Residual;
		tmprecpt *=corFactorL2L3Residual;
	}
	
	total_JEC = total_cor;
	
	return;     
}

void Read_JER(std::string mPtResoFile, std::string mPtSFFile, double tmprecpt, TLorentzVector pfjet4v, double Rho, edm::Handle<reco::GenJetCollection>  genjets, double dRcut, vector<double> &SFs)
{
 
	JME::JetResolution resolution;
	resolution = JME::JetResolution(mPtResoFile.c_str());
	JME::JetResolutionScaleFactor res_sf;
	res_sf = JME::JetResolutionScaleFactor(mPtSFFile.c_str());
	
	JME::JetParameters parameters_5 = {{JME::Binning::JetPt, tmprecpt}, {JME::Binning::JetEta, pfjet4v.Eta()}, {JME::Binning::Rho, Rho}};
	double rp = resolution.getResolution(parameters_5);
	double gaus_rp = gRandom->Gaus(0.,rp);
	double sf = res_sf.getScaleFactor(parameters_5, Variation::NOMINAL);
	double sf_up = res_sf.getScaleFactor(parameters_5, Variation::UP);
	double sf_dn = res_sf.getScaleFactor(parameters_5, Variation::DOWN);
	
	bool match = false;
	int match_gen = -1;
		
	for (unsigned get = 0; get<(genjets->size()); get++) {
		TLorentzVector genjet4v((*genjets)[get].px(),(*genjets)[get].py(),(*genjets)[get].pz(), (*genjets)[get].energy());
		if((delta2R(pfjet4v.Rapidity(),pfjet4v.Phi(),genjet4v.Rapidity(),genjet4v.Phi()) < (dRcut)) &&(fabs(tmprecpt-genjet4v.Pt())<(3*fabs(rp)*tmprecpt))){
			match = true;
			match_gen = get;
			break;
		}
	}
		
	if(match && (match_gen>=0)){
	  
		SFs.push_back((sf-1.)*(tmprecpt-(*genjets)[match_gen].pt())*1./tmprecpt);
		SFs.push_back((sf_up-1.)*(tmprecpt-(*genjets)[match_gen].pt())*1./tmprecpt);
		SFs.push_back((sf_dn-1.)*(tmprecpt-(*genjets)[match_gen].pt())*1./tmprecpt);
	  
	}else{
	  
		SFs.push_back(sqrt(max(0.,(sf*sf-1))) * gaus_rp);
		SFs.push_back(sqrt(max(0.,(sf_up*sf_up-1))) * gaus_rp);
		SFs.push_back(sqrt(max(0.,(sf_dn*sf_dn-1))) * gaus_rp);
	}
      	
}

bool Assign_JetVeto(TLorentzVector p4, bool jetID, JetIDVars IDVars,  edm::Handle<edm::View<pat::Muon>>  Muon_collection, TH2D *h_map, float pt_cut=15, float EMF_cut=0.9, float dRmu_cut=0.2)
{
	
	//See: https://cms-talk.web.cern.ch/t/jet-veto-maps-for-run3-data/18444
	
	bool is_veto = false;
	
	TLorentzVector jet_p4;
	jet_p4.SetPtEtaPhiM(p4.Pt(),p4.Eta(), p4.Phi(), p4.M());
		
	if(jet_p4.Pt()>pt_cut && jetID && ((IDVars.NEMF+IDVars.CEMF)<EMF_cut)){
				  
			for(unsigned int iMuon = 0; iMuon < Muon_collection->size(); iMuon++ ) {  
			
				const auto &muon_cand = (*Muon_collection)[iMuon];
				TLorentzVector muon_cand_p4;
				muon_cand_p4.SetPtEtaPhiM(muon_cand.pt(),muon_cand.eta(),muon_cand.phi(),muon_cand.mass());
			
				if(muon_cand.isPFMuon() && jet_p4.DeltaR(muon_cand_p4)>dRmu_cut){
				
					int eta_bin_index = h_map -> GetXaxis() -> FindBin(jet_p4.Eta());
					int phi_bin_index = h_map -> GetYaxis() -> FindBin(jet_p4.Phi());
					if(h_map ->  GetBinContent(eta_bin_index, phi_bin_index) > 0 )  { is_veto = true; }
				
			}  
		}
	}
		
	return is_veto;
}
 

float getEtaForEA(auto obj){
	float eta;
	if(abs(obj->pdgId())==11||abs(obj->pdgId())==22) { eta = obj->superCluster()->eta(); }     
	else { eta = obj->eta(); }
	return eta;    
}

std::unique_ptr<EffectiveAreas> ea_mu_miniiso_, ea_el_miniiso_;

void Read_MiniIsolation(auto obj, double Rho, vector<float> &isovalues)
{
	pat::PFIsolation iso = obj->miniPFIsolation();                                                                                                                                                                                                   
	float chg = iso.chargedHadronIso();                                                                                                                     
	float neu = iso.neutralHadronIso();                                                                                                                     
	float pho = iso.photonIso();                                                                                       
	                                                                                   
	float ea;
	//if(abs(obj->pdgId())==13) { ea = ea_mu_miniiso_->getEffectiveArea(fabs(getEtaForEA(obj))); }
	//else { ea = ea_el_miniiso_->getEffectiveArea(fabs(getEtaForEA(obj))); }  
	if(abs(obj->pdgId())==13) { float abseta = abs(obj->eta());  ea = ea_mu_miniiso_->getEffectiveArea(abseta); }
	else { float abseta = abs(obj->superCluster()->eta()); ea = ea_el_miniiso_->getEffectiveArea(abseta); }  
		                                                                                 
	float R = 10.0/std::min(std::max(obj->pt(), 50.0),200.0);                                                                      
	ea *= std::pow(R / 0.3, 2);                                                                                                                  	
	float tot = (chg+std::max(0.0,neu+pho-(Rho)*ea));
	
	isovalues.push_back(tot);
	isovalues.push_back(chg);
	isovalues.push_back(neu);
	isovalues.push_back(pho);	
	
	for(unsigned ij=0; ij<isovalues.size(); ij++){
		isovalues[ij] *= 1./obj->pt();
	}
}

std::unique_ptr<EffectiveAreas> ea_el_pfiso_;

void Read_ElePFIsolation(auto obj, double Rho, vector<float> &isovalues)
{
	auto iso = obj->pfIsolationVariables();   
	auto  ea = ea_el_pfiso_->getEffectiveArea(fabs(getEtaForEA(obj)));   
	//auto ea = ea_el_pfiso_->getEffectiveArea(abs(obj->superCluster()->eta()));   
    float val = iso.sumChargedHadronPt + max(0., iso.sumNeutralHadronEt + iso.sumPhotonEt - (Rho)*ea); 
    float val04 = (obj->chargedHadronIso()+std::max(0.0,obj->neutralHadronIso()+obj->photonIso()-(Rho)*ea*16./9.));
    isovalues.push_back(val);
    isovalues.push_back(val04);
    
    for(unsigned ij=0; ij<isovalues.size(); ij++){
		isovalues[ij] *= 1./obj->pt();
	}    
}


//class declaration
//
//class Leptop : public edm::EDAnalyzer {
class Leptop : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  explicit Leptop(const edm::ParameterSet&);
  ~Leptop();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;
  
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  void fillmetarray();
  void InitializeBranches();
 
  // ----------member data ---------------------------
  
  int Nevt;
  bool isData;
  bool isMC;
  bool isFastSIM;
  string year;
  bool isRun3;
  bool isUltraLegacy;
  bool isSoftDrop;
  bool add_prefireweights;
  bool store_electron_scalnsmear;
  bool store_fatjet_constituents;
  bool read_btagSF;
  bool subtractLepton_fromAK4, subtractLepton_fromAK8;
  bool store_electrons, store_muons, store_photons, store_ak4jets, store_ak8jets, store_taus, store_CHS_met, store_PUPPI_met;
  bool store_jet_id_variables, store_muon_id_variables, store_additional_muon_id_variables, store_electron_id_variables, store_additional_electron_id_variables, store_photon_id_variables, store_tau_id_variables;
  
  uint nPDFsets;
  
  std::string theRootFileName;
  std::string theHLTTag;
  std::string softdropmass;
  std::string Nsubjettiness_tau1;
  std::string Nsubjettiness_tau2;
  std::string Nsubjettiness_tau3;
  std::string subjets;
  std::string toptagger_DAK8;
  std::string Wtagger_DAK8;
  std::string Ztagger_DAK8;
  std::string Htagger_DAK8;
  std::string bbtagger_DAK8;
  std::string toptagger_PNet;
  std::string Wtagger_PNet;
  std::string Ztagger_PNet;
  std::string Hbbtagger_PNet;
  std::string Hcctagger_PNet;
  std::string H4qtagger_PNet;
  std::string Xbbtagger_PNet;
  std::string Xcctagger_PNet;
  std::string Xqqtagger_PNet;
  std::string Xggtagger_PNet;
  std::string Xtetagger_PNet;
  std::string Xtmtagger_PNet;
  std::string Xtttagger_PNet;
  std::string QCD0HFtagger_PNet;
  std::string QCD1HFtagger_PNet;
  std::string QCD2HFtagger_PNet;
  std::string Xbbtagger_PartT, Xcctagger_PartT, Xcstagger_PartT, Xqqtagger_PartT;
  std::string TopbWqqtagger_PartT, TopbWqtagger_PartT, TopbWevtagger_PartT, TopbWmvtagger_PartT, TopbWtauvtagger_PartT;
  std::string QCDtagger_PartT;
  std::string XWW4qtagger_PartT, XWW3qtagger_PartT, XWWqqevtagger_PartT, XWWqqmvtagger_PartT;
  std::string TvsQCDtagger_PartT, WvsQCDtagger_PartT, ZvsQCDtagger_PartT;
  std::string mass_cor_PNet;
  std::string mass_cor_PartT_genertic, mass_cor_PartT_twoprong;
  
  edm::EDGetTokenT<double> tok_Rho_;
  edm::EDGetTokenT<reco::BeamSpot> tok_beamspot_;
  // vertices //
  edm::EDGetTokenT<reco::VertexCollection> tok_primaryVertices_;
  //edm::EDGetTokenT<edm::ValueMap<float>> pvsScore_;
  edm::EDGetTokenT<reco::VertexCompositePtrCandidateCollection> tok_sv;
  // MET //
  edm::EDGetTokenT<pat::METCollection>tok_mets_, tok_mets_PUPPI_;
  // Jets //
  edm::EDGetTokenT<edm::View<pat::Jet>>tok_pfjetAK8s_;
  edm::EDGetTokenT<edm::View<pat::Jet>>tok_pfjetAK4s_;
  // Muons //
  edm::EDGetTokenT<edm::View<pat::Muon>> tok_muons_;
  // Electrons //
  edm::EDGetTokenT<edm::View<pat::Electron>> tok_electrons_;
  // Photons //
  edm::EDGetTokenT<edm::View<pat::Photon>>tok_photons_;
  // Taus //
  edm::EDGetTokenT<edm::View<pat::Tau>>tok_taus_;
  
  // Photon ID //
  edm::EDGetTokenT <edm::ValueMap <float> > tok_mvaPhoID_FallV2_raw;
  
  // Trigger //
  edm::EDGetTokenT<edm::TriggerResults> triggerBits_;
  edm::EDGetTokenT<edm::TriggerResults> tok_METfilters_;
  edm::EDGetTokenT<pat::TriggerObjectStandAloneCollection> triggerObjects_;
  edm::EDGetTokenT<pat::PackedTriggerPrescales> triggerPrescales_;
  
  // L1 //
  
  //edm::EDGetTokenT<GlobalAlgBlkBxCollection> l1AlgosToken;
  edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd> tok_L1_menu;
  edm::EDGetToken tok_L1_GtHandle;

  // Prefire //

  edm::EDGetTokenT< double > prefweight_token;
  edm::EDGetTokenT< double > prefweightup_token;
  edm::EDGetTokenT< double > prefweightdown_token;
  
  // GEN level objects //
  edm::EDGetTokenT<reco::GenMETCollection>tok_genmets_;
  edm::EDGetTokenT<reco::GenJetCollection>tok_genjetAK8s_;
  edm::EDGetTokenT<reco::GenJetCollection>tok_genjetAK4s_;
  edm::EDGetTokenT<reco::GenJetCollection>tok_genjetAK4swNu_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>>tok_genparticles_;
  edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> jetFlavourInfosToken_;
  // other GEN level info //
  edm::EDGetTokenT<HepMCProduct> tok_HepMC ;
  edm::EDGetTokenT<GenEventInfoProduct> tok_wt_;
  edm::EDGetTokenT<LHEEventProduct> lheEventProductToken_;
  //pileup 
  edm::EDGetTokenT<std::vector<PileupSummaryInfo> > pileup_;
  
  // Effective area (relevant for leptons)
  //std::unique_ptr<EffectiveAreas> ea_miniiso_;
  
  // object cuts //
  
  double min_pt_AK4jet;
  double min_pt_AK8jet;
  double min_pt_mu;
  double min_pt_el;
  double min_pt_gamma;
  double min_pt_tau;
  double min_pt_GENjet;
  double min_pt_AK8GENjet;
  double max_eta; // for all muon, electron, AK4 & AK8 jets
  double max_eta_tau;
  double max_eta_GENjet;
  
  // Soft-drop parameters //
  
  double beta, z_cut;     
  
  // Root file & tree //
  
  TFile* theFile;
  
  TTree* T1;
  TTree* T2;
  
  // maximum numbers of jets, gen particles, etc. //
  
  static const int njetmx = 10; 
  static const int njetmxAK8 = 5;
  static const int npartmx = 50; 
  static const int nconsmax = 1000; 
  static const int njetconsmax = 3; 
  static const int ngenjetAK8mx =10;
  static const int nlhemax = 10;
  
  // variables in ntuple //
  
  unsigned ievt;
    
  int irunold;
  int irun, ilumi, ifltr, ibrnch;
    
  int nprim, npvert, PV_npvsGood, PV_ndof;
  float PV_x, PV_y, PV_z, PV_chi2;// PV_score;
  
  double Rho ;
  
  // Event weights (detector-level) //
  
  double prefiringweight, prefiringweightup, prefiringweightdown;
  
  // MET filter booleans //
  
  bool Flag_goodVertices_;
  bool Flag_globalSuperTightHalo2016Filter_;
  bool Flag_EcalDeadCellTriggerPrimitiveFilter_;
  bool Flag_BadPFMuonFilter_;
  bool Flag_BadPFMuonDzFilter_;
  bool Flag_hfNoisyHitsFilter_;
  bool Flag_eeBadScFilter_;
  bool Flag_ecalBadCalibFilter_;
  
  // MET //
  
  float miset , misphi , sumEt, misetsig;
  float miset_covXX, miset_covXY, miset_covYY;
  float miset_UnclusEup, miset_UnclusEdn;
  float misphi_UnclusEup, misphi_UnclusEdn;
  
  float miset_PUPPI , misphi_PUPPI , sumEt_PUPPI, misetsig_PUPPI;
  float miset_PUPPI_covXX, miset_PUPPI_covXY, miset_PUPPI_covYY;
  float miset_PUPPI_JESup, miset_PUPPI_JESdn, miset_PUPPI_JERup, miset_PUPPI_JERdn, miset_PUPPI_UnclusEup, miset_PUPPI_UnclusEdn;
  float misphi_PUPPI_JESup, misphi_PUPPI_JESdn, misphi_PUPPI_JERup, misphi_PUPPI_JERdn, misphi_PUPPI_UnclusEup, misphi_PUPPI_UnclusEdn;
  
  // AK8 jets //
  
  int nPFJetAK8;
  // Kinematic properties //
  float PFJetAK8_pt[njetmxAK8], PFJetAK8_y[njetmxAK8], PFJetAK8_eta[njetmxAK8], PFJetAK8_phi[njetmxAK8], PFJetAK8_mass[njetmxAK8];
  // Jet ID //
  bool PFJetAK8_jetID_tightlepveto[njetmxAK8], PFJetAK8_jetID[njetmxAK8];
  // Tagging scores //
  //float PFJetAK8_btag_DeepCSV[njetmxAK8]; //useless 
  // DeepAK8
  float PFJetAK8_DeepTag_DAK8_TvsQCD[njetmxAK8], PFJetAK8_DeepTag_DAK8_WvsQCD[njetmxAK8], PFJetAK8_DeepTag_DAK8_ZvsQCD[njetmxAK8], PFJetAK8_DeepTag_DAK8_HvsQCD[njetmxAK8], PFJetAK8_DeepTag_DAK8_bbvsQCD[njetmxAK8];  // DeepAK8 scores (obsolete)
  // ParticleNet
  float PFJetAK8_DeepTag_PNet_TvsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_WvsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_ZvsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_HbbvsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_HccvsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_H4qvsQCD[njetmxAK8]; //mass-correlated  ParticleNet scores
  float PFJetAK8_DeepTag_PNet_XbbvsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_XccvsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_XqqvsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_XggvsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_XtevsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_XtmvsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_XttvsQCD[njetmxAK8], PFJetAK8_DeepTag_PNet_QCD[njetmxAK8], PFJetAK8_DeepTag_PNet_QCD0HF[njetmxAK8], PFJetAK8_DeepTag_PNet_QCD1HF[njetmxAK8], PFJetAK8_DeepTag_PNet_QCD2HF[njetmxAK8]; //mass-decorrelated ParticleNet scores
  // PartT
  float PFJetAK8_DeepTag_PartT_Xbb[njetmxAK8], PFJetAK8_DeepTag_PartT_Xcc[njetmxAK8], PFJetAK8_DeepTag_PartT_Xcs[njetmxAK8], PFJetAK8_DeepTag_PartT_Xqq[njetmxAK8];
  float PFJetAK8_DeepTag_PartT_TopbWqq[njetmxAK8], PFJetAK8_DeepTag_PartT_TopbWq[njetmxAK8], PFJetAK8_DeepTag_PartT_TopbWev[njetmxAK8], PFJetAK8_DeepTag_PartT_TopbWmv[njetmxAK8], PFJetAK8_DeepTag_PartT_TopbWtauv[njetmxAK8];
  float PFJetAK8_DeepTag_PartT_QCD[njetmxAK8];
  float PFJetAK8_DeepTag_PartT_XWW4q[njetmxAK8], PFJetAK8_DeepTag_PartT_XWW3q[njetmxAK8], PFJetAK8_DeepTag_PartT_XWWqqev[njetmxAK8], PFJetAK8_DeepTag_PartT_XWWqqmv[njetmxAK8];
  float PFJetAK8_DeepTag_PartT_TvsQCD[njetmxAK8], PFJetAK8_DeepTag_PartT_WvsQCD[njetmxAK8], PFJetAK8_DeepTag_PartT_ZvsQCD[njetmxAK8];
  // Regressed mass //
  float PFJetAK8_particleNet_massCorr[njetmxAK8], PFJetAK8_partT_massCorr_generic[njetmxAK8], PFJetAK8_partT_massCorr_twoprong[njetmxAK8];

  int PFJetAK8_nBHadrons[njetmxAK8], PFJetAK8_nCHadrons[njetmxAK8];
  // jet ID variables //
  float PFJetAK8_CHF[njetmxAK8], PFJetAK8_NHF[njetmxAK8], PFJetAK8_MUF[njetmxAK8], PFJetAK8_PHF[njetmxAK8], PFJetAK8_CEMF[njetmxAK8], PFJetAK8_NEMF[njetmxAK8], PFJetAK8_EEF[njetmxAK8], PFJetAK8_HFHF[njetmxAK8], /*PFJetAK8_HFEMF[njetmxAK8],*/ PFJetAK8_HOF[njetmxAK8];
  int PFJetAK8_CHM[njetmxAK8], PFJetAK8_NHM[njetmxAK8], PFJetAK8_MUM[njetmxAK8], PFJetAK8_PHM[njetmxAK8], PFJetAK8_Neucons[njetmxAK8], PFJetAK8_Chcons[njetmxAK8], PFJetAK8_EEM[njetmxAK8], PFJetAK8_HFHM[njetmxAK8];// PFJetAK8_HFEMM[njetmxAK8];
  // A few jet substructure variables //
  float PFJetAK8_chrad[njetmxAK8], PFJetAK8_pTD[njetmxAK8]; 
  float PFJetAK8_sdmass[njetmxAK8], PFJetAK8_tau1[njetmxAK8], PFJetAK8_tau2[njetmxAK8], PFJetAK8_tau3[njetmxAK8];
  // Subjet properties //
  float PFJetAK8_sub1pt[njetmxAK8], PFJetAK8_sub1eta[njetmxAK8], PFJetAK8_sub1phi[njetmxAK8], PFJetAK8_sub1mass[njetmxAK8], PFJetAK8_sub1JEC[njetmxAK8], PFJetAK8_sub1btag[njetmxAK8]; 
  float PFJetAK8_sub2pt[njetmxAK8], PFJetAK8_sub2eta[njetmxAK8], PFJetAK8_sub2phi[njetmxAK8], PFJetAK8_sub2mass[njetmxAK8], PFJetAK8_sub2JEC[njetmxAK8], PFJetAK8_sub2btag[njetmxAK8];
  
  // JEC factors & uncs //
  float PFJetAK8_JEC[njetmxAK8];
  float PFJetAK8_jesup_AbsoluteStat[njetmxAK8], PFJetAK8_jesdn_AbsoluteStat[njetmxAK8];
  float PFJetAK8_jesup_AbsoluteScale[njetmxAK8], PFJetAK8_jesdn_AbsoluteScale[njetmxAK8];
  float PFJetAK8_jesup_AbsoluteMPFBias[njetmxAK8], PFJetAK8_jesdn_AbsoluteMPFBias[njetmxAK8];
  float PFJetAK8_jesup_FlavorQCD[njetmxAK8], PFJetAK8_jesdn_FlavorQCD[njetmxAK8];
  float PFJetAK8_jesup_Fragmentation[njetmxAK8], PFJetAK8_jesdn_Fragmentation[njetmxAK8];
  float PFJetAK8_jesup_PileUpDataMC[njetmxAK8], PFJetAK8_jesdn_PileUpDataMC[njetmxAK8];
  float PFJetAK8_jesup_PileUpPtBB[njetmxAK8], PFJetAK8_jesdn_PileUpPtBB[njetmxAK8];
  float PFJetAK8_jesup_PileUpPtEC1[njetmxAK8], PFJetAK8_jesdn_PileUpPtEC1[njetmxAK8];
  float PFJetAK8_jesup_PileUpPtEC2[njetmxAK8], PFJetAK8_jesdn_PileUpPtEC2[njetmxAK8];
  float PFJetAK8_jesup_PileUpPtRef[njetmxAK8], PFJetAK8_jesdn_PileUpPtRef[njetmxAK8];
  float PFJetAK8_jesup_RelativeFSR[njetmxAK8], PFJetAK8_jesdn_RelativeFSR[njetmxAK8];
  float PFJetAK8_jesup_RelativeJEREC1[njetmxAK8], PFJetAK8_jesdn_RelativeJEREC1[njetmxAK8];
  float PFJetAK8_jesup_RelativeJEREC2[njetmxAK8], PFJetAK8_jesdn_RelativeJEREC2[njetmxAK8];
  float PFJetAK8_jesup_RelativePtBB[njetmxAK8], PFJetAK8_jesdn_RelativePtBB[njetmxAK8];
  float PFJetAK8_jesup_RelativePtEC1[njetmxAK8], PFJetAK8_jesdn_RelativePtEC1[njetmxAK8];
  float PFJetAK8_jesup_RelativePtEC2[njetmxAK8], PFJetAK8_jesdn_RelativePtEC2[njetmxAK8];
  float PFJetAK8_jesup_RelativeBal[njetmxAK8], PFJetAK8_jesdn_RelativeBal[njetmxAK8];
  float PFJetAK8_jesup_RelativeSample[njetmxAK8], PFJetAK8_jesdn_RelativeSample[njetmxAK8];
  float PFJetAK8_jesup_RelativeStatEC[njetmxAK8], PFJetAK8_jesdn_RelativeStatEC[njetmxAK8];
  float PFJetAK8_jesup_RelativeStatFSR[njetmxAK8], PFJetAK8_jesdn_RelativeStatFSR[njetmxAK8];
  float PFJetAK8_jesup_SinglePionECAL[njetmxAK8], PFJetAK8_jesdn_SinglePionECAL[njetmxAK8];
  float PFJetAK8_jesup_SinglePionHCAL[njetmxAK8], PFJetAK8_jesdn_SinglePionHCAL[njetmxAK8];
  float PFJetAK8_jesup_TimePtEta[njetmxAK8], PFJetAK8_jesdn_TimePtEta[njetmxAK8];
  float PFJetAK8_jesup_Total[njetmxAK8], PFJetAK8_jesdn_Total[njetmxAK8];
  // JER factors & uncs //
  float PFJetAK8_reso[njetmxAK8], PFJetAK8_resoup[njetmxAK8], PFJetAK8_resodn[njetmxAK8];
  float PFJetAK8_jesup_pu[njetmx], PFJetAK8_jesup_rel[njetmx], PFJetAK8_jesup_scale[njetmx], PFJetAK8_jesup_total[njetmx], PFJetAK8_jesdn_pu[njetmx], PFJetAK8_jesdn_rel[njetmx], PFJetAK8_jesdn_scale[njetmx], PFJetAK8_jesdn_total[njetmx];
  // Veto Flags //
  bool PFJetAK8_jetveto_Flag[njetmxAK8], PFJetAK8_jetveto_eep_Flag[njetmxAK8];
  // Jet consituents //
  int nPFJetAK8_cons;
  float PFJetAK8_cons_pt[nconsmax], PFJetAK8_cons_eta[nconsmax], PFJetAK8_cons_phi[nconsmax], PFJetAK8_cons_mass[nconsmax];
  int PFJetAK8_cons_jetIndex[nconsmax], PFJetAK8_cons_pdgId[nconsmax];
  
  // AK4 jets //
  
  int nPFJetAK4;
  // Kinematic properties //
  float PFJetAK4_pt[njetmx], PFJetAK4_eta[njetmx], PFJetAK4_y[njetmx], PFJetAK4_phi[njetmx], PFJetAK4_mass[njetmx], PFJetAK4_area[njetmx];
  // Jet ID //
  bool PFJetAK4_jetID[njetmx], PFJetAK4_jetID_tightlepveto[njetmx];
  // Jet Veto ID //
  bool PFJetAK4_jetveto_Flag[njetmx], PFJetAK4_jetveto_eep_Flag[njetmx];
  // B tag scores //
  float PFJetAK4_btag_DeepCSV[njetmx], PFJetAK4_btag_DeepFlav[njetmx]; 
  float PFJetAK4_btagDeepFlavB[njetmx], PFJetAK4_btagDeepFlavCvB[njetmx], PFJetAK4_btagDeepFlavCvL[njetmx], PFJetAK4_btagDeepFlavQG[njetmx];
  float PFJetAK4_btagPNetB[njetmx], PFJetAK4_btagPNetCvNotB[njetmx], PFJetAK4_btagPNetCvB[njetmx], PFJetAK4_btagPNetCvL[njetmx], PFJetAK4_btagPNetQG[njetmx];
  float PFJetAK4_btagRobustParTAK4B[njetmx], PFJetAK4_btagRobustParTAK4CvB[njetmx], PFJetAK4_btagRobustParTAK4CvL[njetmx], PFJetAK4_btagRobustParTAK4QG[njetmx];
  // Energy regression for b jets //
  float PFJetAK4_PNetRegPtRawCorr[njetmx], PFJetAK4_PNetRegPtRawCorrNeutrino[njetmx], PFJetAK4_PNetRegPtRawRes[njetmx];
  // DeepFlav SFs (usually not stored here) //
  float PFJetAK4_btag_DeepFlav_SF[njetmx], PFJetAK4_btag_DeepFlav_SF_up[njetmx], PFJetAK4_btag_DeepFlav_SF_dn[njetmx];
  // JER factor and uncs //
  float PFJetAK4_reso[njetmx], PFJetAK4_resoup[njetmx], PFJetAK4_resodn[njetmx];
  // JEC factor and uncs //
  float PFJetAK4_JEC[njetmx];
  float PFJetAK4_jesup_AbsoluteStat[njetmx], PFJetAK4_jesdn_AbsoluteStat[njetmx];
  float PFJetAK4_jesup_AbsoluteScale[njetmx], PFJetAK4_jesdn_AbsoluteScale[njetmx];
  float PFJetAK4_jesup_AbsoluteMPFBias[njetmx], PFJetAK4_jesdn_AbsoluteMPFBias[njetmx];
  float PFJetAK4_jesup_FlavorQCD[njetmx], PFJetAK4_jesdn_FlavorQCD[njetmx];
  float PFJetAK4_jesup_Fragmentation[njetmx], PFJetAK4_jesdn_Fragmentation[njetmx];
  float PFJetAK4_jesup_PileUpDataMC[njetmx], PFJetAK4_jesdn_PileUpDataMC[njetmx];
  float PFJetAK4_jesup_PileUpPtBB[njetmx], PFJetAK4_jesdn_PileUpPtBB[njetmx];
  float PFJetAK4_jesup_PileUpPtEC1[njetmx], PFJetAK4_jesdn_PileUpPtEC1[njetmx];
  float PFJetAK4_jesup_PileUpPtEC2[njetmx], PFJetAK4_jesdn_PileUpPtEC2[njetmx];
  float PFJetAK4_jesup_PileUpPtRef[njetmx], PFJetAK4_jesdn_PileUpPtRef[njetmx];
  float PFJetAK4_jesup_RelativeFSR[njetmx], PFJetAK4_jesdn_RelativeFSR[njetmx];
  float PFJetAK4_jesup_RelativeJEREC1[njetmx], PFJetAK4_jesdn_RelativeJEREC1[njetmx];
  float PFJetAK4_jesup_RelativeJEREC2[njetmx], PFJetAK4_jesdn_RelativeJEREC2[njetmx];
  float PFJetAK4_jesup_RelativePtBB[njetmx], PFJetAK4_jesdn_RelativePtBB[njetmx];
  float PFJetAK4_jesup_RelativePtEC1[njetmx], PFJetAK4_jesdn_RelativePtEC1[njetmx];
  float PFJetAK4_jesup_RelativePtEC2[njetmx], PFJetAK4_jesdn_RelativePtEC2[njetmx];
  float PFJetAK4_jesup_RelativeBal[njetmx], PFJetAK4_jesdn_RelativeBal[njetmx];
  float PFJetAK4_jesup_RelativeSample[njetmx], PFJetAK4_jesdn_RelativeSample[njetmx];
  float PFJetAK4_jesup_RelativeStatEC[njetmx], PFJetAK4_jesdn_RelativeStatEC[njetmx];
  float PFJetAK4_jesup_RelativeStatFSR[njetmx], PFJetAK4_jesdn_RelativeStatFSR[njetmx];
  float PFJetAK4_jesup_SinglePionECAL[njetmx], PFJetAK4_jesdn_SinglePionECAL[njetmx];
  float PFJetAK4_jesup_SinglePionHCAL[njetmx], PFJetAK4_jesdn_SinglePionHCAL[njetmx];
  float PFJetAK4_jesup_TimePtEta[njetmx], PFJetAK4_jesdn_TimePtEta[njetmx];
  float PFJetAK4_jesup_Total[njetmx], PFJetAK4_jesdn_Total[njetmx];
  // GEN level flavor //
  int PFJetAK4_hadronflav[njetmx], PFJetAK4_partonflav[njetmx];
  int PFJetAK4_Ncons[njetmx];
  // QG Likelihood (probably not required for Run3)
  float PFJetAK4_qgl[njetmx];
  // Pileup ID
  float PFJetAK4_PUID[njetmx];
  // jet charge //
  float PFJetAK4_charge_kappa_0p3[njetmx], PFJetAK4_charge_kappa_0p6[njetmx], PFJetAK4_charge_kappa_1p0[njetmx];
  float PFJetAK4_charged_ptsum[njetmx];
  
  // muon variables //
  
  int nMuon;
  
  float Muon_charge[njetmx], Muon_p[njetmx], Muon_pt[njetmx], Muon_eta[njetmx], Muon_phi[njetmx], Muon_tunePBestTrack_pt[njetmx];
  // ID //
  bool Muon_isPF[njetmx], Muon_isGL[njetmx], Muon_isTRK[njetmx], Muon_isStandAloneMuon[njetmx];
  bool Muon_isGoodGL[njetmx], Muon_isTight[njetmx], Muon_isHighPt[njetmx], Muon_isHighPttrk[njetmx], Muon_isMed[njetmx], Muon_isMedPr[njetmx], Muon_isLoose[njetmx], Muon_TightID[njetmx], Muon_mediumPromptId[njetmx];
  int Muon_MVAID[njetmx];
  int Muon_mvaMuID_WP[njetmx];
  float Muon_mvaMuID[njetmx];
  unsigned int Muon_multiIsoId[njetmx], Muon_puppiIsoId[njetmx], Muon_tkIsoId[njetmx];
  // iso //
  float Muon_minisoall[njetmx]; 
  //float Muon_minchiso[njetmx], Muon_minnhiso[njetmx], Muon_minphiso[njetmx];
  float Muon_miniPFRelIso_all[njetmx], Muon_miniPFRelIso_Chg[njetmx];
  unsigned int Muon_PF_iso[njetmx], Muon_Mini_iso[njetmx];
  // displacement //
  float Muon_dxy[njetmx], Muon_dxybs[njetmx], Muon_dxyErr[njetmx], Muon_dz[njetmx], Muon_dzErr[njetmx], Muon_ip3d[njetmx], Muon_sip3d[njetmx];
  // other variables //
  float Muon_ptErr[njetmx], Muon_chi[njetmx], Muon_ecal[njetmx], Muon_hcal[njetmx]; //Muon_emiso[njetmx], Muon_hadiso[njetmx], Muon_tkpt03[njetmx], Muon_tkpt05[njetmx];
  float Muon_posmatch[njetmx], Muon_trkink[njetmx], Muon_segcom[njetmx], Muon_pfiso[njetmx], Muon_pfiso03[njetmx], Muon_hit[njetmx], Muon_pixhit[njetmx], Muon_mst[njetmx], Muon_trklay[njetmx], Muon_valfrac[njetmx],Muon_dxy_sv[njetmx];
  int Muon_ndf[njetmx];
  // corrected momentum (Rochester correction) //
  float Muon_corrected_pt[njetmx], Muon_correctedUp_pt[njetmx], Muon_correctedDown_pt[njetmx];
  
  // electron variables //
  
  int nElectron;
  
  float Electron_charge[njetmx], Electron_pt[njetmx], Electron_eta[njetmx], Electron_phi[njetmx], Electron_e[njetmx], Electron_e_ECAL[njetmx], Electron_p[njetmx];
  // super-cluster //
  float Electron_supcl_eta[njetmx], Electron_supcl_phi[njetmx], Electron_supcl_e[njetmx], Electron_supcl_rawE[njetmx]; 
  // ID //
  bool Electron_mvaid_Fallv2WP90[njetmx], Electron_mvaid_Fallv2WP90_noIso[njetmx], Electron_mvaid_Fallv2WP80[njetmx], Electron_mvaid_Fallv2WP80_noIso[njetmx], Electron_mvaid_Fallv2WPLoose[njetmx], Electron_mvaid_Fallv2WPLoose_noIso[njetmx];
  bool Electron_mvaid_Winter22v1WP90[njetmx], Electron_mvaid_Winter22v1WP90_noIso[njetmx], Electron_mvaid_Winter22v1WP80[njetmx], Electron_mvaid_Winter22v1WP80_noIso[njetmx];// Electron_mvaid_Winter22v1WPLoose[njetmx], Electron_mvaid_Winter22v1WPLoose_noIso[njetmx];
  float Electron_mvaid_Fallv2_value[njetmx], Electron_mvaid_Fallv2noIso_value[njetmx];
  float Electron_mvaid_Winter22IsoV1_value[njetmx], Electron_mvaid_Winter22NoIsoV1_value[njetmx];
  int Electron_cutbased_id[njetmx];
  // iso //
  float Electron_pfiso_drcor[njetmx];
  float Electron_pfiso_eacor[njetmx];
  float Electron_pfiso04_eacor[njetmx];
  float Electron_pfRelIso03_all[njetmx], Electron_pfRelIso04_all[njetmx]; //Electron_emiso03[njetmx], Electron_hadiso03[njetmx], Electron_emiso04[njetmx], Electron_hadiso04[njetmx];
  float Electron_pfisolsumphet[njetmx], Electron_pfisolsumchhadpt[njetmx], Electron_pfsiolsumneuhadet[njetmx];
  //float Electron_minchiso[njetmx], Electron_minnhiso[njetmx], Electron_minphiso[njetmx];
  float Electron_minisoall[njetmx]; 
  float Electron_miniPFRelIso_all[njetmx], Electron_miniPFRelIso_chg[njetmx];
  // displacement //
  float Electron_dxy[njetmx],  Electron_dxyErr[njetmx], Electron_dxy_sv[njetmx], Electron_dz[njetmx], Electron_dzErr[njetmx], Electron_ip3d[njetmx], Electron_sip3d[njetmx];
  // other properties //
  float Electron_hovere[njetmx], Electron_qovrper[njetmx], Electron_chi[njetmx]; 
  float Electron_eoverp[njetmx], Electron_ietaieta[njetmx], Electron_etain[njetmx], Electron_phiin[njetmx], Electron_fbrem[njetmx]; 
  float Electron_nohits[njetmx], Electron_misshits[njetmx];
  int Electron_ndf[njetmx];
  float Electron_eccalTrkEnergyPostCorr[njetmx];
  float Electron_energyScaleValue[njetmx];
  float Electron_energyScaleUp[njetmx];
  float Electron_energyScaleDown[njetmx];
  float Electron_energySigmaValue[njetmx];
  float Electron_energySigmaUp[njetmx];
  float Electron_energySigmaDown[njetmx];
  float Electron_sigmaieta[njetmx], Electron_sigmaiphi[njetmx];
  float Electron_r9full[njetmx];
  float Electron_supcl_etaw[njetmx];
  float Electron_supcl_phiw[njetmx];
  float Electron_hcaloverecal[njetmx];
  float Electron_cloctftrkn[njetmx];
  float Electron_cloctftrkchi2[njetmx];
  float Electron_e1x5bye5x5[njetmx];
  float Electron_normchi2[njetmx];
  float Electron_hitsmiss[njetmx];
  float Electron_trkmeasure[njetmx];
  float Electron_convtxprob[njetmx];
  float Electron_ecloverpout[njetmx];
  float Electron_ecaletrkmomentum[njetmx];
  float Electron_deltaetacltrkcalo[njetmx];
  float Electron_supcl_preshvsrawe[njetmx];
  bool Electron_convVeto[njetmx]; 
  int8_t Electron_seediEtaOriX[njetmx], Electron_seediPhiOriY[njetmx];
  
  // photon variables //
  
  int nPhoton;
  // ID //
  float Photon_e[njetmx];
  float Photon_eta[njetmx];
  float Photon_phi[njetmx];
  bool Photon_mvaid_RunIIIWinter22V1_WP90[njetmx];
  bool Photon_mvaid_RunIIIWinter22V1_WP80[njetmx];
  bool Photon_mvaid_Fall17V2_WP90[njetmx];
  bool Photon_mvaid_Fall17V2_WP80[njetmx];
  float Photon_mvaid_Fall17V2_raw[njetmx];
  bool Photon_mvaid_Spring16V1_WP90[njetmx];
  bool Photon_mvaid_Spring16V1_WP80[njetmx];
  // other variables //
  float Photon_e1by9[njetmx];
  float Photon_e9by25[njetmx];
  float Photon_hadbyem[njetmx];
  float Photon_trkiso[njetmx];
  float Photon_emiso[njetmx];
  float Photon_hadiso[njetmx];
  float Photon_chhadiso[njetmx];
  float Photon_neuhadiso[njetmx];
  float Photon_PUiso[njetmx];
  float Photon_phoiso[njetmx];
  float Photon_ietaieta[njetmx];
  
  // tau variables //
  
  int nTau;
  float Tau_pt[njetmx];
  float Tau_eta[njetmx];
  float Tau_phi[njetmx];
  float Tau_e[njetmx];
  bool Tau_isPF[njetmx];
  float Tau_dxy[njetmx];
  float Tau_dz[njetmx];
  int Tau_charge[njetmx];
  // ID (DeepTau 2017v2p1)
  float Tau_jetiso_deeptau2017v2p1_raw[njetmx];
  int Tau_jetiso_deeptau2017v2p1[njetmx];
  float Tau_eiso_deeptau2017v2p1_raw[njetmx];
  int Tau_eiso_deeptau2017v2p1[njetmx];
  float Tau_muiso_deeptau2017v2p1_raw[njetmx];
  int Tau_muiso_deeptau2017v2p1[njetmx];
  // ID (DeepTau 2018v2p5)
  float Tau_jetiso_deeptau2018v2p5_raw[njetmx];
  int Tau_jetiso_deeptau2018v2p5[njetmx];
  float Tau_eiso_deeptau2018v2p5_raw[njetmx];
  int Tau_eiso_deeptau2018v2p5[njetmx];
  float Tau_muiso_deeptau2018v2p5_raw[njetmx];
  int Tau_muiso_deeptau2018v2p5[njetmx];
  // iso & other variables //
  float Tau_rawiso[njetmx];
  float Tau_rawisodR03[njetmx];
  int Tau_muiso[njetmx];
  float Tau_eiso_raw[njetmx];
  int Tau_eiso[njetmx];
  float Tau_eiso2018_raw[njetmx];
  int Tau_eiso2018[njetmx];
  float Tau_puCorr[njetmx];
  float Tau_leadtrkpt[njetmx];
  float Tau_leadtrketa[njetmx];
  float Tau_leadtrkphi[njetmx];
  float Tau_leadtrkdxy[njetmx];
  float Tau_leadtrkdz[njetmx];
  int Tau_decayMode[njetmx];
  bool Tau_decayModeinding[njetmx];
  bool Tau_decayModeindingNewDMs[njetmx];

  // Trigger Object Info //
  
  int nTrigObj;
  float TrigObj_pt[njetmx], TrigObj_eta[njetmx],TrigObj_phi[njetmx], TrigObj_mass[njetmx];
  // HLT, L1, or Both? Booleans //
  bool TrigObj_HLT[njetmx], TrigObj_L1[njetmx],  TrigObj_Both[njetmx];
  // HLT index, object ID, and type
  int  TrigObj_Ihlt[njetmx], TrigObj_pdgId[njetmx], TrigObj_type[njetmx];
  vector<string> TrigObj_HLTname;
  
  // HL triggers //
  
  // Check prescale info online in: https://cmshltinfo.app.cern.ch/summary
  
  static const int nHLTmx = 58;
  const char *hlt_name[nHLTmx] = {
	// single-muon triggers
		"HLT_IsoMu24","HLT_IsoTkMu24", "HLT_IsoMu27",                //3
	// single-muon triggers (non-isolated)
		"HLT_Mu50", "HLT_TkMu50", "HLT_TkMu100","HLT_OldMu100",		//4
		"HLT_HighPtTkMu100","HLT_CascadeMu100", //Run3				//2
  	// single-electron triggers  (isolated)
		"HLT_Ele27_WPTight_Gsf", "HLT_Ele30_WPTight_Gsf",  "HLT_Ele32_WPTight_Gsf", "HLT_Ele35_WPTight_Gsf", //4
		"HLT_Ele28_eta2p1_WPTight_Gsf_HT150", "HLT_Ele32_WPTight_Gsf_L1DoubleEG",								//2
  	// single-electron triggers  (non-isolated)
		"HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165","HLT_Ele115_CaloIdVT_GsfTrkIdT", 							//2
	// double-muon triggers
		"HLT_Mu37_TkMu27",											//1
		"HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL","HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",					//2
		"HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL","HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ","HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8","HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8", //4
	//	double-electron triggers //
		"HLT_DoubleEle25_CaloIdL_MW", "HLT_DoubleEle33_CaloIdL_MW","HLT_DoubleEle33_CaloIdL_GsfTrkIdVL", //3
		"HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL","HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",    //2
	//	emu triggers
		"HLT_Mu37_Ele27_CaloIdL_MW", "HLT_Mu27_Ele37_CaloIdL_MW",    //2
		"HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL","HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",   //2
		"HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL","HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", //2
	// jet triggers
		"HLT_PFHT800", "HLT_PFHT900", "HLT_PFHT1050", //3
		"HLT_PFJet450","HLT_PFJet500",                //2
		"HLT_AK8PFJet450",  "HLT_AK8PFJet500", 		  //2
		"HLT_AK8PFJet400_TrimMass30", "HLT_AK8PFHT800_TrimMass50", //2
		"HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35", 
		"HLT_AK8PFJet220_SoftDropMass40_PNetBB0p35_DoubleAK4PFJet60_30_PNet2BTagMean0p50", //, // not in 2022  
		"HLT_AK8PFJet425_SoftDropMass40", //2022+2023
		"HLT_AK8PFJet420_MassSD30",     //2023  //4
	//4b triggers in Run 3
		"HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",//2022
		"HLT_QuadPFJet70_50_40_35_PNet2BTagMean0p65", //2023 (only in 6.2 fb-1)
		"HLT_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70", //2023 
		"HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55", // 2023 parking (21 fb-1)  //4
	// photon trigger
		"HLT_Photon175","HLT_Photon200",  //2
	// MET trigger
		"HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60","HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight", "HLT_PFMETTypeOne140_PFMHT140_IDTight" //4
  }; 
  
  // HLT trigger booleans //

  bool hlt_IsoMu24;
  bool hlt_IsoTkMu24;
  bool hlt_IsoMu27;
 
  bool hlt_Mu50; 
  bool hlt_TkMu50;
  bool hlt_TkMu100;
  bool hlt_OldMu100;
  bool hlt_HighPtTkMu100;
  bool hlt_CascadeMu100;
  
  bool hlt_Ele27_WPTight_Gsf;
  bool hlt_Ele30_WPTight_Gsf;
  bool hlt_Ele32_WPTight_Gsf;
  bool hlt_Ele35_WPTight_Gsf;
  bool hlt_Ele28_eta2p1_WPTight_Gsf_HT150;
  bool hlt_Ele32_WPTight_Gsf_L1DoubleEG;
  
  bool hlt_Ele50_CaloIdVT_GsfTrkIdT_PFJet165; 
  bool hlt_Ele115_CaloIdVT_GsfTrkIdT;
  
  bool hlt_Mu37_TkMu27;
  bool hlt_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL;
  bool hlt_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ;
  bool hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL;
  bool hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ;
  bool hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8;
  bool hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8;
  
  bool hlt_DoubleEle25_CaloIdL_MW;
  bool hlt_DoubleEle33_CaloIdL_MW;
  bool hlt_DoubleEle33_CaloIdL_GsfTrkIdVL;
  bool hlt_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL;
  bool hlt_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;
  
  bool hlt_Mu37_Ele27_CaloIdL_MW;
  bool hlt_Mu27_Ele37_CaloIdL_MW;
  bool hlt_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL;
  bool hlt_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ;
  bool hlt_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL;
  bool hlt_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ;
  
  bool hlt_PFHT800;
  bool hlt_PFHT900;
  bool hlt_PFHT1050;
  bool hlt_PFJet450;
  bool hlt_PFJet500;
  bool hlt_AK8PFJet450;
  bool hlt_AK8PFJet500;
  bool hlt_AK8PFJet400_TrimMass30;
  bool hlt_AK8PFHT800_TrimMass50;
  //AK8 jet triggers in Run 3
  bool hlt_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35;
  bool hlt_AK8PFJet220_SoftDropMass40_PNetBB0p35_DoubleAK4PFJet60_30_PNet2BTagMean0p50;
  bool hlt_AK8PFJet425_SoftDropMass40;
  bool hlt_AK8PFJet420_MassSD30;
 
  //4b2j triggers in Run 3//
  bool hlt_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65;
  bool hlt_QuadPFJet70_50_40_35_PNet2BTagMean0p65;
  bool hlt_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70;
  bool hlt_PFHT280_QuadPFJet30_PNet2BTagMean0p55;
  
  bool hlt_Photon175;
  bool hlt_Photon200;
  
  bool hlt_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60;
  bool hlt_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60;
  bool hlt_PFMETNoMu140_PFMHTNoMu140_IDTight;
  bool hlt_PFMETTypeOne140_PFMHT140_IDTight;
  
  //int trig_value;
  vector<bool> trig_bits;
  vector<string> trig_paths;
  
  // L1 trigger index //
  int idx_L1_HTT280er, idx_L1_QuadJet60er2p5, idx_L1_HTT320er, idx_L1_HTT360er, idx_L1_HTT400er, idx_L1_HTT450er;
  int idx_L1_HTT280er_QuadJet_70_55_40_35_er2p5, idx_L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3;
  int idx_L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3, idx_L1_Mu6_HTT240er, idx_L1_SingleJet60;

 // L1 trigger booleans //
  bool L1_QuadJet60er2p5;
  bool L1_HTT280er;
  bool L1_HTT320er;
  bool L1_HTT360er;
  bool L1_HTT400er;
  bool L1_HTT450er;
  bool L1_HTT280er_QuadJet_70_55_40_35_er2p5;
  bool L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3;
  bool L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3;
  bool L1_Mu6_HTT240er;
  bool L1_SingleJet60;
  
  // Prescales //
  //HLTConfigProvider hltConfig_;
  

  ///// GEN level variables ////

  // Collision Info //
  float Generator_qscale, Generator_x1, Generator_x2, Generator_xpdf1, Generator_xpdf2, Generator_scalePDF;
  int Generator_id1, Generator_id2;
  
  // Pileup vetices //
  int npu_vert;
  int npu_vert_true;
  
  // Weights //
  double Generator_weight, LHE_weight;
  double weights[njetmx];
  
  // LHE weights (many) //
  static const int nlheweightmax = 300;
  int nLHEWeights;
  //float LHEWeights[nlheweightmax];
  vector<float>LHEWeights;
  // LHE scale weights //
  static const int nlhescalemax = 9;
  int nLHEScaleWeights;
  float LHEScaleWeights[nlhescalemax];
  // LHE PDF weights //
  static const int nlhepdfmax = 103; // be consistent with nPDFsets (nlhepdfmax should be >= nPDFsets)
  int nLHEPDFWeights;
  float LHEPDFWeights[nlhepdfmax];
  // LHE parton-shower weights //
  static const int nlhepsmax = 8;
  int nLHEPSWeights;
  float LHEPSWeights[nlhepsmax];
  
  // Gen MET //
  float genmiset, genmisphi, genmisetsig;
  
  // Gen AK8 jets //
  int nGenJetAK8;
  float GenJetAK8_pt[njetmxAK8], GenJetAK8_eta[njetmxAK8], GenJetAK8_phi[njetmxAK8], GenJetAK8_mass[njetmxAK8], GenJetAK8_sdmass[njetmxAK8]; 
  int GenJetAK8_hadronflav[njetmxAK8], GenJetAK8_partonflav[njetmxAK8];
  // Gen AK8 jet constituents //
  int nGenJetAK8_cons;
  float GenJetAK8_cons_pt[nconsmax], GenJetAK8_cons_eta[nconsmax], GenJetAK8_cons_phi[nconsmax], GenJetAK8_cons_mass[nconsmax];
  int GenJetAK8_cons_jetIndex[nconsmax], GenJetAK8_cons_pdgId[nconsmax];
  
  // Gen AK4 jets //
  int nGenJetAK4;
  float GenJetAK4_pt[njetmx], GenJetAK4_eta[njetmx], GenJetAK4_phi[njetmx], GenJetAK4_mass[njetmx];
  int GenJetAK4_hadronflav[njetmx], GenJetAK4_partonflav[njetmx];
  
  // Gen AK4 jets (including neutrinos) //
  int nGenJetAK4wNu;
  float GenJetAK4wNu_pt[njetmx], GenJetAK4wNu_eta[njetmx], GenJetAK4wNu_phi[njetmx], GenJetAK4wNu_mass[njetmx];
  int GenJetAK4wNu_hadronflav[njetmx], GenJetAK4wNu_partonflav[njetmx];
  
  // Generator particles //
  int nGenPart;
  int GenPart_status[npartmx], GenPart_pdg[npartmx], GenPart_mompdg[npartmx], GenPart_momstatus[npartmx], GenPart_grmompdg[npartmx], GenPart_momid[npartmx], GenPart_daugno[npartmx];
  float GenPart_pt[npartmx], GenPart_eta[npartmx], GenPart_phi[npartmx], GenPart_mass[npartmx]; //GenPart_q[npartmx];
  bool GenPart_fromhard[npartmx], GenPart_fromhardbFSR[npartmx], GenPart_isPromptFinalState[npartmx], GenPart_isLastCopyBeforeFSR[npartmx], GenPart_isDirectPromptTauDecayProductFinalState[npartmx];
  
  // LHE-level particles //
  int nLHEPart;
  float LHEPart_pt[nlhemax], LHEPart_eta[nlhemax], LHEPart_phi[nlhemax], LHEPart_m[nlhemax];
  int LHEPart_pdg[nlhemax];
  
  
  //HLTPrescaleProvider hltPrescaleProvider_;
    
  // ---- Jet Corrector Parameter ---- //
  
  JetCorrectorParameters *L1FastAK4, *L2RelativeAK4, *L3AbsoluteAK4, *L2L3ResidualAK4;
  vector<JetCorrectorParameters> vecL1FastAK4, vecL2RelativeAK4, vecL3AbsoluteAK4, vecL2L3ResidualAK4;
  FactorizedJetCorrector *jecL1FastAK4, *jecL2RelativeAK4, *jecL3AbsoluteAK4, *jecL2L3ResidualAK4;
  
  JetCorrectorParameters *L1FastAK8, *L2RelativeAK8, *L3AbsoluteAK8, *L2L3ResidualAK8;
  vector<JetCorrectorParameters> vecL1FastAK8, vecL2RelativeAK8, vecL3AbsoluteAK8, vecL2L3ResidualAK8;
  FactorizedJetCorrector *jecL1FastAK8, *jecL2RelativeAK8, *jecL3AbsoluteAK8, *jecL2L3ResidualAK8;
  
  // ---- Jet Corrector Parameter End---- //

  // BTagCalibration Begin //

  BTagCalibration calib_deepflav;
  BTagCalibrationReader reader_deepflav;
  
  // BTagCalibration End //

  // ---- Jet Resolution Parameter ---- //
  
  std::string mJECL1FastFileAK4, mJECL2RelativeFileAK4, mJECL3AbsoluteFileAK4, mJECL2L3ResidualFileAK4, mJECL1FastFileAK8, mJECL2RelativeFileAK8, mJECL3AbsoluteFileAK8, mJECL2L3ResidualFileAK8;
  std::string mPtResoFileAK4, mPtResoFileAK8, mPtSFFileAK4, mPtSFFileAK8;
  
  std::string mJECUncFileAK4;
  std::vector<JetCorrectionUncertainty*> vsrc ;
  
  std::string mJECUncFileAK8;
  std::vector<JetCorrectionUncertainty*> vsrcAK8 ;
  
  std::string mJetVetoMap;
  
  // ---- Jet Resolution Parameter End---- //
  
   // jet veto map files & histograms //
  
  TFile *file_jetvetomap;
  TH2D *h_jetvetomap;
  TH2D *h_jetvetomap_eep;
  
  // ---- B tagging scale factor files --- //
  
  std::string mBtagSF_DeepCSV;
  std::string mBtagSF_DeepFlav;
  
  // ---- B tagging scale factor files End --- //
  
  // ---- Rochester correction files --- //
  
  std::string mRochcorFolder;
  
  // Rochester correction class for muons//
  RoccoR roch_cor; 
  
  // Electron MVA ID //
  
  std::string melectronID_isowp90, melectronID_noisowp90;
  std::string melectronID_isowp80, melectronID_noisowp80;
  std::string melectronID_isowploose, melectronID_noisowploose;
  std::string melectronID_isowp90_Fall17, melectronID_noisowp90_Fall17;
  std::string melectronID_isowp80_Fall17, melectronID_noisowp80_Fall17;
  std::string melectronID_isowploose_Fall17, melectronID_noisowploose_Fall17;
  std::string melectronID_cutbased_loose, melectronID_cutbased_medium, melectronID_cutbased_tight;
  
  // Photon MVA ID //
  
  std::string mPhoID_RunIIIWinter22V1_WP90, mPhoID_RunIIIWinter22V1_WP80;
  std::string mPhoID_FallV2_WP90, mPhoID_FallV2_WP80;
  std::string mPhoID_SpringV1_WP90, mPhoID_SpringV1_WP80;
 
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

Leptop::Leptop(const edm::ParameterSet& pset):
  //hltPrescaleProvider_(pset, consumesCollector(), *this)
  tok_L1_menu(esConsumes<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd>())
{
  //now do what ever initialization is needed
  
  edm::Service<TFileService> fs;
  
  // Booleans //
  
  isData    = pset.getUntrackedParameter<bool>("Data",false);
  isMC      = pset.getUntrackedParameter<bool>("MonteCarlo", false);
  isFastSIM      = pset.getUntrackedParameter<bool>("FastSIM", false);
  year		=  pset.getUntrackedParameter<string>("YEAR","2018");
  isRun3 	= pset.getUntrackedParameter<bool>("isRun3", false);
  isUltraLegacy = pset.getUntrackedParameter<bool>("UltraLegacy", false);
  isSoftDrop      = pset.getUntrackedParameter<bool>("SoftDrop_ON",false);
  theRootFileName = pset.getUntrackedParameter<string>("RootFileName");
  theHLTTag = pset.getUntrackedParameter<string>("HLTTag", "HLT");
  add_prefireweights = pset.getUntrackedParameter<bool>("add_prefireweights", false);
  store_electron_scalnsmear = pset.getUntrackedParameter<bool>("store_electron_scalnsmear", false);
  store_fatjet_constituents = pset.getUntrackedParameter<bool>("store_fatjet_constituents", false);
  read_btagSF = pset.getUntrackedParameter<bool>("Read_btagging_SF", false);
  subtractLepton_fromAK8 = pset.getUntrackedParameter<bool>("Subtract_Lepton_fromAK8", false);
  subtractLepton_fromAK4 = pset.getUntrackedParameter<bool>("Subtract_Lepton_fromAK4", false);
  
  store_electrons = pset.getUntrackedParameter<bool>("store_electrons", false);
  store_muons = pset.getUntrackedParameter<bool>("store_muons", false);
  store_photons = pset.getUntrackedParameter<bool>("store_photons", false);
  store_ak4jets = pset.getUntrackedParameter<bool>("store_ak4jets", false);
  store_ak8jets = pset.getUntrackedParameter<bool>("store_ak8jets", false);
  store_taus = pset.getUntrackedParameter<bool>("store_taus", false);
  store_CHS_met   = pset.getUntrackedParameter<bool>("store_CHS_met", false);
  store_PUPPI_met = pset.getUntrackedParameter<bool>("store_PUPPI_met", false);

  store_jet_id_variables  = pset.getUntrackedParameter<bool>("store_jet_id_variables", false);
  store_muon_id_variables  = pset.getUntrackedParameter<bool>("store_muon_id_variables", false);
  store_additional_muon_id_variables  = pset.getUntrackedParameter<bool>("store_additional_muon_id_variables", false);
  store_electron_id_variables  = pset.getUntrackedParameter<bool>("store_electron_id_variables", false);
  store_additional_electron_id_variables  = pset.getUntrackedParameter<bool>("store_additional_electron_id_variables", false);
  store_photon_id_variables  = pset.getUntrackedParameter<bool>("store_photon_id_variables", false);
  store_tau_id_variables  = pset.getUntrackedParameter<bool>("store_tau_id_variables", false);
  
  // object thresholds //
  
  min_pt_AK4jet = pset.getUntrackedParameter<double>("minJetPt",25.);
  min_pt_GENjet = pset.getUntrackedParameter<double>("minGenJetPt",15.);
  min_pt_AK8jet = pset.getUntrackedParameter<double>("minAK8JetPt",180.);
  min_pt_AK8GENjet = pset.getUntrackedParameter<double>("minGenAK8JetPt");//,150.);
   
  min_pt_mu = pset.getUntrackedParameter<double>("minMuonPt",10.);
  min_pt_el = pset.getUntrackedParameter<double>("minElectronPt",10.);
  min_pt_gamma = pset.getUntrackedParameter<double>("minPhotonPt",10.);
  min_pt_tau = pset.getUntrackedParameter<double>("minTauPt",10.);
  
  max_eta = pset.getUntrackedParameter<double>("maxEta",3.);
  max_eta_GENjet = pset.getUntrackedParameter<double>("maxGenJetEta",3.);
  max_eta_tau = pset.getUntrackedParameter<double>("maxTauEta",2.3);
 
  // parameters for jet substructure variables //
 
  softdropmass = pset.getUntrackedParameter<string>("softdropmass");
  beta = pset.getUntrackedParameter<double>("beta",0);
  z_cut = pset.getUntrackedParameter<double>("z_cut",0.1); 
  Nsubjettiness_tau1 = pset.getUntrackedParameter<string>("tau1");
  Nsubjettiness_tau2 = pset.getUntrackedParameter<string>("tau2");
  Nsubjettiness_tau3 = pset.getUntrackedParameter<string>("tau3");
  
  // AK8 subjet collection name //
  
  subjets = pset.getUntrackedParameter<string>("subjets");
  
  // AK8 tagger scores //
  
  toptagger_DAK8 = pset.getUntrackedParameter<string>("toptagger_DAK8");
  Wtagger_DAK8 = pset.getUntrackedParameter<string>("Wtagger_DAK8");
  Ztagger_DAK8 = pset.getUntrackedParameter<string>("Ztagger_DAK8");
  Htagger_DAK8 = pset.getUntrackedParameter<string>("Htagger_DAK8");
  bbtagger_DAK8 = pset.getUntrackedParameter<string>("bbtagger_DAK8");  
  toptagger_PNet = pset.getUntrackedParameter<string>("toptagger_PNet");
  Wtagger_PNet = pset.getUntrackedParameter<string>("Wtagger_PNet");
  Ztagger_PNet = pset.getUntrackedParameter<string>("Ztagger_PNet");
  Hbbtagger_PNet = pset.getUntrackedParameter<string>("Hbbtagger_PNet");
  Hcctagger_PNet = pset.getUntrackedParameter<string>("Hcctagger_PNet");
  H4qtagger_PNet = pset.getUntrackedParameter<string>("H4qtagger_PNet");
  Xbbtagger_PNet = pset.getUntrackedParameter<string>("Xbbtagger_PNet");
  Xcctagger_PNet = pset.getUntrackedParameter<string>("Xcctagger_PNet");  
  Xqqtagger_PNet = pset.getUntrackedParameter<string>("Xqqtagger_PNet");  
  Xggtagger_PNet = pset.getUntrackedParameter<string>("Xggtagger_PNet"); 
  Xtetagger_PNet = pset.getUntrackedParameter<string>("Xtetagger_PNet"); 
  Xtmtagger_PNet = pset.getUntrackedParameter<string>("Xtmtagger_PNet"); 
  Xtttagger_PNet = pset.getUntrackedParameter<string>("Xtttagger_PNet"); 
  QCD0HFtagger_PNet = pset.getUntrackedParameter<string>("QCD0HFtagger_PNet"); 
  QCD1HFtagger_PNet = pset.getUntrackedParameter<string>("QCD1HFtagger_PNet"); 
  QCD2HFtagger_PNet = pset.getUntrackedParameter<string>("QCD2HFtagger_PNet"); 
  Xbbtagger_PartT = pset.getUntrackedParameter<string>("Xbbtagger_PartT"); 
  Xcctagger_PartT = pset.getUntrackedParameter<string>("Xcctagger_PartT"); 
  Xcstagger_PartT = pset.getUntrackedParameter<string>("Xcstagger_PartT"); 
  Xqqtagger_PartT = pset.getUntrackedParameter<string>("Xqqtagger_PartT"); 
  TopbWqqtagger_PartT = pset.getUntrackedParameter<string>("TopbWqqtagger_PartT"); 
  TopbWqtagger_PartT = pset.getUntrackedParameter<string>("TopbWqtagger_PartT"); 
  TopbWevtagger_PartT = pset.getUntrackedParameter<string>("TopbWevtagger_PartT"); 
  TopbWmvtagger_PartT = pset.getUntrackedParameter<string>("TopbWmvtagger_PartT"); 
  TopbWtauvtagger_PartT = pset.getUntrackedParameter<string>("TopbWtauvtagger_PartT"); 
  QCDtagger_PartT = pset.getUntrackedParameter<string>("QCDtagger_PartT"); 
  XWW4qtagger_PartT = pset.getUntrackedParameter<string>("XWW4qtagger_PartT"); 
  XWW3qtagger_PartT = pset.getUntrackedParameter<string>("XWW3qtagger_PartT"); 
  XWWqqevtagger_PartT = pset.getUntrackedParameter<string>("XWWqqevtagger_PartT"); 
  XWWqqmvtagger_PartT = pset.getUntrackedParameter<string>("XWWqqmvtagger_PartT"); 
  TvsQCDtagger_PartT = pset.getUntrackedParameter<string>("TvsQCDtagger_PartT"); 
  WvsQCDtagger_PartT = pset.getUntrackedParameter<string>("WvsQCDtagger_PartT"); 
  ZvsQCDtagger_PartT = pset.getUntrackedParameter<string>("ZvsQCDtagger_PartT"); 
  mass_cor_PNet = pset.getUntrackedParameter<string>("mass_cor_PNet"); 
  mass_cor_PartT_genertic = pset.getUntrackedParameter<string>("mass_cor_PartT_genertic"); 
  mass_cor_PartT_twoprong = pset.getUntrackedParameter<string>("mass_cor_PartT_twoprong"); 
  
  // Effective areas //
  
  ea_mu_miniiso_.reset(new EffectiveAreas((pset.getParameter<edm::FileInPath>("EAFile_MuonMiniIso")).fullPath()));
  ea_el_miniiso_.reset(new EffectiveAreas((pset.getParameter<edm::FileInPath>("EAFile_EleMiniIso")).fullPath()));
  ea_el_pfiso_.reset(new EffectiveAreas((pset.getParameter<edm::FileInPath>("EAFile_ElePFIso")).fullPath()));
  
  // Beam spot & vertices //
  
  tok_beamspot_ = consumes<reco::BeamSpot> (pset.getParameter<edm::InputTag>("Beamspot"));
  tok_primaryVertices_ =consumes<reco::VertexCollection>( pset.getParameter<edm::InputTag>("PrimaryVertices"));
  //pvsScore_ =consumes<edm::ValueMap<float>>( pset.getParameter<edm::InputTag>("PrimaryVeticesSource"));
  
  tok_sv =consumes<reco::VertexCompositePtrCandidateCollection>( pset.getParameter<edm::InputTag>("SecondaryVertices"));
  
  tok_Rho_ = consumes<double>(pset.getParameter<edm::InputTag>("PFRho"));
     
  // objects (reco) //
  
  tok_mets_= consumes<pat::METCollection> ( pset.getParameter<edm::InputTag>("PFMet"));
  tok_mets_PUPPI_ = consumes<pat::METCollection> ( pset.getParameter<edm::InputTag>("PuppiMet"));
    
  tok_muons_ = consumes<edm::View<pat::Muon>> ( pset.getParameter<edm::InputTag>("Muons"));
  tok_electrons_ = consumes<edm::View<pat::Electron>> ( pset.getParameter<edm::InputTag>("Electrons"));
  tok_photons_ = consumes<edm::View<pat::Photon>>  ( pset.getParameter<edm::InputTag>("Photons"));
  tok_taus_ = consumes<edm::View<pat::Tau>>  ( pset.getParameter<edm::InputTag>("Taus"));
  
  tok_pfjetAK8s_= consumes<edm::View<pat::Jet>>( pset.getParameter<edm::InputTag>("PFJetsAK8"));
  tok_pfjetAK4s_= consumes<edm::View<pat::Jet>>( pset.getParameter<edm::InputTag>("PFJetsAK4"));
 
  // objects GEN //
  
  if(isMC){
	  
	tok_genmets_= consumes<reco::GenMETCollection> ( pset.getParameter<edm::InputTag>("GENMet"));  
	  
    tok_genjetAK8s_= consumes<reco::GenJetCollection>( pset.getParameter<edm::InputTag>("GENJetAK8"));
    tok_genjetAK4s_= consumes<reco::GenJetCollection>( pset.getParameter<edm::InputTag>("GENJetAK4"));
    tok_genjetAK4swNu_= consumes<reco::GenJetCollection>( pset.getParameter<edm::InputTag>("GENJetAK4wNu"));
    tok_genparticles_ = consumes<std::vector<reco::GenParticle>>( pset.getParameter<edm::InputTag>("GenParticles"));
    jetFlavourInfosToken_ = consumes<reco::JetFlavourInfoMatchingCollection>(pset.getParameter<edm::InputTag>("jetFlavourInfos"));
    
    tok_HepMC = consumes<HepMCProduct>(pset.getParameter<edm::InputTag>("Generator"));
    tok_wt_ = consumes<GenEventInfoProduct>(pset.getParameter<edm::InputTag>("Generator")) ;
    lheEventProductToken_ = consumes<LHEEventProduct>(pset.getParameter<edm::InputTag>("LHEEventProductInputTag")) ;
    pileup_ = consumes<std::vector<PileupSummaryInfo> >(pset.getParameter<edm::InputTag>("slimmedAddPileupInfo"));
    nPDFsets      = pset.getUntrackedParameter<uint>("nPDFsets", 103);
    
  }
  
  // Electron ID //
  
  melectronID_isowp90       = pset.getParameter<std::string>("electronID_isowp90");
  melectronID_noisowp90     = pset.getParameter<std::string>("electronID_noisowp90");
  melectronID_isowp80       = pset.getParameter<std::string>("electronID_isowp80");
  melectronID_noisowp80     = pset.getParameter<std::string>("electronID_noisowp80");
  melectronID_isowploose       = pset.getParameter<std::string>("electronID_isowploose");
  melectronID_noisowploose     = pset.getParameter<std::string>("electronID_noisowploose");
  melectronID_isowp90_Fall17       = pset.getParameter<std::string>("electronID_isowp90_Fall17");
  melectronID_noisowp90_Fall17     = pset.getParameter<std::string>("electronID_noisowp90_Fall17");
  melectronID_isowp80_Fall17       = pset.getParameter<std::string>("electronID_isowp80_Fall17");
  melectronID_noisowp80_Fall17     = pset.getParameter<std::string>("electronID_noisowp80_Fall17");
  melectronID_isowploose_Fall17       = pset.getParameter<std::string>("electronID_isowploose_Fall17");
  melectronID_noisowploose_Fall17     = pset.getParameter<std::string>("electronID_noisowploose_Fall17");
  
  melectronID_cutbased_loose     = pset.getParameter<std::string>("electronID_cutbased_loose");
  melectronID_cutbased_medium    = pset.getParameter<std::string>("electronID_cutbased_medium");
  melectronID_cutbased_tight     = pset.getParameter<std::string>("electronID_cutbased_tight");
  
  // Photon ID //
  
  mPhoID_RunIIIWinter22V1_WP90       = pset.getParameter<std::string>("PhoID_RunIIIWinter22V1_WP90");
  mPhoID_RunIIIWinter22V1_WP80       = pset.getParameter<std::string>("PhoID_RunIIIWinter22V1_WP80");
  mPhoID_FallV2_WP90       = pset.getParameter<std::string>("PhoID_FallV2_WP90");
  mPhoID_FallV2_WP80       = pset.getParameter<std::string>("PhoID_FallV2_WP80");
  mPhoID_SpringV1_WP90       = pset.getParameter<std::string>("PhoID_SpringV1_WP90");
  mPhoID_SpringV1_WP80       = pset.getParameter<std::string>("PhoID_SpringV1_WP80");
  
  tok_mvaPhoID_FallV2_raw = consumes<edm::ValueMap <float> >(pset.getParameter<edm::InputTag>("label_mvaPhoID_FallV2_Value"));
  
  // JEC Files //
  
  mJECL1FastFileAK4         = pset.getParameter<std::string>("jecL1FastFileAK4");
  mJECL1FastFileAK8         = pset.getParameter<std::string>("jecL1FastFileAK8");
  mJECL2RelativeFileAK4     = pset.getParameter<std::string>("jecL2RelativeFileAK4");
  mJECL2RelativeFileAK8     = pset.getParameter<std::string>("jecL2RelativeFileAK8");
  mJECL3AbsoluteFileAK4     = pset.getParameter<std::string>("jecL3AbsoluteFileAK4");
  mJECL3AbsoluteFileAK8     = pset.getParameter<std::string> ("jecL3AbsoluteFileAK8");
  mJECL2L3ResidualFileAK4   = pset.getParameter<std::string> ("jecL2L3ResidualFileAK4");
  mJECL2L3ResidualFileAK8   = pset.getParameter<std::string> ("jecL2L3ResidualFileAK8");
  
  mJECUncFileAK4 = pset.getParameter<std::string>("JECUncFileAK4");
  mJECUncFileAK8 = pset.getParameter<std::string>("JECUncFileAK8");       
  
  // JER Files //
  
  mPtResoFileAK4  = pset.getParameter<std::string>("PtResoFileAK4");
  mPtResoFileAK8  = pset.getParameter<std::string>("PtResoFileAK8");
  mPtSFFileAK4  = pset.getParameter<std::string>("PtSFFileAK4");
  mPtSFFileAK8  = pset.getParameter<std::string>("PtSFFileAK8");
  
  // Jet Veto Map Files //
  
  mJetVetoMap = pset.getParameter<std::string>("JetVetoMap");               
  
  // Rochestor correction folder //
  
  mRochcorFolder = pset.getParameter<std::string>("RochcorFolder");
  
  // B tag SF name //
  
  mBtagSF_DeepFlav = pset.getParameter<std::string>("BtagSFFile_DeepFlav");
  
  // Triggers //
  
  if(!isFastSIM){
	triggerBits_ = consumes<edm::TriggerResults> ( pset.getParameter<edm::InputTag>("bits"));
	tok_METfilters_ = consumes<edm::TriggerResults> ( pset.getParameter<edm::InputTag>("MET_Filters"));
	triggerObjects_ = consumes<pat::TriggerObjectStandAloneCollection>(pset.getParameter<edm::InputTag>("TriggerObjects"));
	triggerPrescales_ = consumes<pat::PackedTriggerPrescales>(pset.getParameter<edm::InputTag>("prescales"));
	tok_L1_GtHandle = consumes<BXVector<GlobalAlgBlk>>( pset.getParameter<edm::InputTag>("L1_GtHandle"));            
    //l1GtMenuToken_           (esConsumes<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd>());
  
  }
  
  // Prefire //
  
  if(add_prefireweights){
	prefweight_token = consumes< double >(edm::InputTag("prefiringweight:nonPrefiringProb"));
	prefweightup_token = consumes< double >(edm::InputTag("prefiringweight:nonPrefiringProbUp"));
	prefweightdown_token = consumes< double >(edm::InputTag("prefiringweight:nonPrefiringProbDown"));
  }
  
  // NTuple //
  
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();
 
  T1 = new TTree("Events", "XtoYH");
 
  T1->Branch("irun", &irun, "irun/I");  
  T1->Branch("ilumi", &ilumi, "ilumi/I");  
  T1->Branch("ievt", &ievt, "ievt/i");
  
  // primary vertices //
  
  T1->Branch("nprim", &nprim, "nprim/I");
  T1->Branch("npvert", &npvert, "npvert/I");
  T1->Branch("PV_npvsGood", &PV_npvsGood, "PV_npvsGood/I");
  T1->Branch("PV_ndof", &PV_ndof, "PV_ndof/I");
  T1->Branch("PV_chi2", &PV_chi2, "PV_chi2/F");
  T1->Branch("PV_x", &PV_x, "PV_x/F");
  T1->Branch("PV_y", &PV_y, "PV_y/F");
  T1->Branch("PV_z", &PV_z, "PV_z/F");
  //T1->Branch("PV_score", &PV_score, "PV_score/F");
  
  // energy density //
  
  T1->Branch("Rho", &Rho, "Rho/D") ;
  
  // MET filters //
  
  T1->Branch("Flag_goodVertices",&Flag_goodVertices_,"Flag_goodVertices_/O");
  T1->Branch("Flag_globalSuperTightHalo2016Filter",&Flag_globalSuperTightHalo2016Filter_,"Flag_globalSuperTightHalo2016Filter_/O");
  T1->Branch("Flag_EcalDeadCellTriggerPrimitiveFilter",&Flag_EcalDeadCellTriggerPrimitiveFilter_,"Flag_EcalDeadCellTriggerPrimitiveFilter_/O");
  T1->Branch("Flag_BadPFMuonFilter",&Flag_BadPFMuonFilter_,"Flag_BadPFMuonFilter_/O");
  T1->Branch("Flag_BadPFMuonDzFilter",&Flag_BadPFMuonDzFilter_,"Flag_BadPFMuonDzFilter_/O");
  T1->Branch("Flag_hfNoisyHitsFilter",&Flag_hfNoisyHitsFilter_,"Flag_hfNoisyHitsFilter_/O");
  T1->Branch("Flag_eeBadScFilter",&Flag_eeBadScFilter_,"Flag_eeBadScFilter_/O");
  T1->Branch("Flag_ecalBadCalibFilter",&Flag_ecalBadCalibFilter_,"Flag_ecalBadCalibFilter_/O");
  
  // trigger info //
  
  //T1->Branch("trig_value",&trig_value,"trig_value/I");  
  T1->Branch("trig_bits","std::vector<bool>",&trig_bits);
  T1->Branch("trig_paths","std::vector<string>",&trig_paths);
  //single-muon triggers (isolated)//
  T1->Branch("hlt_IsoMu24",&hlt_IsoMu24,"hlt_IsoMu24/O");
  T1->Branch("hlt_IsoTkMu24",&hlt_IsoTkMu24,"hlt_IsoTkMu24/O");
  T1->Branch("hlt_IsoMu27", &hlt_IsoMu27, "hlt_IsoMu27/O");
  //single-muon triggers (non-isolated)//
  T1->Branch("hlt_Mu50",&hlt_Mu50,"hlt_Mu50/O");
  T1->Branch("hlt_TkMu50", &hlt_TkMu50, "hlt_TkMu50/O");
  T1->Branch("hlt_TkMu100", &hlt_TkMu100, "hlt_TkMu100/O");
  T1->Branch("hlt_OldMu100", &hlt_OldMu100, "hlt_OldMu100/O");
  T1->Branch("hlt_HighPtTkMu100",&hlt_HighPtTkMu100,"hlt_HighPtTkMu100/O");
  T1->Branch("hlt_CascadeMu100",&hlt_CascadeMu100,"hlt_CascadeMu100/O");
  //single-electron triggers//
  T1->Branch("hlt_Ele27_WPTight_Gsf",&hlt_Ele27_WPTight_Gsf,"hlt_Ele27_WPTight_Gsf/O");
  T1->Branch("hlt_Ele30_WPTight_Gsf",&hlt_Ele30_WPTight_Gsf,"hlt_Ele30_WPTight_Gsf/O");
  T1->Branch("hlt_Ele32_WPTight_Gsf",&hlt_Ele32_WPTight_Gsf,"hlt_Ele32_WPTight_Gsf/O");
  T1->Branch("hlt_Ele35_WPTight_Gsf", &hlt_Ele35_WPTight_Gsf, "hlt_Ele35_WPTight_Gsf/O");
  T1->Branch("hlt_Ele28_eta2p1_WPTight_Gsf_HT150",&hlt_Ele28_eta2p1_WPTight_Gsf_HT150,"hlt_Ele28_eta2p1_WPTight_Gsf_HT150/O");
  T1->Branch("hlt_Ele32_WPTight_Gsf_L1DoubleEG", &hlt_Ele32_WPTight_Gsf_L1DoubleEG, "hlt_Ele32_WPTight_Gsf_L1DoubleEG/O");
  T1->Branch("hlt_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",&hlt_Ele50_CaloIdVT_GsfTrkIdT_PFJet165,"hlt_Ele50_CaloIdVT_GsfTrkIdT_PFJet165/O");
  T1->Branch("hlt_Ele115_CaloIdVT_GsfTrkIdT",&hlt_Ele115_CaloIdVT_GsfTrkIdT,"hlt_Ele115_CaloIdVT_GsfTrkIdT/O");
  //double muon triggers//
  T1->Branch("hlt_Mu37_TkMu27",&hlt_Mu37_TkMu27,"hlt_Mu37_TkMu27/O");
  T1->Branch("hlt_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL", &hlt_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL, "hlt_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL/O");
  T1->Branch("hlt_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ", &hlt_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ, "hlt_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ/O");
  T1->Branch("hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL", &hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL, "hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL/O");
  T1->Branch("hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ", &hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ, "hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ/O");
  T1->Branch("hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8", &hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8, "hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8/O");
  T1->Branch("hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8", &hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8, "hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8/O");
  //double electron triggers//
  T1->Branch("hlt_DoubleEle25_CaloIdL_MW",&hlt_DoubleEle25_CaloIdL_MW,"hlt_DoubleEle25_CaloIdL_MW/O");
  T1->Branch("hlt_DoubleEle33_CaloIdL_MW", &hlt_DoubleEle33_CaloIdL_MW, "hlt_DoubleEle33_CaloIdL_MW/O");
  T1->Branch("hlt_DoubleEle33_CaloIdL_GsfTrkIdVL", &hlt_DoubleEle33_CaloIdL_GsfTrkIdVL, "hlt_DoubleEle33_CaloIdL_GsfTrkIdVL/O");
  T1->Branch("hlt_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL", &hlt_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL, "hlt_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL/O");
  T1->Branch("hlt_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", &hlt_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ, "hlt_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/O");
  //muon-electron cross triggers//
  T1->Branch("hlt_Mu37_Ele27_CaloIdL_MW",&hlt_Mu37_Ele27_CaloIdL_MW,"hlt_Mu37_Ele27_CaloIdL_MW/O");
  T1->Branch("hlt_Mu27_Ele37_CaloIdL_MW",&hlt_Mu27_Ele37_CaloIdL_MW,"hlt_Mu27_Ele37_CaloIdL_MW/O");
  T1->Branch("hlt_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL", &hlt_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL, "hlt_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL/O");
  T1->Branch("hlt_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", &hlt_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ, "hlt_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/O");
  T1->Branch("hlt_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", &hlt_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL, "hlt_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL/O");
  T1->Branch("hlt_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", &hlt_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ, "hlt_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/O");
  // jet triggers //
  T1->Branch("hlt_PFHT800",&hlt_PFHT800,"hlt_PFHT800/O");
  T1->Branch("hlt_PFHT900",&hlt_PFHT900,"hlt_PFHT900/O");
  T1->Branch("hlt_PFHT1050",&hlt_PFHT1050,"hlt_PFHT1050/O");
  T1->Branch("hlt_PFJet450",&hlt_PFJet450,"hlt_PFJet450/O");
  T1->Branch("hlt_PFJet500",&hlt_PFJet500,"hlt_PFJet500/O");
  T1->Branch("hlt_AK8PFJet450",&hlt_AK8PFJet450,"hlt_AK8PFJet450/O");
  T1->Branch("hlt_AK8PFJet500",&hlt_AK8PFJet500,"hlt_AK8PFJet500/O");
  T1->Branch("hlt_AK8PFJet400_TrimMass30",&hlt_AK8PFJet400_TrimMass30,"hlt_AK8PFJet400_TrimMass30/O");
  T1->Branch("hlt_AK8PFHT800_TrimMass50",&hlt_AK8PFHT800_TrimMass50,"hlt_AK8PFHT800_TrimMass50/O");
  // jet triggers in Run3
  T1->Branch("hlt_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",&hlt_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35,"hlt_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35/O");
  T1->Branch("hlt_AK8PFJet220_SoftDropMass40_PNetBB0p35_DoubleAK4PFJet60_30_PNet2BTagMean0p50",&hlt_AK8PFJet220_SoftDropMass40_PNetBB0p35_DoubleAK4PFJet60_30_PNet2BTagMean0p50,"hlt_AK8PFJet220_SoftDropMass40_PNetBB0p35_DoubleAK4PFJet60_30_PNet2BTagMean0p50/O");
  T1->Branch("hlt_AK8PFJet425_SoftDropMass40",&hlt_AK8PFJet425_SoftDropMass40,"hlt_AK8PFJet425_SoftDropMass40/O");
  T1->Branch("hlt_AK8PFJet420_MassSD30",&hlt_AK8PFJet420_MassSD30,"hlt_AK8PFJet420_MassSD30/O");
  // 4b jet triggers //
  T1->Branch("hlt_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",&hlt_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65,"hlt_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65/O");
  T1->Branch("hlt_QuadPFJet70_50_40_35_PNet2BTagMean0p65",&hlt_QuadPFJet70_50_40_35_PNet2BTagMean0p65,"hlt_QuadPFJet70_50_40_35_PNet2BTagMean0p65/O");
  T1->Branch("hlt_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",&hlt_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70,"hlt_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70/O");
  T1->Branch("hlt_PFHT280_QuadPFJet30_PNet2BTagMean0p55",&hlt_PFHT280_QuadPFJet30_PNet2BTagMean0p55,"hlt_PFHT280_QuadPFJet30_PNet2BTagMean0p55/O");
  // Photon triggers //
  T1->Branch("hlt_Photon175",&hlt_Photon175,"hlt_Photon175/O");
  T1->Branch("hlt_Photon200",&hlt_Photon200,"hlt_Photon200/O");
  // MET triggers //
  T1->Branch("hlt_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60",&hlt_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60,"hlt_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60/O");
  T1->Branch("hlt_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60",&hlt_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60,"hlt_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60/O");
  T1->Branch("hlt_PFMETNoMu140_PFMHTNoMu140_IDTight",&hlt_PFMETNoMu140_PFMHTNoMu140_IDTight,"hlt_PFMETNoMu140_PFMHTNoMu140_IDTight/O");
  T1->Branch("hlt_PFMETTypeOne140_PFMHT140_IDTight",&hlt_PFMETTypeOne140_PFMHT140_IDTight,"hlt_PFMETTypeOne140_PFMHT140_IDTight/O");
 
  // Trigger object info //
  
  T1->Branch("nTrigObj",&nTrigObj,"nTrigObj/I");
  T1->Branch("TrigObj_pt",TrigObj_pt,"TrigObj_pt[nTrigObj]/F");
  T1->Branch("TrigObj_eta",TrigObj_eta,"TrigObj_eta[nTrigObj]/F");
  T1->Branch("TrigObj_phi",TrigObj_phi,"TrigObj_phi[nTrigObj]/F");
  T1->Branch("TrigObj_mass",TrigObj_mass,"TrigObj_mass[nTrigObj]/F");
  T1->Branch("TrigObj_HLT",TrigObj_HLT,"TrigObj_HLT[nTrigObj]/O");
  T1->Branch("TrigObj_L1",TrigObj_L1,"TrigObj_L1[nTrigObj]/O");
  T1->Branch("TrigObj_Both",TrigObj_Both,"TrigObj_Both[nTrigObj]/O");
  T1->Branch("TrigObj_Ihlt",TrigObj_Ihlt,"TrigObj_Ihlt[nTrigObj]/I");
  T1->Branch("TrigObj_HLTname","std::vector<string>",&TrigObj_HLTname);
  T1->Branch("TrigObj_pdgId",TrigObj_pdgId,"TrigObj_pdgId[nTrigObj]/I");
  T1->Branch("TrigObj_type",TrigObj_type,"TrigObj_type[nTrigObj]/I");
  
  // L1 trigger decision info //
  
  T1->Branch("L1_QuadJet60er2p5",&L1_QuadJet60er2p5,"L1_QuadJet60er2p5/O");
  T1->Branch("L1_HTT280er",&L1_HTT280er,"L1_HTT280er/O");
  T1->Branch("L1_HTT320er",&L1_HTT320er,"L1_HTT320er/O");
  T1->Branch("L1_HTT360er",&L1_HTT360er,"L1_HTT360er/O");
  T1->Branch("L1_HTT400er",&L1_HTT400er,"L1_HTT400er/O");
  T1->Branch("L1_HTT450er",&L1_HTT450er,"L1_HTT450er/O");
  T1->Branch("L1_HTT280er_QuadJet_70_55_40_35_er2p5",&L1_HTT280er_QuadJet_70_55_40_35_er2p5,"L1_HTT280er_QuadJet_70_55_40_35_er2p5/O");
  T1->Branch("L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3",&L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3,"L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3/O");
  T1->Branch("L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3",&L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3,"L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3/O");
  T1->Branch("L1_Mu6_HTT240er",&L1_Mu6_HTT240er,"L1_Mu6_HTT240er/O");
  T1->Branch("L1_SingleJet60",&L1_SingleJet60,"L1_SingleJet60/O");
  
  // Prefire weights //
  
  if(add_prefireweights){
  
  T1->Branch("prefiringweight",&prefiringweight,"prefiringweight/D");
  T1->Branch("prefiringweightup",&prefiringweightup,"prefiringweightup/D");
  T1->Branch("prefiringweightdown",&prefiringweightdown,"prefiringweightdown/D");
  
  }
  
  // MET info //
  
  if(store_CHS_met){
  
  T1->Branch("CHSMET_pt",&miset,"miset/F") ;
  T1->Branch("CHSMET_phi",&misphi,"misphi/F") ;
  T1->Branch("CHSMET_sig",&misetsig,"misetsig/F");
  T1->Branch("CHSMET_sumEt",&sumEt,"sumEt/F");
  
  T1->Branch("CHSMET_covXX",&miset_covXX,"miset_covXX/F") ;
  T1->Branch("CHSMET_covXY",&miset_covXY,"miset_covXY/F") ;
  T1->Branch("CHSMET_covYY",&miset_covYY,"miset_covYY/F") ;
  
  T1->Branch("CHSMET_pt_UnclusEup",&miset_UnclusEup,"miset_CHS_UnclusEup/F") ;
  T1->Branch("CHSMET_pt_UnclusEdn",&miset_UnclusEdn,"miset_CHS_UnclusEdn/F") ;
  T1->Branch("CHSMET_phi_UnclusEup",&misphi_UnclusEup,"CHSMET_phi_UnclusEup/F") ;
  T1->Branch("CHSMET_phi_UnclusEdn",&misphi_UnclusEdn,"CHSMET_phi_UnclusEdn/F") ;
  
  }
  
  if(store_PUPPI_met){
  
  T1->Branch("PuppiMET_pt",&miset_PUPPI,"miset_PUPPI/F") ;
  T1->Branch("PuppiMET_phi",&misphi_PUPPI,"misphi_PUPPI/F") ;
  T1->Branch("PuppiMET_sig",&misetsig_PUPPI,"misetsig_PUPPI/F");
  T1->Branch("PuppiMET_sumEt",&sumEt_PUPPI,"sumEt_PUPPI/F");
  
  T1->Branch("PuppiMET_covXX",&miset_PUPPI_covXX,"miset_PUPPI_covXX/F") ;
  T1->Branch("PuppiMET_covXY",&miset_PUPPI_covXY,"miset_PUPPI_covXY/F") ;
  T1->Branch("PuppiMET_covYY",&miset_PUPPI_covYY,"miset_PUPPI_covYY/F") ;
  
  T1->Branch("PuppiMET_pt_JESup",&miset_PUPPI_JESup,"miset_PUPPI_JESup/F") ;
  T1->Branch("PuppiMET_pt_JESdn",&miset_PUPPI_JESdn,"miset_PUPPI_JESdn/F") ;
  T1->Branch("PuppiMET_pt_JERup",&miset_PUPPI_JERup,"miset_PUPPI_JERup/F") ;
  T1->Branch("PuppiMET_pt_JERdn",&miset_PUPPI_JERdn,"miset_PUPPI_JERdn/F") ;
  T1->Branch("PuppiMET_pt_UnclusEup",&miset_PUPPI_UnclusEup,"miset_PUPPI_UnclusEup/F") ;
  T1->Branch("PuppiMET_pt_UnclusEdn",&miset_PUPPI_UnclusEdn,"miset_PUPPI_UnclusEdn/F") ;
  T1->Branch("PuppiMET_phi_JESup",&misphi_PUPPI_JESup,"misphi_PUPPI_JESup/F") ;
  T1->Branch("PuppiMET_phi_JESdn",&misphi_PUPPI_JESdn,"misphi_PUPPI_JESdn/F") ;
  T1->Branch("PuppiMET_phi_JERup",&misphi_PUPPI_JERup,"misphi_PUPPI_JERup/F") ;
  T1->Branch("PuppiMET_phi_JERdn",&misphi_PUPPI_JERdn,"misphi_PUPPI_JERdn/F") ;
  T1->Branch("PuppiMET_phi_UnclusEup",&misphi_PUPPI_UnclusEup,"misphi_PUPPI_UnclusEup/F") ;
  T1->Branch("PuppiMET_phi_UnclusEdn",&misphi_PUPPI_UnclusEdn,"misphi_PUPPI_UnclusEdn/F") ;
  
  }
  
  // AK8 jet info //
  
  if(store_ak8jets){
  
  T1->Branch("nPFJetAK8",&nPFJetAK8, "nPFJetAK8/I"); 
  T1->Branch("PFJetAK8_pt",PFJetAK8_pt,"PFJetAK8_pt[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_y",PFJetAK8_y,"PFJetAK8_y[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_eta",PFJetAK8_eta,"PFJetAK8_eta[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_phi",PFJetAK8_phi,"PFJetAK8_phi[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_mass",PFJetAK8_mass,"PFJetAK8_mass[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jetID",PFJetAK8_jetID,"PFJetAK8_jetID[nPFJetAK8]/O");
  T1->Branch("PFJetAK8_jetID_tightlepveto",PFJetAK8_jetID_tightlepveto,"PFJetAK8_jetID_tightlepveto[nPFJetAK8]/O");
  T1->Branch("PFJetAK8_jetveto_Flag",PFJetAK8_jetveto_Flag,"PFJetAK8_jetveto_Flag[nPFJetAK8]/O");
  T1->Branch("PFJetAK8_jetveto_eep_Flag",PFJetAK8_jetveto_eep_Flag,"PFJetAK8_jetveto_eep_Flag[nPFJetAK8]/O");
 
  if(store_jet_id_variables){
  T1->Branch("PFJetAK8_CHF",PFJetAK8_CHF,"PFJetAK8_CHF[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_NHF",PFJetAK8_NHF,"PFJetAK8_NHF[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_CEMF",PFJetAK8_CEMF,"PFJetAK8_CEMF[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_NEMF",PFJetAK8_NEMF,"PFJetAK8_NEMF[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_MUF",PFJetAK8_MUF,"PFJetAK8_MUF[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_PHF",PFJetAK8_PHF,"PFJetAK8_PHF[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_EEF",PFJetAK8_EEF,"PFJetAK8_EEF[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_HFHF",PFJetAK8_HFHF,"PFJetAK8_HFHF[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_CHM",PFJetAK8_CHM,"PFJetAK8_CHM[nPFJetAK8]/I");
  T1->Branch("PFJetAK8_NHM",PFJetAK8_NHM,"PFJetAK8_NHM[nPFJetAK8]/I");
  T1->Branch("PFJetAK8_MUM",PFJetAK8_MUM,"PFJetAK8_MUM[nPFJetAK8]/I");
  T1->Branch("PFJetAK8_PHM",PFJetAK8_PHM,"PFJetAK8_PHM[nPFJetAK8]/I");
  T1->Branch("PFJetAK8_EEM",PFJetAK8_EEM,"PFJetAK8_EEM[nPFJetAK8]/I");
  T1->Branch("PFJetAK8_HFHM",PFJetAK8_HFHM,"PFJetAK8_HFHM[nPFJetAK8]/I");
  T1->Branch("PFJetAK8_Neucons",PFJetAK8_Neucons,"PFJetAK8_Neucons[nPFJetAK8]/I");
  T1->Branch("PFJetAK8_Chcons",PFJetAK8_Chcons,"PFJetAK8_Chcons[nPFJetAK8]/I");
  }
  
  T1->Branch("PFJetAK8_JEC",PFJetAK8_JEC,"PFJetAK8_JEC[nPFJetAK8]/F");
  
  T1->Branch("PFJetAK8_JER",PFJetAK8_reso,"PFJetAK8_reso[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_JERup",PFJetAK8_resoup,"PFJetAK8_resoup[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_JERdn",PFJetAK8_resodn,"PFJetAK8_resodn[nPFJetAK8]/F");
  
  T1->Branch("PFJetAK8_msoftdrop",PFJetAK8_sdmass,"PFJetAK8_sdmass[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_tau1",PFJetAK8_tau1,"PFJetAK8_tau1[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_tau2",PFJetAK8_tau2,"PFJetAK8_tau2[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_tau3",PFJetAK8_tau3,"PFJetAK8_tau3[nPFJetAK8]/F");
  
  //T1->Branch("PFJetAK8_btag_DeepCSV",PFJetAK8_btag_DeepCSV,"PFJetAK8_btag_DeepCSV[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_DAK8MD_TvsQCD",PFJetAK8_DeepTag_DAK8_TvsQCD,"PFJetAK8_DeepTag_DAK8_TvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_DAK8MD_WvsQCD",PFJetAK8_DeepTag_DAK8_WvsQCD,"PFJetAK8_DeepTag_DAK8_WvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_DAK8MD_ZvsQCD",PFJetAK8_DeepTag_DAK8_ZvsQCD,"PFJetAK8_DeepTag_DAK8_ZvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_DAK8MD_HvsQCD",PFJetAK8_DeepTag_DAK8_HvsQCD,"PFJetAK8_DeepTag_DAK8_HvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_DAK8MD_bbvsQCD",PFJetAK8_DeepTag_DAK8_bbvsQCD,"PFJetAK8_DeepTag_DAK8_bbvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNet_TvsQCD",PFJetAK8_DeepTag_PNet_TvsQCD,"PFJetAK8_DeepTag_PNet_TvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNet_WvsQCD",PFJetAK8_DeepTag_PNet_WvsQCD,"PFJetAK8_DeepTag_PNet_WvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNet_ZvsQCD",PFJetAK8_DeepTag_PNet_ZvsQCD,"PFJetAK8_DeepTag_PNet_ZvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNet_HbbvsQCD",PFJetAK8_DeepTag_PNet_HbbvsQCD,"PFJetAK8_DeepTag_PNet_HbbvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNet_HccvsQCD",PFJetAK8_DeepTag_PNet_HccvsQCD,"PFJetAK8_DeepTag_PNet_HccvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNet_H4qvsQCD",PFJetAK8_DeepTag_PNet_H4qvsQCD,"PFJetAK8_DeepTag_PNet_H4qvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNetMD_XbbvsQCD",PFJetAK8_DeepTag_PNet_XbbvsQCD,"PFJetAK8_DeepTag_PNet_XbbvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNetMD_XccvsQCD",PFJetAK8_DeepTag_PNet_XccvsQCD,"PFJetAK8_DeepTag_PNet_XccvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNetMD_XqqvsQCD",PFJetAK8_DeepTag_PNet_XqqvsQCD,"PFJetAK8_DeepTag_PNet_XqqvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNetMD_XggvsQCD",PFJetAK8_DeepTag_PNet_XggvsQCD,"PFJetAK8_DeepTag_PNet_XggvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNetMD_XtevsQCD",PFJetAK8_DeepTag_PNet_XtevsQCD,"PFJetAK8_DeepTag_PNet_XtevsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNetMD_XtmvsQCD",PFJetAK8_DeepTag_PNet_XtmvsQCD,"PFJetAK8_DeepTag_PNet_XtmvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNetMD_XttvsQCD",PFJetAK8_DeepTag_PNet_XttvsQCD,"PFJetAK8_DeepTag_PNet_XttvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNetMD_QCD",PFJetAK8_DeepTag_PNet_QCD,"PFJetAK8_DeepTag_PNet_QCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNetMD_QCD0HF",PFJetAK8_DeepTag_PNet_QCD0HF,"PFJetAK8_DeepTag_PNet_QCD0HF[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNetMD_QCD1HF",PFJetAK8_DeepTag_PNet_QCD1HF,"PFJetAK8_DeepTag_PNet_QCD1HF[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PNetMD_QCD2HF",PFJetAK8_DeepTag_PNet_QCD2HF,"PFJetAK8_DeepTag_PNet_QCD2HF[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_Xbb",PFJetAK8_DeepTag_PartT_Xbb,"PFJetAK8_DeepTag_PartT_Xbb[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_Xcc",PFJetAK8_DeepTag_PartT_Xcc,"PFJetAK8_DeepTag_PartT_Xcc[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_Xcs",PFJetAK8_DeepTag_PartT_Xcs,"PFJetAK8_DeepTag_PartT_Xcs[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_Xqq",PFJetAK8_DeepTag_PartT_Xqq,"PFJetAK8_DeepTag_PartT_Xqq[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_TopbWqq",PFJetAK8_DeepTag_PartT_TopbWqq,"PFJetAK8_DeepTag_PartT_TopbWqq[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_TopbWq",PFJetAK8_DeepTag_PartT_TopbWq,"PFJetAK8_DeepTag_PartT_TopbWq[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_TopbWev",PFJetAK8_DeepTag_PartT_TopbWev,"PFJetAK8_DeepTag_PartT_TopbWev[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_TopbWmv",PFJetAK8_DeepTag_PartT_TopbWmv,"PFJetAK8_DeepTag_PartT_TopbWmv[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_TopbWtauv",PFJetAK8_DeepTag_PartT_TopbWtauv,"PFJetAK8_DeepTag_PartT_TopbWtauv[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_QCD",PFJetAK8_DeepTag_PartT_QCD,"PFJetAK8_DeepTag_PartT_QCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_XWW4q",PFJetAK8_DeepTag_PartT_XWW4q,"PFJetAK8_DeepTag_PartT_XWW4q[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_XWW3q",PFJetAK8_DeepTag_PartT_XWW3q,"PFJetAK8_DeepTag_PartT_XWW3q[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_XWWqqev",PFJetAK8_DeepTag_PartT_XWWqqev,"PFJetAK8_DeepTag_PartT_XWWqqev[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_XWWqqmv",PFJetAK8_DeepTag_PartT_XWWqqmv,"PFJetAK8_DeepTag_PartT_XWWqqmv[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_TvsQCD",PFJetAK8_DeepTag_PartT_TvsQCD,"PFJetAK8_DeepTag_PartT_TvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_WvsQCD",PFJetAK8_DeepTag_PartT_WvsQCD,"PFJetAK8_DeepTag_PartT_WvsQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_DeepTag_PartT_ZvsQCD",PFJetAK8_DeepTag_PartT_ZvsQCD,"PFJetAK8_DeepTag_PartT_ZvsQCD[nPFJetAK8]/F");
  
  T1->Branch("PFJetAK8_particleNet_massCorr",PFJetAK8_particleNet_massCorr,"PFJetAK8_particleNet_massCorr[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_partT_massCorr_generic",PFJetAK8_partT_massCorr_generic,"PFJetAK8_partT_massCorr_generic[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_partT_massCorr_twoprong",PFJetAK8_partT_massCorr_twoprong,"PFJetAK8_partT_massCorr_twoprong[nPFJetAK8]/F");
  
  T1->Branch("PFJetAK8_sub1pt",PFJetAK8_sub1pt,"PFJetAK8_sub1pt[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_sub1eta",PFJetAK8_sub1eta,"PFJetAK8_sub1eta[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_sub1phi",PFJetAK8_sub1phi,"PFJetAK8_sub1phi[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_sub1mass",PFJetAK8_sub1mass,"PFJetAK8_sub1mass[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_sub1JEC",PFJetAK8_sub1JEC,"PFJetAK8_sub1JEC[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_sub1btag",PFJetAK8_sub1btag,"PFJetAK8_sub1btag[nPFJetAK8]/F");
  
  T1->Branch("PFJetAK8_sub2pt",PFJetAK8_sub2pt,"PFJetAK8_sub2pt[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_sub2eta",PFJetAK8_sub2eta,"PFJetAK8_sub2eta[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_sub2phi",PFJetAK8_sub2phi,"PFJetAK8_sub2phi[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_sub2mass",PFJetAK8_sub2mass,"PFJetAK8_sub2mass[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_sub2JEC",PFJetAK8_sub2JEC,"PFJetAK8_sub2JEC[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_sub2btag",PFJetAK8_sub2btag,"PFJetAK8_sub2btag[nPFJetAK8]/F");
  
  T1->Branch("PFJetAK8_jesup_AbsoluteStat",PFJetAK8_jesup_AbsoluteStat,"PFJetAK8_jesup_AbsoluteStat[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_AbsoluteScale",PFJetAK8_jesup_AbsoluteScale,"PFJetAK8_jesup_AbsoluteScale[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_AbsoluteMPFBias",PFJetAK8_jesup_AbsoluteMPFBias,"PFJetAK8_jesup_AbsoluteMPFBias[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_FlavorQCD",PFJetAK8_jesup_FlavorQCD,"PFJetAK8_jesup_FlavorQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_Fragmentation",PFJetAK8_jesup_Fragmentation,"PFJetAK8_jesup_Fragmentation[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_PileUpDataMC",PFJetAK8_jesup_PileUpDataMC,"PFJetAK8_jesup_PileUpDataMC[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_PileUpPtBB",PFJetAK8_jesup_PileUpPtBB,"PFJetAK8_jesup_PileUpPtBB[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_PileUpPtEC1",PFJetAK8_jesup_PileUpPtEC1,"PFJetAK8_jesup_PileUpPtEC1[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_PileUpPtEC2",PFJetAK8_jesup_PileUpPtEC2,"PFJetAK8_jesup_PileUpPtEC2[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_PileUpPtRef",PFJetAK8_jesup_PileUpPtRef,"PFJetAK8_jesup_PileUpPtRef[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_RelativeFSR",PFJetAK8_jesup_RelativeFSR,"PFJetAK8_jesup_RelativeFSR[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_RelativeJEREC1",PFJetAK8_jesup_RelativeJEREC1,"PFJetAK8_jesup_RelativeJEREC1[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_RelativeJEREC2",PFJetAK8_jesup_RelativeJEREC2,"PFJetAK8_jesup_RelativeJEREC2[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_RelativePtBB",PFJetAK8_jesup_RelativePtBB,"PFJetAK8_jesup_RelativePtBB[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_RelativePtEC1",PFJetAK8_jesup_RelativePtEC1,"PFJetAK8_jesup_RelativePtEC1[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_RelativePtEC2",PFJetAK8_jesup_RelativePtEC2,"PFJetAK8_jesup_RelativePtEC2[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_RelativeBal",PFJetAK8_jesup_RelativeBal,"PFJetAK8_jesup_RelativeBal[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_RelativeSample",PFJetAK8_jesup_RelativeSample,"PFJetAK8_jesup_RelativeSample[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_RelativeStatEC",PFJetAK8_jesup_RelativeStatEC,"PFJetAK8_jesup_RelativeStatEC[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_RelativeStatFSR",PFJetAK8_jesup_RelativeStatFSR,"PFJetAK8_jesup_RelativeStatFSR[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_SinglePionECAL",PFJetAK8_jesup_SinglePionECAL,"PFJetAK8_jesup_SinglePionECAL[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_SinglePionHCAL",PFJetAK8_jesup_SinglePionHCAL,"PFJetAK8_jesup_SinglePionHCAL[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_TimePtEta",PFJetAK8_jesup_TimePtEta,"PFJetAK8_jesup_TimePtEta[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesup_Total",PFJetAK8_jesup_Total,"PFJetAK8_jesup_Total[nPFJetAK8]/F");
  
  T1->Branch("PFJetAK8_jesdn_AbsoluteStat",PFJetAK8_jesdn_AbsoluteStat,"PFJetAK8_jesdn_AbsoluteStat[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_AbsoluteScale",PFJetAK8_jesdn_AbsoluteScale,"PFJetAK8_jesdn_AbsoluteScale[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_AbsoluteMPFBias",PFJetAK8_jesdn_AbsoluteMPFBias,"PFJetAK8_jesdn_AbsoluteMPFBias[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_FlavorQCD",PFJetAK8_jesdn_FlavorQCD,"PFJetAK8_jesdn_FlavorQCD[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_Fragmentation",PFJetAK8_jesdn_Fragmentation,"PFJetAK8_jesdn_Fragmentation[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_PileUpDataMC",PFJetAK8_jesdn_PileUpDataMC,"PFJetAK8_jesdn_PileUpDataMC[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_PileUpPtBB",PFJetAK8_jesdn_PileUpPtBB,"PFJetAK8_jesdn_PileUpPtBB[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_PileUpPtEC1",PFJetAK8_jesdn_PileUpPtEC1,"PFJetAK8_jesdn_PileUpPtEC1[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_PileUpPtEC2",PFJetAK8_jesdn_PileUpPtEC2,"PFJetAK8_jesdn_PileUpPtEC2[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_PileUpPtRef",PFJetAK8_jesdn_PileUpPtRef,"PFJetAK8_jesdn_PileUpPtRef[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_RelativeFSR",PFJetAK8_jesdn_RelativeFSR,"PFJetAK8_jesdn_RelativeFSR[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_RelativeJEREC1",PFJetAK8_jesdn_RelativeJEREC1,"PFJetAK8_jesdn_RelativeJEREC1[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_RelativeJEREC2",PFJetAK8_jesdn_RelativeJEREC2,"PFJetAK8_jesdn_RelativeJEREC2[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_RelativePtBB",PFJetAK8_jesdn_RelativePtBB,"PFJetAK8_jesdn_RelativePtBB[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_RelativePtEC1",PFJetAK8_jesdn_RelativePtEC1,"PFJetAK8_jesdn_RelativePtEC1[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_RelativePtEC2",PFJetAK8_jesdn_RelativePtEC2,"PFJetAK8_jesdn_RelativePtEC2[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_RelativeBal",PFJetAK8_jesdn_RelativeBal,"PFJetAK8_jesdn_RelativeBal[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_RelativeSample",PFJetAK8_jesdn_RelativeSample,"PFJetAK8_jesdn_RelativeSample[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_RelativeStatEC",PFJetAK8_jesdn_RelativeStatEC,"PFJetAK8_jesdn_RelativeStatEC[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_RelativeStatFSR",PFJetAK8_jesdn_RelativeStatFSR,"PFJetAK8_jesdn_RelativeStatFSR[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_SinglePionECAL",PFJetAK8_jesdn_SinglePionECAL,"PFJetAK8_jesdn_SinglePionECAL[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_SinglePionHCAL",PFJetAK8_jesdn_SinglePionHCAL,"PFJetAK8_jesdn_SinglePionHCAL[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_TimePtEta",PFJetAK8_jesdn_TimePtEta,"PFJetAK8_jesdn_TimePtEta[nPFJetAK8]/F");
  T1->Branch("PFJetAK8_jesdn_Total",PFJetAK8_jesdn_Total,"PFJetAK8_jesdn_Total[nPFJetAK8]/F");
  
  T1->Branch("PFJetAK8_nBHadrons",PFJetAK8_nBHadrons,"PFJetAK8_nBHadrons[nPFJetAK8]/I");
  T1->Branch("PFJetAK8_nCHadrons",PFJetAK8_nCHadrons,"PFJetAK8_nCHadrons[nPFJetAK8]/I");
  
  //gROOT->ProcessLine(".L CustomRootDict.cc+");
  
  if(store_fatjet_constituents){
    T1->Branch("nPFJetAK8_cons",&nPFJetAK8_cons,"nPFJetAK8_cons/I");
    T1->Branch("PFJetAK8_cons_pt",PFJetAK8_cons_pt, "PFJetAK8_cons_pt[nPFJetAK8_cons]/F");
    T1->Branch("PFJetAK8_cons_eta",PFJetAK8_cons_eta, "PFJetAK8_cons_eta[nPFJetAK8_cons]/F");
    T1->Branch("PFJetAK8_cons_phi",PFJetAK8_cons_phi, "PFJetAK8_cons_phi[nPFJetAK8_cons]/F");
    T1->Branch("PFJetAK8_cons_mass",PFJetAK8_cons_mass, "PFJetAK8_cons_mass[nPFJetAK8_cons]/F");
    T1->Branch("PFJetAK8_cons_pdgId",PFJetAK8_cons_pdgId, "PFJetAK8_cons_pdgId[nPFJetAK8_cons]/I");
    T1->Branch("PFJetAK8_cons_jetIndex",PFJetAK8_cons_jetIndex, "PFJetAK8_cons_jetIndex[nPFJetAK8_cons]/I");
  }
  
  }

  // AK4 jet info //
 
  if(store_ak4jets){
 
  T1->Branch("nPFJetAK4",&nPFJetAK4,"nPFJetAK4/I"); 

  T1->Branch("PFJetAK4_pt",PFJetAK4_pt,"PFJetAK4_pt[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_eta",PFJetAK4_eta,"PFJetAK4_eta[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_y",PFJetAK4_y,"PFJetAK4_y[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_phi",PFJetAK4_phi,"PFJetAK4_phi[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_mass",PFJetAK4_mass,"PFJetAK4_mass[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_area",PFJetAK4_area,"PFJetAK4_area[nPFJetAK4]/F");
  
  T1->Branch("PFJetAK4_jetID",PFJetAK4_jetID,"PFJetAK4_jetID[nPFJetAK4]/O");
  T1->Branch("PFJetAK4_jetID_tightlepveto",PFJetAK4_jetID_tightlepveto,"PFJetAK4_jetID_tightlepveto[nPFJetAK4]/O");
  
  T1->Branch("PFJetAK4_jetveto_Flag",PFJetAK4_jetveto_Flag,"PFJetAK8_jetveto_Flag[nPFJetAK4]/O");
  T1->Branch("PFJetAK4_jetveto_eep_Flag",PFJetAK4_jetveto_eep_Flag,"PFJetAK8_jetveto_eep_Flag[nPFJetAK4]/O");
  
  T1->Branch("PFJetAK4_btag_DeepCSV",PFJetAK4_btag_DeepCSV,"PFJetAK4_btag_DeepCSV[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btag_DeepFlav",PFJetAK4_btag_DeepFlav,"PFJetAK4_btag_DeepFlav[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagDeepFlavB",PFJetAK4_btagDeepFlavB,"PFJetAK4_btagDeepFlavB[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagDeepFlavCvB",PFJetAK4_btagDeepFlavCvB,"PFJetAK4_btagDeepFlavCvB[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagDeepFlavCvL",PFJetAK4_btagDeepFlavCvL,"PFJetAK4_btagDeepFlavCvL[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagDeepFlavQG",PFJetAK4_btagDeepFlavQG,"PFJetAK4_btagDeepFlavQG[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagPNetB",PFJetAK4_btagPNetB,"PFJetAK4_btagPNetB[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagPNetCvB",PFJetAK4_btagPNetCvB,"PFJetAK4_btagPNetCvB[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagPNetCvL",PFJetAK4_btagPNetCvL,"PFJetAK4_btagPNetCvL[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagPNetCvNotB",PFJetAK4_btagPNetCvNotB,"PFJetAK4_btagPNetCvNotB[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagPNetQG",PFJetAK4_btagPNetQG,"PFJetAK4_btagPNetQG[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagRobustParTAK4B",PFJetAK4_btagRobustParTAK4B,"PFJetAK4_btagRobustParTAK4B[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagRobustParTAK4CvB",PFJetAK4_btagRobustParTAK4CvB,"PFJetAK4_btagRobustParTAK4CvB[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagRobustParTAK4CvL",PFJetAK4_btagRobustParTAK4CvL,"PFJetAK4_btagRobustParTAK4CvL[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_btagRobustParTAK4QG",PFJetAK4_btagRobustParTAK4QG,"PFJetAK4_btagRobustParTAK4QG[nPFJetAK4]/F");
  
  T1->Branch("PFJetAK4_PNetRegPtRawCorr",PFJetAK4_PNetRegPtRawCorr,"PFJetAK4_PNetRegPtRawCorr[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_PNetRegPtRawCorrNeutrino",PFJetAK4_PNetRegPtRawCorrNeutrino,"PFJetAK4_PNetRegPtRawCorrNeutrino[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_PNetRegPtRawRes",PFJetAK4_PNetRegPtRawRes,"PFJetAK4_PNetRegPtRawRes[nPFJetAK4]/F");
  
  T1->Branch("PFJetAK4_JEC",PFJetAK4_JEC,"PFJetAK4_JEC[nPFJetAK4]/F");
  
  T1->Branch("PFJetAK4_JER",PFJetAK4_reso,"PFJetAK4_reso[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_JERup",PFJetAK4_resoup,"PFJetAK4_resoup[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_JERdn",PFJetAK4_resodn,"PFJetAK4_resodn[nPFJetAK4]/F"); 
  
  T1->Branch("PFJetAK4_hadronflav",PFJetAK4_hadronflav,"PFJetAK4_hadronflav[nPFJetAK4]/I");
  T1->Branch("PFJetAK4_partonflav",PFJetAK4_partonflav,"PFJetAK4_partonflav[nPFJetAK4]/I");
  T1->Branch("PFJetAK4_qgl",PFJetAK4_qgl,"PFJetAK4_qgl[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_PUID",PFJetAK4_PUID,"PFJetAK4_PUID[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_charge_kappa_0p3",PFJetAK4_charge_kappa_0p3,"PFJetAK4_charge_kappa_0p3[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_charge_kappa_0p6",PFJetAK4_charge_kappa_0p6,"PFJetAK4_charge_kappa_0p6[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_charge_kappa_1p0",PFJetAK4_charge_kappa_1p0,"PFJetAK4_charge_kappa_1p0[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_charged_ptsum",PFJetAK4_charged_ptsum,"PFJetAK4_charged_ptsum[nPFJetAK4]/F");
  
  T1->Branch("PFJetAK4_jesup_AbsoluteStat",PFJetAK4_jesup_AbsoluteStat,"PFJetAK4_jesup_AbsoluteStat[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_AbsoluteScale",PFJetAK4_jesup_AbsoluteScale,"PFJetAK4_jesup_AbsoluteScale[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_AbsoluteMPFBias",PFJetAK4_jesup_AbsoluteMPFBias,"PFJetAK4_jesup_AbsoluteMPFBias[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_FlavorQCD",PFJetAK4_jesup_FlavorQCD,"PFJetAK4_jesup_FlavorQCD[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_Fragmentation",PFJetAK4_jesup_Fragmentation,"PFJetAK4_jesup_Fragmentation[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_PileUpDataMC",PFJetAK4_jesup_PileUpDataMC,"PFJetAK4_jesup_PileUpDataMC[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_PileUpPtBB",PFJetAK4_jesup_PileUpPtBB,"PFJetAK4_jesup_PileUpPtBB[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_PileUpPtEC1",PFJetAK4_jesup_PileUpPtEC1,"PFJetAK4_jesup_PileUpPtEC1[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_PileUpPtEC2",PFJetAK4_jesup_PileUpPtEC2,"PFJetAK4_jesup_PileUpPtEC2[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_PileUpPtRef",PFJetAK4_jesup_PileUpPtRef,"PFJetAK4_jesup_PileUpPtRef[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_RelativeFSR",PFJetAK4_jesup_RelativeFSR,"PFJetAK4_jesup_RelativeFSR[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_RelativeJEREC1",PFJetAK4_jesup_RelativeJEREC1,"PFJetAK4_jesup_RelativeJEREC1[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_RelativeJEREC2",PFJetAK4_jesup_RelativeJEREC2,"PFJetAK4_jesup_RelativeJEREC2[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_RelativePtBB",PFJetAK4_jesup_RelativePtBB,"PFJetAK4_jesup_RelativePtBB[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_RelativePtEC1",PFJetAK4_jesup_RelativePtEC1,"PFJetAK4_jesup_RelativePtEC1[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_RelativePtEC2",PFJetAK4_jesup_RelativePtEC2,"PFJetAK4_jesup_RelativePtEC2[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_RelativeBal",PFJetAK4_jesup_RelativeBal,"PFJetAK4_jesup_RelativeBal[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_RelativeSample",PFJetAK4_jesup_RelativeSample,"PFJetAK4_jesup_RelativeSample[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_RelativeStatEC",PFJetAK4_jesup_RelativeStatEC,"PFJetAK4_jesup_RelativeStatEC[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_RelativeStatFSR",PFJetAK4_jesup_RelativeStatFSR,"PFJetAK4_jesup_RelativeStatFSR[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_SinglePionECAL",PFJetAK4_jesup_SinglePionECAL,"PFJetAK4_jesup_SinglePionECAL[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_SinglePionHCAL",PFJetAK4_jesup_SinglePionHCAL,"PFJetAK4_jesup_SinglePionHCAL[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_TimePtEta",PFJetAK4_jesup_TimePtEta,"PFJetAK4_jesup_TimePtEta[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesup_Total",PFJetAK4_jesup_Total,"PFJetAK4_jesup_Total[nPFJetAK4]/F");
  
  T1->Branch("PFJetAK4_jesdn_AbsoluteStat",PFJetAK4_jesdn_AbsoluteStat,"PFJetAK4_jesdn_AbsoluteStat[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_AbsoluteScale",PFJetAK4_jesdn_AbsoluteScale,"PFJetAK4_jesdn_AbsoluteScale[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_AbsoluteMPFBias",PFJetAK4_jesdn_AbsoluteMPFBias,"PFJetAK4_jesdn_AbsoluteMPFBias[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_FlavorQCD",PFJetAK4_jesdn_FlavorQCD,"PFJetAK4_jesdn_FlavorQCD[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_Fragmentation",PFJetAK4_jesdn_Fragmentation,"PFJetAK4_jesdn_Fragmentation[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_PileUpDataMC",PFJetAK4_jesdn_PileUpDataMC,"PFJetAK4_jesdn_PileUpDataMC[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_PileUpPtBB",PFJetAK4_jesdn_PileUpPtBB,"PFJetAK4_jesdn_PileUpPtBB[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_PileUpPtEC1",PFJetAK4_jesdn_PileUpPtEC1,"PFJetAK4_jesdn_PileUpPtEC1[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_PileUpPtEC2",PFJetAK4_jesdn_PileUpPtEC2,"PFJetAK4_jesdn_PileUpPtEC2[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_PileUpPtRef",PFJetAK4_jesdn_PileUpPtRef,"PFJetAK4_jesdn_PileUpPtRef[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_RelativeFSR",PFJetAK4_jesdn_RelativeFSR,"PFJetAK4_jesdn_RelativeFSR[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_RelativeJEREC1",PFJetAK4_jesdn_RelativeJEREC1,"PFJetAK4_jesdn_RelativeJEREC1[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_RelativeJEREC2",PFJetAK4_jesdn_RelativeJEREC2,"PFJetAK4_jesdn_RelativeJEREC2[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_RelativePtBB",PFJetAK4_jesdn_RelativePtBB,"PFJetAK4_jesdn_RelativePtBB[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_RelativePtEC1",PFJetAK4_jesdn_RelativePtEC1,"PFJetAK4_jesdn_RelativePtEC1[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_RelativePtEC2",PFJetAK4_jesdn_RelativePtEC2,"PFJetAK4_jesdn_RelativePtEC2[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_RelativeBal",PFJetAK4_jesdn_RelativeBal,"PFJetAK4_jesdn_RelativeBal[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_RelativeSample",PFJetAK4_jesdn_RelativeSample,"PFJetAK4_jesdn_RelativeSample[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_RelativeStatEC",PFJetAK4_jesdn_RelativeStatEC,"PFJetAK4_jesdn_RelativeStatEC[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_RelativeStatFSR",PFJetAK4_jesdn_RelativeStatFSR,"PFJetAK4_jesdn_RelativeStatFSR[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_SinglePionECAL",PFJetAK4_jesdn_SinglePionECAL,"PFJetAK4_jesdn_SinglePionECAL[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_SinglePionHCAL",PFJetAK4_jesdn_SinglePionHCAL,"PFJetAK4_jesdn_SinglePionHCAL[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_TimePtEta",PFJetAK4_jesdn_TimePtEta,"PFJetAK4_jesdn_TimePtEta[nPFJetAK4]/F");
  T1->Branch("PFJetAK4_jesdn_Total",PFJetAK4_jesdn_Total,"PFJetAK4_jesdn_Total[nPFJetAK4]/F");
  
  if(read_btagSF){
  
	T1->Branch("PFJetAK4_btag_DeepFlav_SF",PFJetAK4_btag_DeepFlav_SF,"PFJetAK4_btag_DeepFlav_SF[nPFJetAK4]/F");
	T1->Branch("PFJetAK4_btag_DeepFlav_SF_up",PFJetAK4_btag_DeepFlav_SF_up,"PFJetAK4_btag_DeepFlav_SF_up[nPFJetAK4]/F");
	T1->Branch("PFJetAK4_btag_DeepFlav_SF_dn",PFJetAK4_btag_DeepFlav_SF_dn,"PFJetAK4_btag_DeepFlav_SF_dn[nPFJetAK4]/F");
  
  }
    
  }
  
  // Muon info //
  
  if(store_muons){
  
  T1->Branch("nMuon",&nMuon,"nMuon/I");
  
  T1->Branch("Muon_isPF",Muon_isPF,"Muon_isPF[nMuon]/O");
  T1->Branch("Muon_isGL",Muon_isGL,"Muon_isGL[nMuon]/O");
  T1->Branch("Muon_isTRK",Muon_isTRK,"Muon_isTRK[nMuon]/O");
  T1->Branch("Muon_isStandAloneMuon",Muon_isStandAloneMuon,"Muon_isStandAloneMuon[nMuon]/O");
  T1->Branch("Muon_isLoose",Muon_isLoose,"Muon_isLoose[nMuon]/O");
  T1->Branch("Muon_isGoodGL",Muon_isGoodGL,"Muon_isGoodGL[nMuon]/O");
  T1->Branch("Muon_isMed",Muon_isMed,"Muon_isMed[nMuon]/O");
  T1->Branch("Muon_isMedPr",Muon_isMedPr,"Muon_isMedPr[nMuon]/O");
  T1->Branch("Muon_mediumPromptId",Muon_mediumPromptId,"Muon_mediumPromptId[nMuon]/O");
  T1->Branch("Muon_isTight",Muon_isTight,"Muon_isTight[nMuon]/O");
  T1->Branch("Muon_isHighPt",Muon_isHighPt,"Muon_isHighPt[nMuon]/O"); 
  T1->Branch("Muon_isHighPttrk",Muon_isHighPttrk,"Muon_isHighPttrk[nMuon]/O");
  T1->Branch("Muon_MVAID",Muon_MVAID,"Muon_MVAID[nMuon]/I");
  T1->Branch("Muon_mvaMuID",Muon_mvaMuID,"Muon_mvaMuID[nMuon]/F");
  T1->Branch("Muon_mvaMuID_WP",Muon_mvaMuID_WP,"Muon_mvaMuID_WP[nMuon]/I");
 
  T1->Branch("Muon_pt",Muon_pt,"Muon_pt[nMuon]/F");
  T1->Branch("Muon_p",Muon_p,"Muon_p[nMuon]/F");
  T1->Branch("Muon_eta",Muon_eta,"Muon_eta[nMuon]/F");
  T1->Branch("Muon_phi",Muon_phi,"Muon_phi[nMuon]/F");
  T1->Branch("Muon_tunePBestTrack_pt",Muon_tunePBestTrack_pt,"Muon_tunePBestTrack_pt[nMuon]/F");
  
  T1->Branch("Muon_dxy",Muon_dxy,"Muon_dxy[nMuon]/F");
  T1->Branch("Muon_dxybs",Muon_dxybs,"Muon_dxybs[nMuon]/F");
  T1->Branch("Muon_dz",Muon_dz,"Muon_dz[nMuon]/F");
  T1->Branch("Muon_dxyErr",Muon_dxyErr,"Muon_dxyErr[nMuon]/F");
  T1->Branch("Muon_dzErr",Muon_dzErr,"Muon_dzErr[nMuon]/F");
  T1->Branch("Muon_ip3d",Muon_ip3d,"Muon_ip3d[nMuon]/F");   // correct the name to sip3d next time
  T1->Branch("Muon_sip3d",Muon_sip3d,"Muon_sip3d[nMuon]/F");
 
  if(store_muon_id_variables){
	  
	  T1->Branch("Muon_valfrac",Muon_valfrac,"Muon_valfrac[nMuon]/F"); 
	  T1->Branch("Muon_chi",Muon_chi,"Muon_chi[nMuon]/F");
	  T1->Branch("Muon_posmatch",Muon_posmatch,"Muon_posmatch[nMuon]/F");
	  T1->Branch("Muon_trkink",Muon_trkink,"Muon_trkink[nMuon]/F");
	  T1->Branch("Muon_segcom",Muon_segcom,"Muon_segcom[nMuon]/F");
	  T1->Branch("Muon_hit",Muon_hit,"Muon_hit[nMuon]/F");
	  T1->Branch("Muon_mst",Muon_mst,"Muon_mst[nMuon]/F");
	  T1->Branch("Muon_trklay",Muon_trklay,"Muon_trklay[nMuon]/F"); 
	  T1->Branch("Muon_pixhit",Muon_pixhit,"Muon_pixhit[nMuon]/F");
	 
  }
  
  if(store_additional_muon_id_variables){
  
	T1->Branch("Muon_ptErr",Muon_ptErr,"Muon_ptErr[nMuon]/F");
	T1->Branch("Muon_ndf",Muon_ndf,"Muon_ndf[nMuon]/I");
	T1->Branch("Muon_ecal",Muon_ecal,"Muon_ecal[nMuon]/F");
	T1->Branch("Muon_hcal",Muon_hcal,"Muon_hcal[nMuon]/F");
	T1->Branch("Muon_dxy_sv",Muon_dxy_sv,"Muon_dxy_sv[nMuon]/F");
	
	T1->Branch("Muon_TightID",Muon_TightID,"Muon_TightID[nMuon]/O");	 
  
  }
  
  // Packed WP-based numbers //
  T1->Branch("Muon_PF_iso",Muon_PF_iso,"Muon_PF_iso[nMuon]/i");
  T1->Branch("Muon_Mini_iso",Muon_Mini_iso,"Muon_Mini_iso[nMuon]/i");
  T1->Branch("Muon_multiIsoId",Muon_multiIsoId,"Muon_multiIsoId[nMuon]/i");
  T1->Branch("Muon_puppiIsoId",Muon_puppiIsoId,"Muon_puppiIsoId[nMuon]/i");
  T1->Branch("Muon_tkIsoId",Muon_tkIsoId,"Muon_tkIsoId[nMuon]/i");
  // Values //
  T1->Branch("Muon_pfiso",Muon_pfiso,"Muon_pfiso[nMuon]/F");
  T1->Branch("Muon_pfiso03",Muon_pfiso03,"Muon_pfiso03[nMuon]/F");
  //T1->Branch("Muon_minisoch", Muon_minchiso, "Muon_minchiso[nMuon]/F");
  //T1->Branch("Muon_minisonh", Muon_minnhiso, "Muon_minnhiso[nMuon]/F");
  //T1->Branch("Muon_minisoph", Muon_minphiso, "Muon_minphiso[nMuon]/F");
  T1->Branch("Muon_minisoall", Muon_minisoall, "Muon_minisoall[nMuon]/F");
  T1->Branch("Muon_miniPFRelIso_all", Muon_miniPFRelIso_all, "Muon_miniPFRelIso_all[nMuon]/F");
  T1->Branch("Muon_miniPFRelIso_Chg", Muon_miniPFRelIso_Chg, "Muon_miniPFRelIso_Chg[nMuon]/F");
  
  T1->Branch("Muon_corrected_pt",Muon_corrected_pt,"Muon_corrected_pt[nMuon]/F");
  T1->Branch("Muon_correctedUp_pt",Muon_correctedUp_pt,"Muon_correctedUp_pt[nMuon]/F");
  T1->Branch("Muon_correctedDown_pt",Muon_correctedDown_pt,"Muon_correctedDown_pt[nMuon]/F");
  
  }
  
  // Electron info //
  
  if(store_electrons){
  
  T1->Branch("nElectron",&nElectron,"nElectron/I");
  T1->Branch("Electron_pt",Electron_pt,"Electron_pt[nElectron]/F");
  T1->Branch("Electron_eta",Electron_eta,"Electron_eta[nElectron]/F");
  T1->Branch("Electron_phi",Electron_phi,"Electron_phi[nElectron]/F");
  T1->Branch("Electron_p",Electron_p,"Electron_p[nElectron]/F");
  T1->Branch("Electron_e",Electron_e,"Electron_e[nElectron]/F");
  
  T1->Branch("Electron_supcl_eta",Electron_supcl_eta,"Electron_supcl_eta[nElectron]/F");
  T1->Branch("Electron_supcl_phi",Electron_supcl_phi,"Electron_supcl_phi[nElectron]/F");
  T1->Branch("Electron_supcl_e",Electron_supcl_e,"Electron_supcl_e[nElectron]/F");
  T1->Branch("Electron_supcl_rawE",Electron_supcl_rawE,"Electron_supcl_rawE[nElectron]/F");
  
  T1->Branch("Electron_cutbased_id",Electron_cutbased_id,"Electron_cutbased_id[nElectron]/I");
  T1->Branch("Electron_mvaid_Fallv2WP90",Electron_mvaid_Fallv2WP90,"Electron_mvaid_Fallv2WP90[nElectron]/O");
  T1->Branch("Electron_mvaid_Fallv2WP90_noIso",Electron_mvaid_Fallv2WP90_noIso,"Electron_mvaid_Fallv2WP90_noIso[nElectron]/O");
  T1->Branch("Electron_mvaid_Fallv2WP80",Electron_mvaid_Fallv2WP80,"Electron_mvaid_Fallv2WP80[nElectron]/O");
  T1->Branch("Electron_mvaid_Fallv2WP80_noIso",Electron_mvaid_Fallv2WP80_noIso,"Electron_mvaid_Fallv2WP80_noIso[nElectron]/O");
  T1->Branch("Electron_mvaid_Fallv2WPLoose",Electron_mvaid_Fallv2WPLoose,"Electron_mvaid_Fallv2WPLoose[nElectron]/O");
  T1->Branch("Electron_mvaid_Fallv2WPLoose_noIso",Electron_mvaid_Fallv2WPLoose_noIso,"Electron_mvaid_Fallv2WPLoose_noIso[nElectron]/O");
  T1->Branch("Electron_mvaid_Winter22v1WP90",Electron_mvaid_Winter22v1WP90,"Electron_mvaid_Winter22v1WP90[nElectron]/O");
  T1->Branch("Electron_mvaid_Winter22v1WP90_noIso",Electron_mvaid_Winter22v1WP90_noIso,"Electron_mvaid_Winter22v1WP90_noIso[nElectron]/O");
  T1->Branch("Electron_mvaid_Winter22v1WP80",Electron_mvaid_Winter22v1WP80,"Electron_mvaid_Winter22v1WP80[nElectron]/O");
  T1->Branch("Electron_mvaid_Winter22v1WP80_noIso",Electron_mvaid_Winter22v1WP80_noIso,"Electron_mvaid_Winter22v1WP80_noIso[nElectron]/O");
  //T1->Branch("Electron_mvaid_Winter22v1WPLoose",Electron_mvaid_Winter22v1WPLoose,"Electron_mvaid_Winter22v1WPLoose[nElectron]/O");
  //T1->Branch("Electron_mvaid_Winter22v1WPLoose_noIso",Electron_mvaid_Winter22v1WPLoose_noIso,"Electron_mvaid_Winter22v1WPLoose_noIso[nElectron]/O");
  
  T1->Branch("Electron_mvaid_Fallv2_value",Electron_mvaid_Fallv2_value,"Electron_mvaid_Fallv2_value[nElectron]/F");
  T1->Branch("Electron_mvaid_Fallv2noIso_value",Electron_mvaid_Fallv2noIso_value,"Electron_mvaid_Fallv2noIso_value[nElectron]/F");
  T1->Branch("Electron_mvaid_Winter22IsoV1_value",Electron_mvaid_Winter22IsoV1_value,"Electron_mvaid_Winter22IsoV1_value[nElectron]/F");
  T1->Branch("Electron_mvaid_Winter22NoIsoV1_value",Electron_mvaid_Winter22NoIsoV1_value,"Electron_mvaid_Winter22NoIsoV1_value[nElectron]/F");
  
  T1->Branch("Electron_dxy",Electron_dxy,"Electron_dxy[nElectron]/F");
  T1->Branch("Electron_dxyErr",Electron_dxyErr,"Electron_dxyErr[nElectron]/F");
  T1->Branch("Electron_dz",Electron_dz,"Electron_dz[nElectron]/F");
  T1->Branch("Electron_dzErr",Electron_dzErr,"Electron_dzErr[nElectron]/F");
  T1->Branch("Electron_ip3d",Electron_ip3d,"Electron_ip3d[nElectron]/F"); // correct the name to sip3d next time
  T1->Branch("Electron_sip3d",Electron_sip3d,"Electron_sip3d[nElectron]/F");
  
  if(store_electron_id_variables){
  
	T1->Branch("Electron_sigmaieta", Electron_sigmaieta, "Electron_sigmaieta[nElectron]/F");
	T1->Branch("Electron_sigmaiphi", Electron_sigmaiphi, "Electron_sigmaiphi[nElectron]/F");
	T1->Branch("Electron_etain",Electron_etain,"Electron_etain[nElectron]/F");
	T1->Branch("Electron_phiin",Electron_phiin,"Electron_phiin[nElectron]/F");
	T1->Branch("Electron_hovere",Electron_hovere,"Electron_hovere[nElectron]/F");
	T1->Branch("Electron_hitsmiss", Electron_hitsmiss, "Electron_hitsmiss[nElectron]/F");
	T1->Branch("Electron_eoverp",Electron_eoverp,"Electron_eoverp[nElectron]/F");
	T1->Branch("Electron_e_ECAL",Electron_e_ECAL,"Electron_e_ECAL[nElectron]/F");
	T1->Branch("Electron_convVeto", Electron_convVeto, "Electron_convVeto[nElectron]/O");

	T1->Branch("Electron_pfiso_drcor",Electron_pfiso_drcor,"Electron_pfiso_drcor[nElectron]/F");
	T1->Branch("Electron_pfiso_eacor",Electron_pfiso_eacor,"Electron_pfiso_eacor[nElectron]/F");
	T1->Branch("Electron_pfiso04_eacor",Electron_pfiso04_eacor,"Electron_pfiso04_eacor[nElectron]/F");
  
	T1->Branch("Electron_pfRelIso03_all",Electron_pfRelIso03_all,"Electron_pfRelIso03_all[nElectron]/F");
	T1->Branch("Electron_pfRelIso04_all",Electron_pfRelIso04_all,"Electron_pfRelIso04_all[nElectron]/F");
  
  }

  //Seed crystal position needed for EE+ leak problem in 2022: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis#From_E_Gamma
  T1->Branch("Electron_seediEtaOriX",Electron_seediEtaOriX,"Electron_seediEtaOriX[nElectron]/i");
  T1->Branch("Electron_seediPhiOriY",Electron_seediPhiOriY,"Electron_seediPhiOriY[nElectron]/i");
  
  //T1->Branch("Electron_minisoch", Electron_minchiso, "Electron_minchiso[nElectron]/F");
  //T1->Branch("Electron_minisonh", Electron_minnhiso, "Electron_minnhiso[nElectron]/F");
  //T1->Branch("Electron_minisoph", Electron_minphiso, "Electron_minphiso[nElectron]/F");
  T1->Branch("Electron_minisoall", Electron_minisoall, "Electron_minisoall[nElectron]/F");
  
  T1->Branch("Electron_miniPFRelIso_all", Electron_miniPFRelIso_all, "Electron_miniPFRelIso_all[nElectron]/F");
  T1->Branch("Electron_miniPFRelIso_chg", Electron_miniPFRelIso_chg, "Electron_miniPFRelIso_chg[nElectron]/F");
  
  if(store_additional_electron_id_variables){
  
	T1->Branch("Electron_chi",Electron_chi,"Electron_chi[nElectron]/F");
	T1->Branch("Electron_ndf",Electron_ndf,"Electron_ndf[nElectron]/I");
	T1->Branch("Electron_ietaieta",Electron_ietaieta,"Electron_ietaieta[nElectron]/F");
	T1->Branch("Electron_misshits",Electron_misshits,"Electron_misshits[nElectron]/F");
	T1->Branch("Electron_r9full", Electron_r9full, "Electron_r9full[nElectron]/F");
	T1->Branch("Electron_hcaloverecal", Electron_hcaloverecal, "Electron_hcaloverecal[nElectron]/F");
	T1->Branch("Electron_ecloverpout", Electron_ecloverpout, "Electron_ecloverpout[nElectron]/F"); 
	T1->Branch("Electron_pfisolsumphet", Electron_pfisolsumphet, "Electron_pfisolsumphet[nElectron]/F");
	T1->Branch("Electron_pfisolsumchhadpt", Electron_pfisolsumchhadpt, "Electron_pfisolsumchhadpt[nElectron]/F");
	T1->Branch("Electron_pfsiolsumneuhadet", Electron_pfsiolsumneuhadet, "Electron_pfsiolsumneuhadet[nElectron]/F");
  
	T1->Branch("Electron_fbrem",Electron_fbrem,"Electron_fbrem[nElectron]/F");
	T1->Branch("Electron_supcl_etaw", Electron_supcl_etaw, "Electron_supcl_etaw[nElectron]/F");
	T1->Branch("Electron_supcl_phiw", Electron_supcl_phiw, "Electron_supcl_phiw[nElectron]/F");
	T1->Branch("Electron_cloctftrkn", Electron_cloctftrkn, "Electron_cloctftrkn[nElectron]/F");
	T1->Branch("Electron_cloctftrkchi2", Electron_cloctftrkchi2, "Electron_cloctftrkchi2[nElectron]/F");
	T1->Branch("Electron_e1x5bye5x5", Electron_e1x5bye5x5, "Electron_e1x5bye5x5[nElectron]/F");
	T1->Branch("Electron_normchi2", Electron_normchi2, "Electron_normchi2[nElectron]/F");
	T1->Branch("Electron_trkmeasure", Electron_trkmeasure, "Electron_trkmeasure[nElectron]/F");
	T1->Branch("Electron_convtxprob", Electron_convtxprob, "Electron_convtxprob[nElectron]/F");
	T1->Branch("Electron_deltaetacltrkcalo", Electron_deltaetacltrkcalo, "Electron_deltaetacltrkcalo[nElectron]/F");
	T1->Branch("Electron_supcl_preshvsrawe", Electron_supcl_preshvsrawe, "Electron_supcl_preshvsrawe[nElectron]/F");
	T1->Branch("Electron_ecaletrkmomentum", Electron_ecaletrkmomentum, "Electron_ecaletrkmomentum[nElectron]/F");
  
	T1->Branch("Electron_dxy_sv",Electron_dxy_sv,"Electron_dxy_sv[nElectron]/F");
  
  }
  
  if(store_electron_scalnsmear){
	T1->Branch("Electron_eccalTrkEnergyPostCorr",Electron_eccalTrkEnergyPostCorr,"Electron_eccalTrkEnergyPostCorr[nElectron]/F");
	T1->Branch("Electron_energyScaleValue",Electron_energyScaleValue,"Electron_energyScaleValue[nElectron]/F");
	T1->Branch("Electron_energyScaleUp",Electron_energyScaleUp,"Electron_energyScaleUp[nElectron]/F");
	T1->Branch("Electron_energyScaleDown",Electron_energyScaleDown,"Electron_energyScaleDown[nElectron]/F");
	T1->Branch("Electron_energySigmaValue",Electron_energySigmaValue,"Electron_energySigmaValue[nElectron]/F");
	T1->Branch("Electron_energySigmaUp",Electron_energySigmaUp,"Electron_energySigmaUp[nElectron]/F");
	T1->Branch("Electron_energySigmaDown",Electron_energySigmaDown,"Electron_energySigmaDown[nElectron]/F");
  }
  
  }
  
  // Photon Info //
  
  if(store_photons){
  
  T1->Branch("nPhoton",&nPhoton,"nPhoton/I");
  T1->Branch("Photon_e",Photon_e,"Photon_e[nPhoton]/F");
  T1->Branch("Photon_eta",Photon_eta,"Photon_eta[nPhoton]/F");
  T1->Branch("Photon_phi",Photon_phi,"Photon_phi[nPhoton]/F");
  T1->Branch("Photon_mvaid_Fall17V2_raw",Photon_mvaid_Fall17V2_raw,"Photon_mvaid_Fall17V2_raw[nPhoton]/F");
  T1->Branch("Photon_mvaid_RunIIIWinter22V1_WP90",Photon_mvaid_RunIIIWinter22V1_WP90,"Photon_mvaid_RunIIIWinter22V1_WP90[nPhoton]/O");
  T1->Branch("Photon_mvaid_RunIIIWinter22V1_WP80",Photon_mvaid_RunIIIWinter22V1_WP80,"Photon_mvaid_RunIIIWinter22V1_WP80[nPhoton]/O");
  T1->Branch("Photon_mvaid_Fall17V2_WP90",Photon_mvaid_Fall17V2_WP90,"Photon_mvaid_Fall17V2_WP90[nPhoton]/O");
  T1->Branch("Photon_mvaid_Fall17V2_WP80",Photon_mvaid_Fall17V2_WP80,"Photon_mvaid_Fall17V2_WP80[nPhoton]/O");
  T1->Branch("Photon_mvaid_Spring16V1_WP90",Photon_mvaid_Spring16V1_WP90,"Photon_mvaid_Spring16V1_WP90[nPhoton]/O");
  T1->Branch("Photon_mvaid_Spring16V1_WP80",Photon_mvaid_Spring16V1_WP80,"Photon_mvaid_Spring16V1_WP80[nPhoton]/O");
  
  if(store_photon_id_variables){
  
	T1->Branch("Photon_e1by9",Photon_e1by9,"Photon_e1by9[nPhoton]/F");
	T1->Branch("Photon_e9by25",Photon_e9by25,"Photon_e9by25[nPhoton]/F");
	T1->Branch("Photon_trkiso",Photon_trkiso,"Photon_trkiso[nPhoton]/F");
	T1->Branch("Photon_emiso",Photon_emiso,"Photon_emiso[nPhoton]/F");
	T1->Branch("Photon_hadiso",Photon_hadiso,"Photon_hadiso[nPhoton]/F");
	T1->Branch("Photon_chhadiso",Photon_chhadiso,"Photon_chhadiso[nPhoton]/F");
	T1->Branch("Photon_neuhadiso",Photon_neuhadiso,"Photon_neuhadiso[nPhoton]/F");
	T1->Branch("Photon_phoiso",Photon_phoiso,"Photon_phoiso[nPhoton]/F");
	T1->Branch("Photon_PUiso",Photon_PUiso,"Photon_PUiso[nPhoton]/F");
	T1->Branch("Photon_hadbyem",Photon_hadbyem,"Photon_hadbyem[nPhoton]/F");
	T1->Branch("Photon_ietaieta",Photon_ietaieta,"Photon_ietaieta[nPhoton]/F");
  
  }//store_photon_id_variables
  
  }//store_photons

  // Tau Info //
  
  if(store_taus){
  
  T1->Branch("nTau",&nTau,"nTau/I");
  T1->Branch("Tau_isPF",Tau_isPF,"Tau_isPF[nTau]/O");
  T1->Branch("Tau_pt",Tau_pt,"Tau_pt[nTau]/F");
  T1->Branch("Tau_eta",Tau_eta,"Tau_eta[nTau]/F");
  T1->Branch("Tau_phi",Tau_phi,"Tau_phi[nTau]/F");
  T1->Branch("Tau_e",Tau_e,"Tau_e[nTau]/F");
  T1->Branch("Tau_charge",Tau_charge,"Tau_charge[nTau]/I");
  T1->Branch("Tau_dxy",Tau_dxy,"Tau_dxy[nTau]/F");
  T1->Branch("Tau_dz",Tau_dz,"Tau_dz[nTau]/F");
 
  T1->Branch("Tau_jetiso_deeptau2018v2p5_raw",Tau_jetiso_deeptau2018v2p5_raw,"Tau_jetiso_deeptau2018v2p5_raw[nTau]/F");
  T1->Branch("Tau_jetiso_deeptau2018v2p5",Tau_jetiso_deeptau2018v2p5,"Tau_jetiso_deeptau2018v2p5[nTau]/I");
  T1->Branch("Tau_eiso_deeptau2018v2p5_raw",Tau_eiso_deeptau2018v2p5_raw,"Tau_eiso_deeptau2018v2p5_raw[nTau]/F");
  T1->Branch("Tau_eiso_deeptau2018v2p5",Tau_eiso_deeptau2018v2p5,"Tau_eiso_deeptau2018v2p5[nTau]/I");
  T1->Branch("Tau_muiso_deeptau2018v2p5_raw",Tau_muiso_deeptau2018v2p5_raw,"Tau_muiso_deeptau2018v2p5_raw[nTau]/F");
  T1->Branch("Tau_muiso_deeptau2018v2p5",Tau_muiso_deeptau2018v2p5,"Tau_muiso_deeptau2018v2p5[nTau]/I");
  
  if(store_tau_id_variables){
  
	T1->Branch("Tau_jetiso_deeptau2017v2p1_raw",Tau_jetiso_deeptau2017v2p1_raw,"Tau_jetiso_deeptau2017v2p1_raw[nTau]/F");
	T1->Branch("Tau_jetiso_deeptau2017v2p1",Tau_jetiso_deeptau2017v2p1,"Tau_jetiso_deeptau2017v2p1[nTau]/I");
	T1->Branch("Tau_eiso_deeptau2017v2p1_raw",Tau_eiso_deeptau2017v2p1_raw,"Tau_eiso_deeptau2017v2p1_raw[nTau]/F");
	T1->Branch("Tau_eiso_deeptau2017v2p1",Tau_eiso_deeptau2017v2p1,"Tau_eiso_deeptau2017v2p1[nTau]/I");
	T1->Branch("Tau_muiso_deeptau2017v2p1_raw",Tau_muiso_deeptau2017v2p1_raw,"Tau_muiso_deeptau2017v2p1_raw[nTau]/F");
	T1->Branch("Tau_muiso_deeptau2017v2p1",Tau_muiso_deeptau2017v2p1,"Tau_muiso_deeptau2017v2p1[nTau]/I");
  
	T1->Branch("Tau_leadtrkdxy",Tau_leadtrkdxy,"Tau_leadtrkdxy[nTau]/F");
	T1->Branch("Tau_leadtrkdz",Tau_leadtrkdz,"Tau_leadtrkdz[nTau]/F");
	T1->Branch("Tau_leadtrkpt",Tau_leadtrkpt,"Tau_leadtrkpt[nTau]/F");
	T1->Branch("Tau_leadtrketa",Tau_leadtrketa,"Tau_leadtrketa[nTau]/F");
	T1->Branch("Tau_leadtrkphi",Tau_leadtrkphi,"Tau_leadtrkphi[nTau]/F");
	T1->Branch("Tau_decayMode",Tau_decayMode,"Tau_decayMode[nTau]/I");
	T1->Branch("Tau_decayModeinding",Tau_decayModeinding,"Tau_decayModeinding[nTau]/O");
	T1->Branch("Tau_decayModeindingNewDMs",Tau_decayModeindingNewDMs,"Tau_decayModeindingNewDMs[nTau]/O");
	T1->Branch("Tau_eiso2018_raw",Tau_eiso2018_raw,"Tau_eiso2018_raw[nTau]/F");
	T1->Branch("Tau_eiso2018",Tau_eiso2018,"Tau_eiso2018[nTau]/I");
	T1->Branch("Tau_rawiso",Tau_rawiso,"Tau_rawiso[nTau]/F");
	T1->Branch("Tau_rawisodR03",Tau_rawisodR03,"Tau_rawisodR03[nTau]/F");
	T1->Branch("Tau_puCorr",Tau_puCorr,"Tau_puCorr[nTau]/F");
  
  }//store_tau_id_variables
  
  }
  
  // MC Info //
  
  if(isMC){
	  
  // generator-related info //
  
  T1->Branch("Generator_weight", &Generator_weight, "Generator_weight/D") ;
  T1->Branch("Generator_qscale",&Generator_qscale,"Generator_qscale/F");
  T1->Branch("Generator_x1",&Generator_x1,"Generator_x1/F");
  T1->Branch("Generator_x2",&Generator_x2,"Generator_x2/F");
  T1->Branch("Generator_xpdf1",&Generator_xpdf1,"Generator_xpdf1/F");
  T1->Branch("Generator_xpdf2",&Generator_xpdf2,"Generator_xpdf2/F");
  T1->Branch("Generator_id1",&Generator_id1,"Generator_id1/I");
  T1->Branch("Generator_id2",&Generator_id2,"Generator_id2/I");
  T1->Branch("Generator_scalePDF",&Generator_scalePDF,"Generator_scalePDF/F");
  
  T1->Branch("npu_vert",&npu_vert,"npu_vert/I"); //Pileup_nPU
  T1->Branch("npu_vert_true",&npu_vert_true,"npu_vert_true/I"); //Pileup_nTrueInt
	  
  // GEN MET info //    
  
  T1->Branch("GENMET_pt",&genmiset,"genmiset/F") ;
  T1->Branch("GENMET_phi",&genmisphi,"genmisphi/F") ;
  
  // GEN AK8 jet info //  
  
  T1->Branch("nGenJetAK8",&nGenJetAK8, "nGenJetAK8/I");
  T1->Branch("GenJetAK8_pt",GenJetAK8_pt,"GenJetAK8_pt[nGenJetAK8]/F");
  T1->Branch("GenJetAK8_eta",GenJetAK8_eta,"GenJetAK8_eta[nGenJetAK8]/F");
  T1->Branch("GenJetAK8_phi",GenJetAK8_phi,"GenJetAK8_phi[nGenJetAK8]/F");
  T1->Branch("GenJetAK8_mass",GenJetAK8_mass,"GenJetAK8_mass[nGenJetAK8]/F"); 
  T1->Branch("GenJetAK8_msoftdrop",GenJetAK8_sdmass,"GenJetAK8_sdmass[nGenJetAK8]/F");
  T1->Branch("GenJetAK8_hadronflav",GenJetAK8_hadronflav,"GenJetAK8_hadronflav[nGenJetAK8]/I");
  T1->Branch("GenJetAK8_partonflav",GenJetAK8_partonflav,"GenJetAK8_partonflav[nGenJetAK8]/I");

  if(store_fatjet_constituents){
    T1->Branch("nGenJetAK8_cons",&nGenJetAK8_cons,"nGenJetAK8_cons/I");
    T1->Branch("GenJetAK8_cons_pt",GenJetAK8_cons_pt, "GenJetAK8_cons_pt[nGenJetAK8_cons]/F");
    T1->Branch("GenJetAK8_cons_eta",GenJetAK8_cons_eta, "GenJetAK8_cons_eta[nGenJetAK8_cons]/F");
    T1->Branch("GenJetAK8_cons_phi",GenJetAK8_cons_phi, "GenJetAK8_cons_phi[nGenJetAK8_cons]/F");
    T1->Branch("GenJetAK8_cons_mass",GenJetAK8_cons_mass, "GenJetAK8_cons_mass[nGenJetAK8_cons]/F");
    T1->Branch("GenJetAK8_cons_pdgId",GenJetAK8_cons_pdgId, "GenJetAK8_cons_pdgId[nGenJetAK8_cons]/I");
    T1->Branch("GenJetAK8_cons_jetIndex",GenJetAK8_cons_jetIndex, "GenJetAK8_cons_jetIndex[nGenJetAK8_cons]/I");
  }
  
  // GEN AK4 jet info //  
 
  T1->Branch("nGenJetAK4",&nGenJetAK4, "nGenJetAK4/I");
  T1->Branch("GenJetAK4_pt",GenJetAK4_pt,"GenJetAK4_pt[nGenJetAK4]/F");
  T1->Branch("GenJetAK4_eta",GenJetAK4_eta,"GenJetAK4_eta[nGenJetAK4]/F");
  T1->Branch("GenJetAK4_phi",GenJetAK4_phi,"GenJetAK4_phi[nGenJetAK4]/F");
  T1->Branch("GenJetAK4_mass",GenJetAK4_mass,"GenJetAK4_mass[nGenJetAK4]/F");
  T1->Branch("GenJetAK4_hadronflav",GenJetAK4_hadronflav,"GenJetAK4_hadronflav[nGenJetAK4]/I");
  T1->Branch("GenJetAK4_partonflav",GenJetAK4_partonflav,"GenJetAK4_partonflav[nGenJetAK4]/I");
  
  T1->Branch("nGenJetAK4wNu",&nGenJetAK4wNu, "nGenJetAK4wNu/I");
  T1->Branch("GenJetAK4wNu_pt",GenJetAK4wNu_pt,"GenJetAK4wNu_pt[nGenJetAK4wNu]/F");
  T1->Branch("GenJetAK4wNu_eta",GenJetAK4wNu_eta,"GenJetAK4wNu_eta[nGenJetAK4wNu]/F");
  T1->Branch("GenJetAK4wNu_phi",GenJetAK4wNu_phi,"GenJetAK4wNu_phi[nGenJetAK4wNu]/F");
  T1->Branch("GenJetAK4wNu_mass",GenJetAK4wNu_mass,"GenJetAK4wNu_mass[nGenJetAK4wNu]/F");
  T1->Branch("GenJetAK4wNu_hadronflav",GenJetAK4wNu_hadronflav,"GenJetAK4wNu_hadronflav[nGenJetAK4wNu]/I");
  T1->Branch("GenJetAK4wNu_partonflav",GenJetAK4wNu_partonflav,"GenJetAK4wNu_partonflav[nGenJetAK4wNu]/I");
  
  // GEN particles info //  
  
  T1->Branch("nGenPart",&nGenPart, "nGenPart/I");
  T1->Branch("GenPart_pt",GenPart_pt,"GenPart_pt[nGenPart]/F");
  T1->Branch("GenPart_eta",GenPart_eta,"GenPart_eta[nGenPart]/F");
  T1->Branch("GenPart_phi",GenPart_phi,"GenPart_phi[nGenPart]/F");
  T1->Branch("GenPart_mass",GenPart_mass,"GenPart_mass[nGenPart]/F");
  T1->Branch("GenPart_status",GenPart_status,"GenPart_status[nGenPart]/I");
  T1->Branch("GenPart_pdgId",GenPart_pdg,"GenPart_pdg[nGenPart]/I");
  T1->Branch("GenPart_mompdgId",GenPart_mompdg,"GenPart_mompdg[nGenPart]/I");
  T1->Branch("GenPart_momstatus",GenPart_momstatus,"GenPart_momstatus[nGenPart]/I");
  T1->Branch("GenPart_grmompdgId",GenPart_grmompdg,"GenPart_grmompdg[nGenPart]/I");
  T1->Branch("GenPart_daugno",GenPart_daugno,"GenPart_daugno[nGenPart]/I");
  T1->Branch("GenPart_fromhard",GenPart_fromhard,"GenPart_fromhard[nGenPart]/O");
  T1->Branch("GenPart_fromhardbFSR",GenPart_fromhardbFSR,"GenPart_fromhardbFSR[nGenPart]/O");
  T1->Branch("GenPart_isPromptFinalState",GenPart_isPromptFinalState,"GenPart_isPromptFinalState[nGenPart]/O");
  T1->Branch("GenPart_isLastCopyBeforeFSR",GenPart_isLastCopyBeforeFSR,"GenPart_isLastCopyBeforeFSR[nGenPart]/O");
  T1->Branch("GenPart_isDirectPromptTauDecayProductFinalState",GenPart_isDirectPromptTauDecayProductFinalState,"GenPart_isDirectPromptTauDecayProductFinalState[nGenPart]/O");
  
  
  // LHE Info //
  
  T1->Branch("nLHEPart",&nLHEPart, "nLHEPart/I");
  T1->Branch("LHEPart_pdg",LHEPart_pdg,"LHEPart_pdg[nLHEPart]/I");
  T1->Branch("LHEPart_pt",LHEPart_pt,"LHEPart_pt[nLHEPart]/F");
  T1->Branch("LHEPart_eta",LHEPart_eta,"LHEPart_eta[nLHEPart]/F");
  T1->Branch("LHEPart_phi",LHEPart_phi,"LHEPart_phi[nLHEPart]/F");
  T1->Branch("LHEPart_m",LHEPart_m,"LHEPart_m[nLHEPart]/F");
  
  T1->Branch("LHE_weight",&LHE_weight, "LHE_weight/D");
  // scale & pdf weights (correct only for ttbar POWHEG samples))
  T1->Branch("nLHEScaleWeights",&nLHEScaleWeights, "nLHEScaleWeights/I");
  T1->Branch("LHEScaleWeights",LHEScaleWeights,"LHEScaleWeights[nLHEScaleWeights]/F");
  T1->Branch("nLHEPDFWeights",&nLHEPDFWeights, "nLHEPDFWeights/I");
  T1->Branch("LHEPDFWeights",LHEPDFWeights,"LHEPDFWeights[nLHEPDFWeights]/F");
  T1->Branch("nLHEPSWeights",&nLHEPSWeights, "nLHEPSWeights/I");
  T1->Branch("LHEPSWeights",LHEPSWeights,"LHEPSWeights[nLHEPSWeights]/F");
  // storing all weights (up to 250) //
  T1->Branch("nLHEWeights",&nLHEWeights, "nLHEWeights/I");
  //T1->Branch("LHEWeights",LHEWeights,"LHEWeights[nLHEWeights]/F");
  T1->Branch("LHEWeights","std::vector<float>",&LHEWeights);
  
  } //isMC
  
  T2 = new TTree("Events_All", "XtoYH");
  
  T2->Branch("ievt", &ievt, "ievt/i");
  T2->Branch("PV_npvsGood", &PV_npvsGood, "PV_npvsGood/I");
  
  if(isMC){
	  
  T2->Branch("Generator_weight", &Generator_weight, "Generator_weight/D") ;
  T2->Branch("npu_vert",&npu_vert,"npu_vert/I");
  T2->Branch("npu_vert_true",&npu_vert_true,"npu_vert_true/I");
  T2->Branch("LHE_weight",&LHE_weight, "LHE_weight/D");
  
  }
  
  Nevt=0;
}


Leptop::~Leptop()
{
 
  // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
Leptop::analyze(const edm::Event& iEvent, const edm::EventSetup& pset) {
	
  logMemoryUsage("Before processing event");	
    
  using namespace edm;
  Nevt++;
  
  InitializeBranches();
  
  irun = iEvent.id().run();
  ilumi = iEvent.luminosityBlock();
  ievt = iEvent.id().event();
  
  if (Nevt%100==1)cout <<"Leptop::analyze "<<Nevt<<" "<<iEvent.id().run()<<" "<<iEvent.id().event()<<endl;
  
  if (Nevt%100==1){
	std::time_t current_time = std::time(nullptr);
	std::cout << "Current time: " << std::ctime(&current_time);   
  }
    
  // First store all MC information //
  
  edm::Handle<reco::GenJetCollection> genjetAK8s;
  edm::Handle<reco::GenJetCollection> genjetAK4s;
  edm::Handle<reco::GenJetCollection> genjetAK4swNu;
  
  if (Nevt==1) { cout<<"YEAR: "<<year<<" isRun3: "<<isRun3<<endl; }
  
  if(isMC){
	
	
	edm::Handle<GenEventInfoProduct>eventinfo;  
	iEvent.getByToken(tok_wt_,eventinfo);
	
	edm::Handle<LHEEventProduct>lheeventinfo ;
    iEvent.getByToken(lheEventProductToken_,lheeventinfo) ;
     
	edm::Handle<std::vector<reco::GenParticle>> genparticles;
	iEvent.getByToken(tok_genparticles_,genparticles);
	
	iEvent.getByToken(tok_genjetAK8s_, genjetAK8s);
	iEvent.getByToken(tok_genjetAK4s_, genjetAK4s);
	iEvent.getByToken(tok_genjetAK4swNu_, genjetAK4swNu);
	
	edm::Handle<reco::JetFlavourInfoMatchingCollection> jetFlavourInfos;
    iEvent.getByToken(jetFlavourInfosToken_, jetFlavourInfos);

	edm::Handle<std::vector<PileupSummaryInfo> > PupInfo;  
	iEvent.getByToken(pileup_, PupInfo);
	
	// MC weights  & collision info //
        
    nLHEPSWeights = 0;
    
    if (eventinfo.isValid()){
		
		Generator_weight = eventinfo->weight();
		
		// Generator information //
		
		Generator_qscale = eventinfo->qScale();
		Generator_x1 = (*eventinfo->pdf()).x.first;
        Generator_x2 = (*eventinfo->pdf()).x.second;
        Generator_id1 = (*eventinfo->pdf()).id.first;
        Generator_id2 = (*eventinfo->pdf()).id.second;
        Generator_xpdf1 = (*eventinfo->pdf()).xPDF.first;
        Generator_xpdf2 = (*eventinfo->pdf()).xPDF.second;
        Generator_scalePDF = (*eventinfo->pdf()).scalePDF;
        
        //cout<<"eventinfo->weights().size() "<<eventinfo->weights().size()<<" GEN weight "<<Generator_weight<<endl;
        
        // Parton shower weights //
        
        if(eventinfo->weights().size()>2){
			for(unsigned int i=2; i<eventinfo->weights().size(); ++i){
				LHEPSWeights[nLHEPSWeights] = eventinfo->weights()[i]/eventinfo->weights()[1];
				nLHEPSWeights++;
				if(nLHEPSWeights >= nlhepsmax) break;
			}
		}
        
    }
          
    // Pileup information //
    
    npu_vert = 0;
	npu_vert_true = 0;
    
    if (PupInfo.isValid()) {
      std::vector<PileupSummaryInfo>::const_iterator PVI;
      for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
		if (PVI->getBunchCrossing()==0) {
			npu_vert = PVI->getPU_NumInteractions();
			npu_vert_true = PVI->getTrueNumInteractions();
			break;
			}
		}
    }
      
    
    // LHE-level particles //
          
    nLHEWeights = 0;
    nLHEScaleWeights = 0;
	nLHEPDFWeights = 0;
	    
    nLHEPart = 0;
    
    if(lheeventinfo.isValid()){
      
      // LHE-level particles //
      
      const auto & hepeup = lheeventinfo->hepeup();
      const auto & pup = hepeup.PUP;
      
      for (unsigned int ij = 0; ij  < pup.size(); ++ij) {
		if(hepeup.ISTUP[ij]==1){// status==1 --> particles which stay up to final state                          
			TLorentzVector p4(pup[ij][0], pup[ij][1], pup[ij][2], pup[ij][3]);
			LHEPart_pt[nLHEPart] = p4.Pt();
			LHEPart_eta[nLHEPart] = p4.Eta();
			LHEPart_phi[nLHEPart] = p4.Phi();
			LHEPart_m[nLHEPart] = p4.M();
			LHEPart_pdg[nLHEPart] = (hepeup.IDUP[ij]);
			nLHEPart++;
			if(nLHEPart>=nlhemax) break;
			}
		}
			
	 // LHE-level weights //
	
	  LHE_weight = lheeventinfo->originalXWGTUP();
	  	 
	  //cout<<"PRINTING all theory weights\n";
 
	  for ( unsigned int index = 0; index < lheeventinfo->weights().size(); ++index ) {	
		 
		 //cout<<"Index "<<index+1<<" Id "<<lheeventinfo->weights()[index].id<<" weight "<<lheeventinfo->weights()[index].wgt/lheeventinfo->originalXWGTUP()<<endl;//" muR "<<lheeventinfo->weights()[index].MUR<<" muF "<<lheeventinfo->weights()[index].MUF<<" DYN Scale "<<lheeventinfo->weights()[index].DYN_SCALE<<endl;
		
		// storing up to a maximum number of weights //
		
		if(nLHEWeights<nlheweightmax){
		//	LHEWeights[nLHEWeights] = lheeventinfo->weights()[index].wgt/lheeventinfo->originalXWGTUP();
			LHEWeights.push_back(lheeventinfo->weights()[index].wgt/lheeventinfo->originalXWGTUP());
			nLHEWeights++;
		}
		
		// the convention of storing LHE Scale, PDF, and AlphaS weights is valid only for ttbar POWHEG samples //
		
		if(index<nlhescalemax && nLHEScaleWeights<nlhescalemax){
			LHEScaleWeights[nLHEScaleWeights] = lheeventinfo->weights()[index].wgt/lheeventinfo->originalXWGTUP();
			nLHEScaleWeights++;
		}
		if(index>=nlhescalemax && index<(nlhescalemax+nPDFsets)  && nLHEPDFWeights<nlhepdfmax){
			LHEPDFWeights[nLHEPDFWeights] = lheeventinfo->weights()[index].wgt/lheeventinfo->originalXWGTUP();
			nLHEPDFWeights++;
		}
				
	  }//index
	  	  		
	}// lheeventinfo.isValid()

    // Flavor tagging of GEN jets using ghost-matching //     
                                                            
	// AK8 GEN jet //
	
    nGenJetAK8 = 0;
    nGenJetAK8_cons = 0;
    
    if(genjetAK8s.isValid()){
		
		std::vector<int> partonFlavour_AK8;
		std::vector<int> hadronFlavour_AK8;

		for (const reco::GenJet & jet : *genjetAK8s) {
			
			bool matched = false;
			
			for (const reco::JetFlavourInfoMatching & jetFlavourInfoMatching : *jetFlavourInfos) {
				if (deltaR(jet.p4(), jetFlavourInfoMatching.first->p4()) < 0.1) {
					partonFlavour_AK8.push_back(jetFlavourInfoMatching.second.getPartonFlavour());
					hadronFlavour_AK8.push_back(jetFlavourInfoMatching.second.getHadronFlavour());
					matched = true;
					break;
				}
			}
		    
			if (!matched) {
				partonFlavour_AK8.push_back(-100);
				hadronFlavour_AK8.push_back(-100);
			}
		}
      
        JetDefinition pfjetAK8_Def(antikt_algorithm,0.8,E_scheme);
		SoftDrop sd(beta,z_cut,0.8);
      
		for(unsigned gjet = 0; gjet<genjetAK8s->size(); gjet++)	{
	
			TLorentzVector genjetAK8_4v((*genjetAK8s)[gjet].px(),(*genjetAK8s)[gjet].py(),(*genjetAK8s)[gjet].pz(), (*genjetAK8s)[gjet].energy());
			if(genjetAK8_4v.Pt()<min_pt_AK8GENjet) continue;
			if(abs(genjetAK8_4v.Eta())>max_eta_GENjet) continue;
	
			GenJetAK8_pt[nGenJetAK8] = genjetAK8_4v.Pt();
			GenJetAK8_eta[nGenJetAK8] = genjetAK8_4v.Eta();
			GenJetAK8_phi[nGenJetAK8] = genjetAK8_4v.Phi();
			GenJetAK8_mass[nGenJetAK8] = (*genjetAK8s)[gjet].mass();
			GenJetAK8_hadronflav[nGenJetAK8] = (int)hadronFlavour_AK8[gjet];
			GenJetAK8_partonflav[nGenJetAK8] = partonFlavour_AK8[gjet];
			
			std::vector<reco::CandidatePtr> daught((*genjetAK8s)[gjet].daughterPtrVector());
	
			vector <fastjet::PseudoJet> fjInputs;
			fjInputs.resize(0);
			for (unsigned int i2 = 0; i2< daught.size(); ++i2) {
				PseudoJet psjet = PseudoJet( (*daught[i2]).px(),(*daught[i2]).py(),(*daught[i2]).pz(),(*daught[i2]).energy() );
				psjet.set_user_index(i2);
				fjInputs.push_back(psjet); 
        
				// Storing 4-momenta of jet constituents//
				if(store_fatjet_constituents && nGenJetAK8<njetconsmax){
					GenJetAK8_cons_pt[nGenJetAK8_cons] = daught[i2]->pt();
					GenJetAK8_cons_eta[nGenJetAK8_cons] = daught[i2]->eta();
					GenJetAK8_cons_phi[nGenJetAK8_cons] = daught[i2]->phi();
					GenJetAK8_cons_mass[nGenJetAK8_cons] = daught[i2]->mass();
					GenJetAK8_cons_pdgId[nGenJetAK8_cons] = daught[i2]->pdgId();
					GenJetAK8_cons_jetIndex[nGenJetAK8_cons] = nGenJetAK8;   
					nGenJetAK8_cons++;
				}
				// end of candidate storage //
        
			} //i2
	
			// Running soft-drop //
	
			vector <fastjet::PseudoJet> sortedJets;
			fastjet::ClusterSequence clustSeq(fjInputs, pfjetAK8_Def);
			fjInputs.clear();
			sortedJets    = sorted_by_pt(clustSeq.inclusive_jets());
	
			if(sortedJets.size()>0){
				GenJetAK8_sdmass[nGenJetAK8] = (sd(sortedJets[0])).m();
			}
			sortedJets.clear();
	
			if (++nGenJetAK8>=njetmxAK8) break;
	
		} // loop over AK8 GEN jets 
		
	}//genjetAK8s.isValid()
      
    // AK4 GEN jet //
      
	nGenJetAK4 = 0;
	
	if(genjetAK4s.isValid()){
		
		std::vector<int> partonFlavour_AK4;
		std::vector<int> hadronFlavour_AK4;
      
		for (const reco::GenJet & jet : *genjetAK4s) {
		  
			bool matched = false;
			
			for (const reco::JetFlavourInfoMatching & jetFlavourInfoMatching : *jetFlavourInfos) {
				if (deltaR(jet.p4(), jetFlavourInfoMatching.first->p4()) < 0.1) {
					partonFlavour_AK4.push_back(jetFlavourInfoMatching.second.getPartonFlavour());
					hadronFlavour_AK4.push_back(jetFlavourInfoMatching.second.getHadronFlavour());
					matched = true;
					break;
				}
			}
		    
			if (!matched) {
				partonFlavour_AK4.push_back(-100);
				hadronFlavour_AK4.push_back(-100);
			}	
		}
	
		for(unsigned gjet = 0; gjet<genjetAK4s->size(); gjet++)	{
	
			TLorentzVector genjetAK44v((*genjetAK4s)[gjet].px(),(*genjetAK4s)[gjet].py(),(*genjetAK4s)[gjet].pz(), (*genjetAK4s)[gjet].energy());
			if(genjetAK44v.Pt()<min_pt_GENjet) continue;
			if(abs(genjetAK44v.Eta())>max_eta_GENjet) continue;
	
			GenJetAK4_pt[nGenJetAK4] = genjetAK44v.Pt();
			GenJetAK4_eta[nGenJetAK4] = genjetAK44v.Eta();
			GenJetAK4_phi[nGenJetAK4] = genjetAK44v.Phi();
			GenJetAK4_mass[nGenJetAK4] = (*genjetAK4s)[gjet].mass();
			GenJetAK4_hadronflav[nGenJetAK4] = (int)hadronFlavour_AK4[gjet];
			GenJetAK4_partonflav[nGenJetAK4] = partonFlavour_AK4[gjet];

			if (++nGenJetAK4>=njetmx) break;
      
		}
		
    } //genjetAK4s.isValid()
        
    // AK4 GEN jet with neutrinos //
    
    nGenJetAK4wNu = 0;
	
	if(genjetAK4swNu.isValid()){
		
		std::vector<int> partonFlavour_AK4;
		std::vector<int> hadronFlavour_AK4;
      
		for (const reco::GenJet & jet : *genjetAK4swNu) {
		  
			bool matched = false;
			
			for (const reco::JetFlavourInfoMatching & jetFlavourInfoMatching : *jetFlavourInfos) {
				if (deltaR(jet.p4(), jetFlavourInfoMatching.first->p4()) < 0.1) {
					partonFlavour_AK4.push_back(jetFlavourInfoMatching.second.getPartonFlavour());
					hadronFlavour_AK4.push_back(jetFlavourInfoMatching.second.getHadronFlavour());
					matched = true;
					break;
				}
			}
		    
			if (!matched) {
				partonFlavour_AK4.push_back(-100);
				hadronFlavour_AK4.push_back(-100);
			}	
		}
	
		for(unsigned gjet = 0; gjet<genjetAK4swNu->size(); gjet++)	{
	
			TLorentzVector genjetAK44v((*genjetAK4swNu)[gjet].px(),(*genjetAK4swNu)[gjet].py(),(*genjetAK4swNu)[gjet].pz(), (*genjetAK4swNu)[gjet].energy());
			if(genjetAK44v.Pt()<min_pt_GENjet) continue;
			if(abs(genjetAK44v.Eta())>max_eta_GENjet) continue;
	
			GenJetAK4wNu_pt[nGenJetAK4wNu] = genjetAK44v.Pt();
			GenJetAK4wNu_eta[nGenJetAK4wNu] = genjetAK44v.Eta();
			GenJetAK4wNu_phi[nGenJetAK4wNu] = genjetAK44v.Phi();
			GenJetAK4wNu_mass[nGenJetAK4wNu] = (*genjetAK4swNu)[gjet].mass();
			GenJetAK4wNu_hadronflav[nGenJetAK4wNu] = (int)hadronFlavour_AK4[gjet];
			GenJetAK4wNu_partonflav[nGenJetAK4wNu] = partonFlavour_AK4[gjet];

			if (++nGenJetAK4wNu>=njetmx) break;
      
		}
		
    } // genjetAK4swNu.isValid()
    
    // Gen particles //
        
	nGenPart = 0;
	
	if(genparticles.isValid()){
	
		for(unsigned ig=0; ig<(genparticles->size()); ig++){
			
			if(!(((*genparticles)[ig].status()==1)||(abs((*genparticles)[ig].status())==22)||((*genparticles)[ig].status()==23)|((*genparticles)[ig].status()==2))) continue;
			if(!((*genparticles)[ig].isHardProcess()||(*genparticles)[ig].fromHardProcessBeforeFSR()||(*genparticles)[ig].isLastCopyBeforeFSR()||(*genparticles)[ig].isDirectPromptTauDecayProductFinalState())) continue;
	  
			if(!((abs((*genparticles)[ig].pdgId())>=1 && abs((*genparticles)[ig].pdgId())<=6) || (abs((*genparticles)[ig].pdgId())>=11 && abs((*genparticles)[ig].pdgId())<=16) || (abs((*genparticles)[ig].pdgId())>=22 && abs((*genparticles)[ig].pdgId())<=24))) continue;
			// important condition on pdg id -> May be changed in other analyses //
	  
			GenPart_status[nGenPart] = (*genparticles)[ig].status();
			GenPart_pdg[nGenPart] = (*genparticles)[ig].pdgId();
			GenPart_daugno[nGenPart] = (*genparticles)[ig].numberOfDaughters();
			GenPart_fromhard[nGenPart] = (*genparticles)[ig].isHardProcess();
			GenPart_fromhardbFSR[nGenPart] = (*genparticles)[ig].fromHardProcessBeforeFSR();
			GenPart_isLastCopyBeforeFSR[nGenPart] = (*genparticles)[ig].isLastCopyBeforeFSR();
			GenPart_isPromptFinalState[nGenPart] = (*genparticles)[ig].isPromptFinalState();
			GenPart_isDirectPromptTauDecayProductFinalState[nGenPart] = (*genparticles)[ig].isDirectPromptTauDecayProductFinalState();
			GenPart_pt[nGenPart] = (*genparticles)[ig].pt();
			GenPart_eta[nGenPart] = (*genparticles)[ig].eta();
			GenPart_phi[nGenPart] = (*genparticles)[ig].phi();
			GenPart_mass[nGenPart] = (*genparticles)[ig].mass();
			
			int mompdg, momstatus, grmompdg;
			mompdg = momstatus = grmompdg = 0;
			
			if((*genparticles)[ig].numberOfMothers()>0){
				
				// mother pdg id & status //
			
				const Candidate * mom = (*genparticles)[ig].mother();
				
				while(mom->pdgId() == (*genparticles)[ig].pdgId())
				{
					mom = mom->mother();
				}
				
				if(!(*genparticles)[ig].isPromptFinalState() && !(*genparticles)[ig].isDirectPromptTauDecayProductFinalState()){
					while(mom->status()==2){
						mom = mom->mother();	
					}
				}
				
				mompdg = mom->pdgId();
				momstatus = mom->status();
	  
				// grand-mother pdg id //
						
				if(mom->numberOfMothers()>0){
	
					const Candidate * grmom = mom->mother();
					
					while(grmom->pdgId() == mompdg)
					{
						if(grmom->numberOfMothers()>0){
							grmom = grmom->mother();
						}
						else{ break; }
					}
					
					grmompdg  = grmom->pdgId();	
				} 
				
			}
			
			GenPart_mompdg[nGenPart] = mompdg;
			GenPart_momstatus[nGenPart] = momstatus;
			GenPart_grmompdg[nGenPart] = grmompdg; 
			
			nGenPart++;
			if(nGenPart>=npartmx) break;
		}
   
    }//genparticles.isValid()
        
  }//isMC
    
  // =========== Vertices & global event properties ================= //
  
  // Define vertex collections & initalize //
  
  Handle<VertexCollection> primaryVertices;
  iEvent.getByToken(tok_primaryVertices_, primaryVertices);
  
  edm::Handle<reco::VertexCompositePtrCandidateCollection> secondaryVertices;
  iEvent.getByToken(tok_sv,secondaryVertices);
  
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_beamspot_, beamSpotH);  //Label("offlineBeamSpot",beamSpotH);
  
  // Primary vertex info //
  
  reco::Vertex vertex;
  
  //const auto& pvsScoreProd = iEvent.get(pvsScore_);
  
  if (primaryVertices.isValid()) {
	  
	if(primaryVertices->size() > 0){  
		vertex = primaryVertices->at(0); 
		PV_ndof = vertex.ndof();
		PV_chi2 = vertex.normalizedChi2();
		PV_x = vertex.position().x();
		PV_y = vertex.position().y();
		PV_z = vertex.position().z();
		//PV_score = pvsScoreProd.get(primaryVertices.id(), 0);
	} 
	 
    int ndofct_org=0;
    int nchict_org=0;
    int nvert_org = 0;
    int nprimi_org = 0;
    
    for (reco::VertexCollection::const_iterator vert=primaryVertices->begin(); vert<primaryVertices->end(); vert++) {
      nvert_org++;
      if (vert->isValid() && !vert->isFake()) {
		if (vert->ndof() > 4 && fabs(vert->position().z()) <= 24 && fabs(vert->position().Rho()) <= 2) {
			nprimi_org++;
			}
		if (vert->ndof()>7) {
			ndofct_org++;
			if (vert->normalizedChi2()<5) nchict_org++;
			}
		}
    }
    
    nprim = min(999,nvert_org) + 1000*min(999,ndofct_org) + 1000000*min(999,nchict_org);
    npvert = nchict_org;
    PV_npvsGood = nprimi_org;
    
  }
 
  reco::TrackBase::Point beamPoint(0,0, 0);
  
  if (beamSpotH.isValid()){
    beamPoint = beamSpotH->position();
  }

  // Energy density info //
  
  edm::Handle<double> Rho_PF;
  iEvent.getByToken(tok_Rho_,Rho_PF);
  Rho = *Rho_PF;
    
  // =========== Trigger information ================= //
  
  bool booltrg[nHLTmx+1]= {false};
 
  if(!isFastSIM){
    
	edm::Handle<edm::TriggerResults> trigRes;
	iEvent.getByToken(triggerBits_, trigRes);
  
	const edm::TriggerNames &names_ = iEvent.triggerNames(*trigRes);
  
	edm::Handle<pat::TriggerObjectStandAloneCollection> triggerObjects;
	iEvent.getByToken(triggerObjects_, triggerObjects);
  
	edm::Handle<pat::PackedTriggerPrescales> triggerPrescales;
	iEvent.getByToken(triggerPrescales_, triggerPrescales);
  	
	const char* variab_trig;
		
	for (int jk=0; jk<nHLTmx; jk++) {
		for(unsigned ij = 0; ij<trigRes->size(); ++ij) {
			std::string name = names_.triggerName(ij);
			variab_trig = name.c_str(); 
			if (strstr(variab_trig,hlt_name[jk]) && ((strlen(variab_trig)-strlen(hlt_name[jk]))<5))
			{
				if ((trigRes->accept(ij))){   //||(isMC)) {
					booltrg[jk] = true; booltrg[nHLTmx] = true;
					break;
				}
			}
		}//ij     
    }//jk
	
		
	//trig_value = 1; 
	//for (int jk=1; jk<(nHLTmx+1); jk++) {  if(booltrg[nHLTmx-jk]) {  trig_value+=(1<<jk); } }
		
	for (int jk=0; jk<(nHLTmx); jk++) {
		trig_bits.push_back(booltrg[jk]);
		trig_paths.push_back(string(hlt_name[jk]));
	}
	    
  // Trigger objects //
    
	vector<triggervar> alltrgobj;
	if (trigRes.isValid()) { 
    
		const char* variab2 ;
    
		alltrgobj.clear(); 
    
		for (pat::TriggerObjectStandAlone obj : *triggerObjects) {
      
			obj.unpackPathNames(names_);
			std::vector<std::string> pathNamesAll  = obj.pathNames(false);
      
			for (unsigned ih = 0, n = pathNamesAll.size(); ih < n; ih++) {
	
				variab2 = pathNamesAll[ih].c_str(); 
	
				for (int jk=0; jk<nHLTmx; jk++) {
					if (strstr(variab2,hlt_name[jk]) && (strlen(variab2)-strlen(hlt_name[jk])<5)) {
	    	    
						if(obj.pt()>20 && fabs(obj.eta())<3.0) {
	      
							triggervar tmpvec1;
	      
							tmpvec1.both = obj.hasPathName( pathNamesAll[ih], true, true );
							tmpvec1.highl  = obj.hasPathName( pathNamesAll[ih], false, true );
							tmpvec1.level1 = obj.hasPathName( pathNamesAll[ih], true, false );
							tmpvec1.trg4v = TLorentzVector(obj.px(), obj.py(), obj.pz(), obj.energy());
							tmpvec1.pdgId = obj.pdgId();
							tmpvec1.prescl = 1;    //triggerPrescales->getPrescaleForIndex(ih);
							tmpvec1.ihlt = jk;
							tmpvec1.hltname = string(hlt_name[jk]);
							tmpvec1.type = (
										1*(obj.type(92)) + 
										2*(obj.coll("hltEgammaCandidates")) + 
										4*obj.type(83) + 
										8*(obj.coll("hltIterL3MuonCandidates")) + 
										16*obj.type(84) + 
										32*(obj.coll("*Tau*")) +
										64*obj.type(87) + 
										128*(obj.coll("L1ETM")) +
										256*obj.type(85) +
										512*obj.coll("hltGtStage2Digis:Jet:HLT") +   //L1 jet 
										1024*obj.coll("hltAK4CaloJetsCorrectedIDPassed:HLT") +  //Calo Jet
										2048*obj.coll("hltAK4PixelOnlyPFJetsTightIDCorrected::HLT") + //Pixel Jet
										4096*obj.coll("hltAK4PFJetsTightIDCorrected::HLT") + //L3-HLT
										8192*obj.coll("hltAK8CaloJetsCorrectedIDPassed::HLT") + //AK8 Calo Jet
										16384*obj.coll("hltAK8PFJets250SoftDropMass40::HLT")  //AK8 Calo Jet
									
										);
										//order: e/gamma + muon + tau + met + jet + L1 jet + Calo jet + Pixel jet+ AK8Calo jet
							alltrgobj.push_back(tmpvec1);
							break;
						}
					}
				}//jk 
			}//ih
		}
	}
	    
	int xht=0;
	nTrigObj = alltrgobj.size();
	if(nTrigObj>njetmx) { nTrigObj = njetmx; }
	if(nTrigObj > 0){
		for(unsigned int iht=0; iht<(unsigned int)nTrigObj; iht++){
			if(alltrgobj[iht].trg4v.Pt()>20 && fabs(alltrgobj[iht].trg4v.Eta())<3.0) {
				TrigObj_pt[xht] = alltrgobj[iht].trg4v.Pt();
				TrigObj_eta[xht] = alltrgobj[iht].trg4v.Eta();
				TrigObj_phi[xht] = alltrgobj[iht].trg4v.Phi();
				TrigObj_mass[xht] = alltrgobj[iht].trg4v.M();
				TrigObj_HLT[xht] = alltrgobj[iht].highl;
				TrigObj_L1[xht] = alltrgobj[iht].level1;
				TrigObj_Both[xht] = alltrgobj[iht].both;
				TrigObj_Ihlt[xht] = alltrgobj[iht].ihlt;
				TrigObj_HLTname.push_back(alltrgobj[iht].hltname);
				TrigObj_pdgId[xht] = alltrgobj[iht].pdgId;
				TrigObj_type[xht] = alltrgobj[iht].type;
				xht++;
				if(xht>=njetmx) break;
			}
		//if(iht == (njetmx-1)) break;
		}
	}
    
  } // !isFastSIM
  
   // fill trigger booleans //
 
  for(int jk=0; jk<nHLTmx; jk++) {
	  // SingleMuon triggers
	  if(jk==0) 	  {  hlt_IsoMu24 = booltrg[jk]; }
	  else if(jk==1)  {  hlt_IsoTkMu24 = booltrg[jk]; }
	  else if(jk==2)  {  hlt_IsoMu27 = booltrg[jk]; }
	  else if(jk==3)  {  hlt_Mu50 = booltrg[jk]; }
	  else if(jk==4)  {  hlt_TkMu50 = booltrg[jk]; }
	  else if(jk==5)  {  hlt_TkMu100 = booltrg[jk]; }
	  else if(jk==6)  {  hlt_OldMu100 = booltrg[jk]; }
	  else if(jk==7)  {  hlt_HighPtTkMu100 = booltrg[jk]; }
	  else if(jk==8)  {  hlt_CascadeMu100 = booltrg[jk]; }
	  // Single Electron triggers
	  else if(jk==9)  {  hlt_Ele27_WPTight_Gsf = booltrg[jk]; }
	  else if(jk==10) {  hlt_Ele30_WPTight_Gsf = booltrg[jk]; }
	  else if(jk==11) {  hlt_Ele32_WPTight_Gsf = booltrg[jk]; }
	  else if(jk==12) {  hlt_Ele35_WPTight_Gsf = booltrg[jk]; }
	  else if(jk==13) {  hlt_Ele28_eta2p1_WPTight_Gsf_HT150 = booltrg[jk]; }
	  else if(jk==14) {  hlt_Ele32_WPTight_Gsf_L1DoubleEG = booltrg[jk]; }
	  else if(jk==15) {  hlt_Ele50_CaloIdVT_GsfTrkIdT_PFJet165 = booltrg[jk]; }
	  else if(jk==16) {  hlt_Ele115_CaloIdVT_GsfTrkIdT = booltrg[jk]; }
	  // Double Muon triggers
	  else if(jk==17) {  hlt_Mu37_TkMu27 = booltrg[jk]; }
	  else if(jk==18) {  hlt_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL = booltrg[jk]; }
	  else if(jk==19) {  hlt_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ = booltrg[jk]; }
	  else if(jk==20) {  hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL = booltrg[jk]; }
      else if(jk==21) {  hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ = booltrg[jk]; }
	  else if(jk==22) {  hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 = booltrg[jk]; }
	  else if(jk==23) {  hlt_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8 = booltrg[jk]; }
	  // Double electron triggers
	  else if(jk==24) {  hlt_DoubleEle25_CaloIdL_MW = booltrg[jk]; }
	  else if(jk==25) {  hlt_DoubleEle33_CaloIdL_MW = booltrg[jk]; }
	  else if(jk==26) {  hlt_DoubleEle33_CaloIdL_GsfTrkIdVL = booltrg[jk]; }
	  else if(jk==27) {  hlt_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL = booltrg[jk]; }
      else if(jk==28) {  hlt_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ   = booltrg[jk]; }
      // EMu cross trigggers 
      else if(jk==29) {  hlt_Mu37_Ele27_CaloIdL_MW   = booltrg[jk]; }
      else if(jk==30) {  hlt_Mu27_Ele37_CaloIdL_MW   = booltrg[jk]; }
      else if(jk==31) {  hlt_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL   = booltrg[jk]; }
      else if(jk==32) {  hlt_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ   = booltrg[jk]; }
      else if(jk==33) {  hlt_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL   = booltrg[jk]; }
      else if(jk==34) {  hlt_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ   = booltrg[jk]; }
      // JetHT triggers
      else if(jk==35) {  hlt_PFHT800   = booltrg[jk]; }
      else if(jk==36) {  hlt_PFHT900   = booltrg[jk]; }
      else if(jk==37) {  hlt_PFHT1050   = booltrg[jk]; }
      // AK4 triggers
      else if(jk==38) {  hlt_PFJet450   = booltrg[jk]; }
      else if(jk==39) {  hlt_PFJet500   = booltrg[jk]; }
      // AK8 triggers
      else if(jk==40) {  hlt_AK8PFJet450   = booltrg[jk]; }
      else if(jk==41) {  hlt_AK8PFJet500   = booltrg[jk]; }
      else if(jk==42) {  hlt_AK8PFJet400_TrimMass30 = booltrg[jk]; }
      else if(jk==43) {  hlt_AK8PFHT800_TrimMass50 = booltrg[jk]; }
      // AK8 triggers in Run3
      else if(jk==44) {  hlt_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35 = booltrg[jk]; }
      else if(jk==45) {  hlt_AK8PFJet220_SoftDropMass40_PNetBB0p35_DoubleAK4PFJet60_30_PNet2BTagMean0p50 = booltrg[jk]; }
      else if(jk==46) {  hlt_AK8PFJet425_SoftDropMass40 = booltrg[jk]; }
      else if(jk==47) {  hlt_AK8PFJet420_MassSD30 = booltrg[jk]; }
      // 4b triggers in Run3 
      else if(jk==48) {  hlt_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65 = booltrg[jk]; }
      else if(jk==49) {  hlt_QuadPFJet70_50_40_35_PNet2BTagMean0p65 = booltrg[jk]; }
      else if(jk==50) {  hlt_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70 = booltrg[jk]; }
      else if(jk==51) {  hlt_PFHT280_QuadPFJet30_PNet2BTagMean0p55 = booltrg[jk]; }
      // Photon triggers
      else if(jk==52) {  hlt_Photon175 = booltrg[jk]; }
      else if(jk==53) {  hlt_Photon200 = booltrg[jk]; }
      // MET triggers
      else if(jk==54) {  hlt_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60 = booltrg[jk]; }
      else if(jk==55) {  hlt_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60 = booltrg[jk]; }
      else if(jk==56) {  hlt_PFMETNoMu140_PFMHTNoMu140_IDTight = booltrg[jk]; }
      else if(jk==57) {  hlt_PFMETTypeOne140_PFMHT140_IDTight = booltrg[jk]; }
  }
    
  // trigger filling end //
  
  // L1 trigger info //
  
  edm::ESHandle<L1TUtmTriggerMenu> L1_menu;
  L1_menu = pset.getHandle(tok_L1_menu);
  
  edm::Handle<BXVector<GlobalAlgBlk>> L1_GtHandle;
  iEvent.getByToken(tok_L1_GtHandle, L1_GtHandle);
  
  L1_HTT280er = false;
  L1_QuadJet60er2p5 = false;
  L1_HTT320er = false;
  L1_HTT360er = false;
  L1_HTT400er = false;
  L1_HTT450er = false;
  L1_HTT280er_QuadJet_70_55_40_35_er2p5 = false;
  L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3 = false;
  L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3 = false;
  L1_Mu6_HTT240er = false;
  L1_SingleJet60 = false;
  
  if(L1_menu.isValid()){
  
	for(auto const & keyval: L1_menu->getAlgorithmMap())
	{
          if(keyval.second.getName() == "L1_HTT280er") idx_L1_HTT280er = keyval.second.getIndex();
          if(keyval.second.getName() == "L1_QuadJet60er2p5") idx_L1_QuadJet60er2p5 = keyval.second.getIndex();
          if(keyval.second.getName() == "L1_HTT320er") idx_L1_HTT320er = keyval.second.getIndex();
          if(keyval.second.getName() == "L1_HTT360er") idx_L1_HTT360er = keyval.second.getIndex();
          if(keyval.second.getName() == "L1_HTT400er") idx_L1_HTT400er = keyval.second.getIndex();
          if(keyval.second.getName() == "L1_HTT450er") idx_L1_HTT450er = keyval.second.getIndex();
          if(keyval.second.getName() == "L1_HTT280er_QuadJet_70_55_40_35_er2p5") idx_L1_HTT280er_QuadJet_70_55_40_35_er2p5 = keyval.second.getIndex();
          if(keyval.second.getName() == "L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3") idx_L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3 = keyval.second.getIndex();
          if(keyval.second.getName() == "L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3") idx_L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3 = keyval.second.getIndex();
          if(keyval.second.getName() == "L1_Mu6_HTT240er") idx_L1_Mu6_HTT240er = keyval.second.getIndex();
          if(keyval.second.getName() == "L1_SingleJet60") idx_L1_SingleJet60 = keyval.second.getIndex();
	}


      if(L1_GtHandle.isValid()){
		  
         int ibx = 0;
         for(auto itr = L1_GtHandle->begin(ibx); itr != L1_GtHandle->end(ibx); ++itr){
          if(itr->getAlgoDecisionFinal(idx_L1_HTT280er)){ L1_HTT280er = true;}
          if(itr->getAlgoDecisionFinal(idx_L1_QuadJet60er2p5)){ L1_QuadJet60er2p5 = true;}
          if(itr->getAlgoDecisionFinal(idx_L1_HTT320er)){ L1_HTT320er = true;}
          if(itr->getAlgoDecisionFinal(idx_L1_HTT360er)){ L1_HTT360er = true;}
          if(itr->getAlgoDecisionFinal(idx_L1_HTT400er)){ L1_HTT400er = true;}
          if(itr->getAlgoDecisionFinal(idx_L1_HTT450er)){ L1_HTT450er = true;}
          if(itr->getAlgoDecisionFinal(idx_L1_HTT280er_QuadJet_70_55_40_35_er2p5)){ L1_HTT280er_QuadJet_70_55_40_35_er2p5 = true;}
          if(itr->getAlgoDecisionFinal(idx_L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3)){ L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3 = true;}
          if(itr->getAlgoDecisionFinal(idx_L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3)){ L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3 = true;}
          if(itr->getAlgoDecisionFinal(idx_L1_Mu6_HTT240er)){ L1_Mu6_HTT240er = true;}
          if(itr->getAlgoDecisionFinal(idx_L1_SingleJet60)){ L1_SingleJet60 = true;}
         }
       }

  
	}
	
  // L1 trigger info end //
  
  // End of trigger info //
  
  //  ====  MET filters   ====  //
  
  edm::Handle<edm::TriggerResults> METFilterResults;
  iEvent.getByToken(tok_METfilters_, METFilterResults);
  if(!(METFilterResults.isValid())) iEvent.getByToken(tok_METfilters_, METFilterResults);
  
  const edm::TriggerNames & metfilterName = iEvent.triggerNames(*METFilterResults);
  //Flag_goodVertices
  unsigned int goodVerticesIndex_ = metfilterName.triggerIndex("Flag_goodVertices");
  Flag_goodVertices_ = METFilterResults.product()->accept(goodVerticesIndex_);
  //Flag_globalSuperTightHalo2016Filter
  Flag_globalSuperTightHalo2016Filter_ = METFilterResults.product()->accept(metfilterName.triggerIndex("Flag_globalSuperTightHalo2016Filter"));
  //Flag_EcalDeadCellTriggerPrimitiveFilter
  Flag_EcalDeadCellTriggerPrimitiveFilter_ = METFilterResults.product()->accept(metfilterName.triggerIndex("Flag_EcalDeadCellTriggerPrimitiveFilter"));
  //Flag_BadPFMuonFilter
  Flag_BadPFMuonFilter_ = METFilterResults.product()->accept(metfilterName.triggerIndex("Flag_BadPFMuonFilter"));
  //Flag_BadPFMuonDzFilter
  Flag_BadPFMuonDzFilter_ = METFilterResults.product()->accept(metfilterName.triggerIndex("Flag_BadPFMuonDzFilter"));
  //Flag_hfNoisyHitsFilter
  Flag_hfNoisyHitsFilter_ = METFilterResults.product()->accept(metfilterName.triggerIndex("Flag_hfNoisyHitsFilter"));
  //Flag_eeBadScFilter
  Flag_eeBadScFilter_ = METFilterResults.product()->accept(metfilterName.triggerIndex("Flag_eeBadScFilter"));
  //Flag_ecalBadCalibFilter
  Flag_ecalBadCalibFilter_ = METFilterResults.product()->accept(metfilterName.triggerIndex("Flag_ecalBadCalibFilter"));
  
  // End of MET filters //
  
  // ====  Prefire weights ==== //
  
  if(add_prefireweights){
  
	edm::Handle< double > theprefweight;
	iEvent.getByToken(prefweight_token, theprefweight ) ;	
	prefiringweight =(*theprefweight);

	edm::Handle< double > theprefweightup;
	iEvent.getByToken(prefweightup_token, theprefweightup ) ;
	prefiringweightup =(*theprefweightup);
    
	edm::Handle< double > theprefweightdown;
	iEvent.getByToken(prefweightdown_token, theprefweightdown ) ;   
	prefiringweightdown =(*theprefweightdown);
 
  }
  
  // End of prefire weights //
  
  // ====== RECO-objects now  ==========//
  
  // MET //
  
  if(store_CHS_met){
    
  // MET uncertainty ids are taken from: https://github.com/cms-sw/cmssw/blob/CMSSW_8_0_X/DataFormats/PatCandidates/interface/MET.h#L152-L158 //
  
  // CHS MET //
  
  edm::Handle<pat::METCollection> pfmet_ ;
  iEvent.getByToken(tok_mets_,pfmet_) ;
  
  if(pfmet_.isValid()){
	  
    const pat::MET &met = pfmet_->front();
    
    miset = met.corPt(); //met.pt();
    misphi = met.corPhi();//met.phi();
    misetsig = met.metSignificance();
    sumEt = met.corSumEt();//sumEt();
            
    miset_covXX = met.getSignificanceMatrix().At(0,0);
    miset_covXY = met.getSignificanceMatrix().At(0,1);
    miset_covYY = met.getSignificanceMatrix().At(1,1);
    
    //MET uncertainty numbering scheme: https://cmssdt.cern.ch/lxr/source/DataFormats/PatCandidates/interface/MET.h 
	    
    miset_UnclusEup = met.shiftedPt(pat::MET::UnclusteredEnUp);  // met.shiftedPt(pat::MET::METUncertainty(10));
	miset_UnclusEdn = met.shiftedPt(pat::MET::UnclusteredEnDown);// met.shiftedPt(pat::MET::METUncertainty(11));
	
	misphi_UnclusEup = met.shiftedPhi(pat::MET::UnclusteredEnUp);  //(pat::MET::METUncertainty(10));
	misphi_UnclusEdn = met.shiftedPhi(pat::MET::UnclusteredEnDown);//(pat::MET::METUncertainty(11));
	    
    if(isMC){
      genmiset = met.genMET()->pt();
      genmisphi = met.genMET()->phi();
      genmisetsig = met.genMET()->significance();
    }
  }
  
  }//store_CHS_met
  
  // PUPPI MET //
  
  if(store_PUPPI_met){
  
  edm::Handle<pat::METCollection> pfmet_PUPPI_ ;
  iEvent.getByToken(tok_mets_PUPPI_,pfmet_PUPPI_) ;
  
  if(pfmet_PUPPI_.isValid()){
	
	const pat::MET &met = pfmet_PUPPI_->front();
	  
	miset_PUPPI = met.corPt(); 
    misphi_PUPPI = met.corPhi();
    misetsig_PUPPI = met.metSignificance();
    sumEt_PUPPI = met.corSumEt();
        
    miset_PUPPI_covXX = met.getSignificanceMatrix().At(0,0);
    miset_PUPPI_covXY = met.getSignificanceMatrix().At(0,1);
    miset_PUPPI_covYY = met.getSignificanceMatrix().At(1,1);
    
    //MET uncertainty numbering scheme: https://cmssdt.cern.ch/lxr/source/DataFormats/PatCandidates/interface/MET.h
    
    miset_PUPPI_JESup = met.shiftedPt(pat::MET::JetEnUp); //(pat::MET::METUncertainty(2));
	miset_PUPPI_JESdn = met.shiftedPt(pat::MET::JetEnDown); //(pat::MET::METUncertainty(3));
	miset_PUPPI_JERup = met.shiftedPt(pat::MET::JetResUp); //(pat::MET::METUncertainty(0));
	miset_PUPPI_JERdn = met.shiftedPt(pat::MET::JetResDown); //(pat::MET::METUncertainty(1));
	miset_PUPPI_UnclusEup = met.shiftedPt(pat::MET::UnclusteredEnUp);  //(pat::MET::METUncertainty(10));
	miset_PUPPI_UnclusEdn = met.shiftedPt(pat::MET::UnclusteredEnDown);//(pat::MET::METUncertainty(11));
	
	misphi_PUPPI_JESup = met.shiftedPhi(pat::MET::JetEnUp); //(pat::MET::METUncertainty(2));
	misphi_PUPPI_JESdn = met.shiftedPhi(pat::MET::JetEnDown); //(pat::MET::METUncertainty(3));
	misphi_PUPPI_JERup = met.shiftedPhi(pat::MET::JetResUp); //(pat::MET::METUncertainty(0));
	misphi_PUPPI_JERdn = met.shiftedPhi(pat::MET::JetResDown); //(pat::MET::METUncertainty(1));
	misphi_PUPPI_UnclusEup = met.shiftedPhi(pat::MET::UnclusteredEnUp);  //(pat::MET::METUncertainty(10));
	misphi_PUPPI_UnclusEdn = met.shiftedPhi(pat::MET::UnclusteredEnDown);//(pat::MET::METUncertainty(11));
	
	//See DataFormats/PatCandidates/interface/MET.h for the names of uncertainty sources //
	
  }
  
  }//store_PUPPI_met
  
  // Muons //
    
  nMuon = 0;                                                                                                                                        
  std::vector<pat::Muon> tlvmu;
  edm::Handle<edm::View<pat::Muon>> muons;                                                                                                          
  iEvent.getByToken(tok_muons_, muons);                                                                                                             
    
  if(muons.isValid() && muons->size()>0 && store_muons) {                                                                                                           
    
	edm::View<pat::Muon>::const_iterator muon1;                                                                                                      

    for( muon1 = muons->begin(); muon1 < muons->end(); muon1++ ) {                                                                                   

		if (StoreMuon(*muon1,min_pt_mu,max_eta)) {                                                                
			
			Muon_pt[nMuon] = muon1->pt();                                                                         
			TrackRef trktrk = muon1->innerTrack();                                                                                                       
			Muon_p[nMuon] = trktrk->p()*muon1->charge();                                                                                                                 
			Muon_eta[nMuon] = muon1->eta();                                                                                                              
			Muon_phi[nMuon] = muon1->phi(); 
						
			Muon_tunePBestTrack_pt[nMuon] = muon1->tunePMuonBestTrack()->pt();
			                                         
			// Basic id variables //    
			                                  
			Muon_isPF[nMuon] = muon1->isPFMuon();                                                                                                        
			Muon_isGL[nMuon] = muon1->isGlobalMuon();                                                                                                    
			Muon_isTRK[nMuon] = muon1->isTrackerMuon();    
			Muon_isStandAloneMuon[nMuon] = muon1->isStandAloneMuon();                                                                                          
			                                                                                   
			Muon_isMedPr[nMuon] = false;                                                                          
			if(muon::isMediumMuon(*muon1)) {                                                                                                             
				if ((std::abs(muon1->muonBestTrack()->dz(vertex.position())) < 0.1) && (std::abs(muon1->muonBestTrack()->dxy(vertex.position())) < 0.02)){                                                                                                                  
					Muon_isMedPr[nMuon] = true;                                                                                                              
				}                                                                                                                                          
			}                                                                                                                                      
			Muon_isGoodGL[nMuon] = (muon1->isGlobalMuon() && muon1->globalTrack()->normalizedChi2() < 3 && muon1->combinedQuality().chi2LocalPosition < 12 && muon1->combinedQuality().trkKink < 20 && (muon::segmentCompatibility(*muon1)) > 0.303);                     
			
			// only ID booleans //
			
			Muon_isLoose[nMuon] =     (muon1->passed(reco::Muon::CutBasedIdLoose)); 		// (muon::isLooseMuon(*muon1));                                                                                           
			Muon_isMed[nMuon] =       (muon1->passed(reco::Muon::CutBasedIdMedium)); 		// (muon::isMediumMuon(*muon1));         
			Muon_mediumPromptId[nMuon] = (muon1->passed(reco::Muon::CutBasedIdMediumPrompt)); 
			Muon_isTight[nMuon] =     (muon1->passed(reco::Muon::CutBasedIdTight)); 		//(muon::isTightMuon(*muon1,vertex));                                                                                    
			Muon_isHighPt[nMuon] =    (muon1->passed(reco::Muon::CutBasedIdGlobalHighPt)); 	//(muon::isHighPtMuon(*muon1,vertex));                                                                                  
			Muon_isHighPttrk[nMuon] = (muon1->passed(reco::Muon::CutBasedIdTrkHighPt)); 	//(muon::isTrackerHighPtMuon(*muon1,vertex));   
			
			//Muon_MVAID[nMuon] = ( ((muon1->passed(reco::Muon::MvaLoose)) << 2) | ((muon1->passed(reco::Muon::MvaMedium)) << 1) | (muon1->passed(reco::Muon::MvaTight)) );
			Muon_MVAID[nMuon] = ((muon1->passed(reco::Muon::MvaLoose))+(muon1->passed(reco::Muon::MvaMedium))+(muon1->passed(reco::Muon::MvaTight)));
			Muon_mvaMuID[nMuon] = muon1->userFloat("mvaIDMuon");
			Muon_mvaMuID_WP[nMuon] = muon1->userFloat("mvaIDMuon_wpMedium")+muon1->userFloat("mvaIDMuon_wpTight");
			 
			// Iso booleans //
			
			//uint8_t muon_pf_iso_packed =   ( ((muon1->passed(reco::Muon::PFIsoVeryLoose)) << 5) | ((muon1->passed(reco::Muon::PFIsoLoose)) << 4) | ((muon1->passed(reco::Muon::PFIsoMedium)) << 3) | ((muon1->passed(reco::Muon::PFIsoTight)) << 2) | ((muon1->passed(reco::Muon::PFIsoVeryTight)) << 1) | (muon1->passed(reco::Muon::PFIsoVeryVeryTight)) );
			//uint8_t muon_mini_iso_packed = ( ((muon1->passed(reco::Muon::MiniIsoLoose)) << 3) | ((muon1->passed(reco::Muon::MiniIsoMedium)) << 2) | ((muon1->passed(reco::Muon::MiniIsoTight)) << 1) | (muon1->passed(reco::Muon::MiniIsoVeryTight)) );
			
			//make it simple//			
			Muon_PF_iso[nMuon] = (muon1->passed(reco::Muon::PFIsoVeryLoose)) + (muon1->passed(reco::Muon::PFIsoLoose)) + (muon1->passed(reco::Muon::PFIsoMedium)) + (muon1->passed(reco::Muon::PFIsoTight)) + (muon1->passed(reco::Muon::PFIsoVeryTight)) + (muon1->passed(reco::Muon::PFIsoVeryVeryTight));
			Muon_Mini_iso[nMuon] = (muon1->passed(reco::Muon::MiniIsoLoose)) + (muon1->passed(reco::Muon::MiniIsoMedium)) + (muon1->passed(reco::Muon::MiniIsoTight)) + (muon1->passed(reco::Muon::MiniIsoVeryTight));
		
			Muon_multiIsoId[nMuon] = (muon1->passed(reco::Muon::MultiIsoMedium))?2:(muon1->passed(reco::Muon::MultiIsoLoose));
			Muon_puppiIsoId[nMuon] = (muon1->passed(reco::Muon::PuppiIsoLoose))+(muon1->passed(reco::Muon::PuppiIsoMedium))+(muon1->passed(reco::Muon::PuppiIsoTight));
			Muon_tkIsoId[nMuon] =    (muon1->passed(reco::Muon::TkIsoTight))?2:(muon1->passed(reco::Muon::TkIsoLoose));
						
			// Displacement //
			
			Muon_dxy[nMuon] = muon1->dB(pat::Muon::PV2D); //muon1->muonBestTrack()->dxy(vertex.position());      
			Muon_dxybs[nMuon] = muon1->dB(pat::Muon::BS2D);                                                                   
			Muon_dz[nMuon] = muon1->dB(pat::Muon::PVDZ);  //muon1->muonBestTrack()->dz(vertex.position());  
			Muon_dxyErr[nMuon] = muon1->edB(pat::Muon::PV2D);    
			Muon_dzErr[nMuon] = muon1->edB(pat::Muon::PVDZ);    
			Muon_ip3d[nMuon] =  muon1->dB(pat::Muon::PV3D);  
			Muon_sip3d[nMuon] =  muon1->dB(pat::Muon::PV3D)/muon1->edB(pat::Muon::PV3D);   
			 
			// energy & track info //
			
			TrackRef trkglb =muon1->globalTrack();                                                                                                       
			
			if ((!muon1->isGlobalMuon())) {                                                                                                              
				if (muon1->isTrackerMuon()) {                                                                                                              
					trkglb =muon1->innerTrack();                                                                                                             
				} else {                                                                                                                                   
					trkglb =muon1->outerTrack();                                                                                                             
				}                                                                                                                                          
			}
			
			if(store_muon_id_variables){	
			            
				Muon_valfrac[nMuon] = trktrk->validFraction();  
				Muon_chi[nMuon] = trkglb->normalizedChi2();  
				Muon_posmatch[nMuon] = muon1->combinedQuality().chi2LocalPosition;  
				Muon_trkink[nMuon] = muon1->combinedQuality().trkKink;  
				Muon_segcom[nMuon] = muon::segmentCompatibility(*muon1);     
				Muon_hit[nMuon] = trkglb->hitPattern().numberOfValidMuonHits();       
				Muon_mst[nMuon] = muon1->numberOfMatchedStations(); 
				Muon_pixhit[nMuon] = trktrk->hitPattern().numberOfValidPixelHits();                                                                          
				Muon_trklay[nMuon] = trktrk->hitPattern().trackerLayersWithMeasurement();                     
			
			}
			
			if(store_additional_muon_id_variables){	
			                                                                                                                                                                                                    
				Muon_ecal[nMuon] = (muon1->calEnergy()).em;                                                                                                  
				Muon_hcal[nMuon] = (muon1->calEnergy()).had;  
				
				// Track info //
		                                                                                    
				Muon_ndf[nMuon] = (int)trkglb->ndof();   
				Muon_ptErr[nMuon] = trktrk->ptError();                                                                                                     
			                                                                                                                         	    
				// Displacement w.r.t secondary vertex //
			                                                                    
				float dzmumin = 1000;                                                                                                                        
				float dxymumin = 1000;                                                                                                                       
				if(secondaryVertices.isValid()){                                                                                                                          
					for(unsigned int isv=0; isv<(secondaryVertices->size()); isv++){                                                                                        
						const auto &sv = (*secondaryVertices)[isv];                                                                                                           
						reco::TrackBase::Point svpoint(sv.vx(),sv.vy(),sv.vz());
						float dztmp = fabs(muon1->muonBestTrack()->dz(svpoint));
						if(dztmp < dzmumin){
							dzmumin = dztmp;                                                                                   
							dxymumin = muon1->muonBestTrack()->dxy(svpoint);                                                                                       
						}                                                                                                                                        
					}                                                                                                                                          
				}  
				                                                                                                                                          
				Muon_dxy_sv[nMuon] = dxymumin; 
			
				bool mu_id = Muon_Tight_ID(Muon_isGL[nMuon],Muon_isPF[nMuon],
										   Muon_chi[nMuon],Muon_hit[nMuon],Muon_mst[nMuon],
										   Muon_dxy[nMuon],Muon_dz[nMuon],Muon_pixhit[nMuon],Muon_trklay[nMuon]);
				Muon_TightID[nMuon] = mu_id;
		                       
			}//store_muon_id_variables                                                                                 
						                                                     
			Muon_pfiso[nMuon] 	= (muon1->pfIsolationR04().sumChargedHadronPt + max(0., muon1->pfIsolationR04().sumNeutralHadronEt + muon1->pfIsolationR04().sumPhotonEt - 0.5*muon1->pfIsolationR04().sumPUPt))/muon1->pt();                                               
			Muon_pfiso03[nMuon] = (muon1->pfIsolationR03().sumChargedHadronPt + max(0., muon1->pfIsolationR03().sumNeutralHadronEt + muon1->pfIsolationR03().sumPhotonEt - 0.5*muon1->pfIsolationR03().sumPUPt))/muon1->pt();                                               
			
			//MiniIsolation: begin//                                                                                      
			vector<float> isovalues;
			Read_MiniIsolation(muon1,Rho,isovalues);
			Muon_minisoall[nMuon] = isovalues[0];
			//Muon_minchiso[nMuon] = isovalues[1];
			//Muon_minnhiso[nMuon] = isovalues[2];
			//Muon_minphiso[nMuon] = isovalues[3];
			
			Muon_miniPFRelIso_all[nMuon] = muon1->userFloat("miniIsoAll")*1./Muon_pt[nMuon];
			Muon_miniPFRelIso_Chg[nMuon] = muon1->userFloat("miniIsoChg")*1./Muon_pt[nMuon];
			
			//MiniIsolation: end//  
			
			//if (Muon_pt[nMuon]>min_pt_mu && fabs(Muon_eta[nMuon])<max_eta && Muon_isLoose[nMuon] && abs(Muon_dxy[nMuon])<0.2 && abs(Muon_dz[nMuon])<0.5) {
			if (Muon_pt[nMuon]>min_pt_mu && fabs(Muon_eta[nMuon])<max_eta && Muon_isLoose[nMuon] && muon1->passed(reco::Muon::PFIsoLoose)){
				tlvmu.push_back(*muon1);
			}
			
			// Application of Rochester correction //
			
			float rcSF, rcSF_error;
			
			if(isUltraLegacy){
			
			if(!isMC){
				// Data
				rcSF = roch_cor.kScaleDT(muon1->charge(), Muon_pt[nMuon], Muon_eta[nMuon], Muon_phi[nMuon]); 
				rcSF_error = roch_cor.kScaleDTerror(muon1->charge(), Muon_pt[nMuon], Muon_eta[nMuon], Muon_phi[nMuon]); 
			}
			else{
				// MC
				bool gen_match = false;
				float match_genpt = -100;
				float dR_cut = 0.1;
				for(int ipart=0; ipart<nGenPart; ipart++)
				{
					if((GenPart_status[ipart]==1) && (GenPart_pdg[ipart]==(-1*muon1->charge()*13)) && (delta2R(GenPart_eta[ipart],GenPart_phi[ipart],Muon_eta[nMuon], Muon_phi[nMuon])<dR_cut))
					{
						dR_cut = delta2R(GenPart_eta[ipart],GenPart_phi[ipart],Muon_eta[nMuon], Muon_phi[nMuon]);
						gen_match = true;
						match_genpt = GenPart_pt[ipart];
					}
				}
				if(gen_match){
					// when matched gen muon is available
					rcSF = roch_cor.kSpreadMC(muon1->charge(), Muon_pt[nMuon], Muon_eta[nMuon], Muon_phi[nMuon], match_genpt); 
					rcSF_error = roch_cor.kSpreadMCerror(muon1->charge(), Muon_pt[nMuon], Muon_eta[nMuon], Muon_phi[nMuon], match_genpt);
				} 
				else{
					// when matched gen muon is not available
					rcSF = roch_cor.kSmearMC(muon1->charge(), Muon_pt[nMuon], Muon_eta[nMuon], Muon_phi[nMuon], Muon_trklay[nMuon], gRandom->Rndm()); 
					rcSF_error = roch_cor.kSmearMCerror(muon1->charge(), Muon_pt[nMuon], Muon_eta[nMuon], Muon_phi[nMuon], Muon_trklay[nMuon], gRandom->Rndm());
				}
			}
			
			}//isUltraLegacy
			else{
				rcSF = 1;
				rcSF_error = 0;
			}
						
			Muon_corrected_pt[nMuon] = Muon_pt[nMuon]*rcSF;
			Muon_correctedUp_pt[nMuon] = Muon_pt[nMuon]*max(rcSF+rcSF_error,float(0.));
			Muon_correctedDown_pt[nMuon] = Muon_pt[nMuon]*max(rcSF-rcSF_error,float(0.));
			
			// End of Rochester correction //
			
			if (++nMuon>=njetmx) break;                                                                                                                 
		
		}                                                                                                                                              
      }                                                                                                                                               
  }// muon loop 
  
  // Electrons //
      
  nElectron = 0;             
  std::vector<pat::Electron> tlvel;
  
  if(store_electrons){
    
  for(const auto& electron1 : iEvent.get(tok_electrons_)) {                                                                                          
 
    if (!StoreElectron(electron1,min_pt_el,max_eta)) continue;
                                                   
	GsfTrackRef gsftrk1 = electron1.gsfTrack();   																														
    TrackRef ctftrk = electron1.closestCtfTrackRef();    
    
    Electron_pt[nElectron] = electron1.pt();   	                                                                                 
    Electron_eta[nElectron] = electron1.eta();                                                                                                                 
    Electron_phi[nElectron] = electron1.phi();                                                                                                                 
    Electron_e[nElectron] = electron1.energy();                                                                                        
    Electron_p[nElectron] = electron1.trackMomentumAtVtx().R()*electron1.charge();     
    
    // Cut-based ID //
    
    Electron_cutbased_id[nElectron] = (electron1.electronID(melectronID_cutbased_loose)) + (electron1.electronID(melectronID_cutbased_medium)) + (electron1.electronID(melectronID_cutbased_tight));
    
    // MVA id booleans //
    
    Electron_mvaid_Winter22v1WP90[nElectron] = electron1.electronID(melectronID_isowp90);                                                                                 
    Electron_mvaid_Winter22v1WP90_noIso[nElectron] = electron1.electronID(melectronID_noisowp90);                                                                             
    Electron_mvaid_Winter22v1WP80[nElectron] = electron1.electronID(melectronID_isowp80);                                                                                 
    Electron_mvaid_Winter22v1WP80_noIso[nElectron] = electron1.electronID(melectronID_noisowp80);  
    // loose MVA IDs don't exist in Run 3 
    //Electron_mvaid_Winter22v1WPLoose[nElectron] = electron1.electronID(melectronID_isowploose);                                                                                 
    //Electron_mvaid_Winter22v1WPLoose_noIso[nElectron] = electron1.electronID(melectronID_noisowploose);   
    
    Electron_mvaid_Fallv2WP90[nElectron] = electron1.electronID(melectronID_isowp90_Fall17);                                                                                 
    Electron_mvaid_Fallv2WP90_noIso[nElectron] = electron1.electronID(melectronID_noisowp90_Fall17);                                                                             
    Electron_mvaid_Fallv2WP80[nElectron] = electron1.electronID(melectronID_isowp80_Fall17);                                                                                 
    Electron_mvaid_Fallv2WP80_noIso[nElectron] = electron1.electronID(melectronID_noisowp80_Fall17);   
    Electron_mvaid_Fallv2WPLoose[nElectron] = electron1.electronID(melectronID_isowploose_Fall17);                                                                                 
    Electron_mvaid_Fallv2WPLoose_noIso[nElectron] = electron1.electronID(melectronID_noisowploose_Fall17);   
    
    // displacement //
                                                                                 
    Electron_dxy[nElectron] = gsftrk1->dxy(vertex.position());  
    Electron_dxyErr[nElectron] = electron1.edB(pat::Electron::PV2D);                                                                                           
    Electron_dz[nElectron] = gsftrk1->dz(vertex.position()); 
    Electron_dzErr[nElectron] = electron1.edB(pat::Electron::PVDZ);     
    Electron_ip3d[nElectron] =  electron1.dB(pat::Electron::PV3D);
    Electron_sip3d[nElectron] =  electron1.dB(pat::Electron::PV3D)/electron1.edB(pat::Electron::PV3D);                                                                                           
    
    // supercluste info //
    
    Electron_supcl_e[nElectron] = electron1.superCluster()->energy();  
    Electron_supcl_rawE[nElectron] = electron1.superCluster()->rawEnergy();      
    Electron_supcl_eta[nElectron] = electron1.superCluster()->eta();                                                                                           
    Electron_supcl_phi[nElectron] = electron1.superCluster()->phi();                                                                                           
    
    // scaling & smearing factors //
    
    if(store_electron_scalnsmear){
		Electron_eccalTrkEnergyPostCorr[nElectron] = electron1.userFloat("ecalTrkEnergyPostCorr");
		Electron_energyScaleValue[nElectron] = electron1.userFloat("energyScaleValue");
		Electron_energySigmaValue[nElectron] = electron1.userFloat("energySigmaValue");
		Electron_energyScaleUp[nElectron] = electron1.userFloat("energyScaleUp");
		Electron_energyScaleDown[nElectron] = electron1.userFloat("energyScaleDown");
		Electron_energySigmaUp[nElectron] = electron1.userFloat("energySigmaUp");
		Electron_energySigmaDown[nElectron] = electron1.userFloat("energySigmaDown");
	}
    // end of scaling & smearing factors //                                                                                                
                 
    // shape of energy deposition // 
    
    if(store_electron_id_variables){    
		
		// MVA ID values //
		
		Electron_mvaid_Fallv2_value[nElectron] = electron1.userFloat("ElectronMVAEstimatorRun2Fall17IsoV2Values");
		Electron_mvaid_Fallv2noIso_value[nElectron] = electron1.userFloat("ElectronMVAEstimatorRun2Fall17NoIsoV2Values");
	
		Electron_mvaid_Winter22IsoV1_value[nElectron] = electron1.userFloat("mvaIso");
		Electron_mvaid_Winter22NoIsoV1_value[nElectron] = electron1.userFloat("mvaNoIso");
        
        // shower shape //
        // following variables need to calculate cut-based ID //
                                                                                                                                                                                                                                                  
		Electron_sigmaieta[nElectron] = electron1.full5x5_sigmaIetaIeta();                                                                                         
		Electron_sigmaiphi[nElectron] = electron1.full5x5_sigmaIphiIphi();   
		Electron_etain[nElectron] = electron1.deltaEtaSuperClusterTrackAtVtx();                                                                                    
		Electron_phiin[nElectron] = electron1.deltaPhiSuperClusterTrackAtVtx();      
		Electron_hovere[nElectron] = electron1.hadronicOverEm();             
		Electron_hitsmiss[nElectron] =  electron1.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);   
		Electron_eoverp[nElectron] = electron1.eSuperClusterOverP();   
		Electron_e_ECAL[nElectron] = electron1.ecalEnergy();     
		Electron_convVeto[nElectron] = electron1.passConversionVeto();
		     
	}
	
	//needed for a veto in Run3 2022EE// https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis#From_E_Gamma
	Electron_seediEtaOriX[nElectron] = electron1.superCluster()->seedCrysIEtaOrIx();
	Electron_seediPhiOriY[nElectron] = electron1.superCluster()->seedCrysIPhiOrIy();
	
	if(store_additional_electron_id_variables){
		
		// shower shape //
                                                                                                                   
		Electron_r9full[nElectron] = electron1.full5x5_r9();                                                                                                                                                                                        
		Electron_hcaloverecal[nElectron] = electron1.full5x5_hcalOverEcal();                                                                                                                                                                                                                                                                                                                            
		Electron_ietaieta[nElectron] = electron1.sigmaIetaIeta();
		Electron_ecloverpout[nElectron] = electron1.eEleClusterOverPout();  
		                                                                                                                                                                                  
		Electron_fbrem[nElectron] = electron1.fbrem();   
		Electron_supcl_preshvsrawe[nElectron] = electron1.superCluster()->preshowerEnergy()/electron1.superCluster()->rawEnergy();
		Electron_cloctftrkn[nElectron] = electron1.closestCtfTrackNLayers();                                                                                       
		Electron_cloctftrkchi2[nElectron] = electron1.closestCtfTrackNormChi2();   
		Electron_e1x5bye5x5[nElectron] = 1.-electron1.full5x5_e1x5()/electron1.full5x5_e5x5();                                                                     
		Electron_normchi2[nElectron] =  electron1.gsfTrack()->normalizedChi2(); 
		Electron_supcl_etaw[nElectron] = electron1.superCluster()->etaWidth();                                                                                     
		Electron_supcl_phiw[nElectron] = electron1.superCluster()->phiWidth(); 
		Electron_trkmeasure[nElectron] = electron1.gsfTrack()->hitPattern().trackerLayersWithMeasurement();  
		Electron_convtxprob[nElectron] = electron1.convVtxFitProb();                                                                                                                                                   
		Electron_ecaletrkmomentum[nElectron] = 1.0/(electron1.ecalEnergy())-1.0/(electron1.trackMomentumAtVtx().R());                                              
		Electron_deltaetacltrkcalo[nElectron] = electron1.deltaEtaSeedClusterTrackAtCalo();           
		
		Electron_pfisolsumphet[nElectron] = electron1.pfIsolationVariables().sumPhotonEt;                                                                          
		Electron_pfisolsumchhadpt[nElectron] = electron1.pfIsolationVariables().sumChargedHadronPt;                                                                
		Electron_pfsiolsumneuhadet[nElectron] = electron1.pfIsolationVariables().sumNeutralHadronEt;   
		
		// track info //
		
		Electron_chi[nElectron] = gsftrk1->chi2();                                                                                                                 
		Electron_ndf[nElectron] = (int)gsftrk1->ndof();                                                                                                            
		Electron_misshits[nElectron] = (int)gsftrk1->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
   
		// Displacement w.r.t secondary vertex //
    
		float dzmin = 1000;                                                                                                                              
		float dxymin = 1000;
		if(secondaryVertices.isValid()){                                                                                                                              
			for(unsigned int isv=0; isv<(secondaryVertices->size()); isv++){                                                                                            
				const auto &sv = (*secondaryVertices)[isv];                                                                                                               
				reco::TrackBase::Point svpoint(sv.vx(),sv.vy(),sv.vz());
				float dztmp =fabs(gsftrk1->dz(svpoint));
				if(dztmp < dzmin){                                                                                                      
					dzmin = dztmp;                                                                                                        
					dxymin = gsftrk1->dxy(svpoint);    
				}                                                                                                                                            
			}                                                                                                                                              
		}     
        
        Electron_dxy_sv[nElectron] = dxymin;  
		
	                                                                                                         	
	} //store_additional_electron_id_variables
	
	// isolation variables //                              
    
    if(store_electron_id_variables){
                                                              
		const reco::GsfElectron::PflowIsolationVariables& pfIso = electron1.pfIsolationVariables();                                                      
		Electron_pfiso_drcor[nElectron] = (pfIso.sumChargedHadronPt + max(0., pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - 0.5*pfIso.sumPUPt))*1./electron1.pt();      
    
		vector<float> pfisovalues;                                                                                     
		Read_ElePFIsolation(&electron1,Rho,pfisovalues);
		Electron_pfiso_eacor[nElectron] = pfisovalues[0];
		Electron_pfiso04_eacor[nElectron] = pfisovalues[1];
    
		Electron_pfRelIso03_all[nElectron] = electron1.userFloat("PFIsoAll")*1./Electron_pt[nElectron];
		Electron_pfRelIso04_all[nElectron] = electron1.userFloat("PFIsoAll04")*1./Electron_pt[nElectron];
    
		//In principle, 
		//Electron_pfiso_eacor =  Electron_pfRelIso03_all
		//Electron_pfiso04_eacor = Electron_pfRelIso04_all
		//the latter ones are stored for convenience (to synchronize with older version of framework)
    
	}
    
    //MiniIsolation: begin//                                                                                      
	
	vector<float> isovalues;
	Read_MiniIsolation(&electron1,Rho,isovalues);
	Electron_minisoall[nElectron] = isovalues[0];
	//Electron_minchiso[nElectron] = isovalues[1];
	//Electron_minnhiso[nElectron] = isovalues[2];
	//Electron_minphiso[nElectron] = isovalues[3];
	
	Electron_miniPFRelIso_all[nElectron] = electron1.userFloat("miniIsoAll")*1./Electron_pt[nElectron];
	Electron_miniPFRelIso_chg[nElectron] = electron1.userFloat("miniIsoChg")*1./Electron_pt[nElectron];

	//In principle, 
	//Electron_miniPFRelIso_all = Electron_minisoall; the latter one is stored for convenience 
	
	//MiniIsolation: end//  
   
	bool impact_pass = 	((fabs(Electron_supcl_eta[nElectron])<1.4442 && fabs(Electron_dxy[nElectron])<0.05 && fabs(Electron_dz[nElectron])<0.1)
					   ||(fabs(Electron_supcl_eta[nElectron])>1.5660 && fabs(Electron_dxy[nElectron])<(2*0.05) && fabs(Electron_dz[nElectron])<(2*0.1)));

	
	//if(Electron_pt[nElectron]>min_pt_el && fabs(Electron_eta[nElectron])<max_eta && Electron_mvaid_Fallv2WP90_noIso[nElectron] && impact_pass){
	if(Electron_pt[nElectron]>min_pt_el && fabs(Electron_eta[nElectron])<max_eta && electron1.electronID(melectronID_cutbased_loose)){
		tlvel.push_back(electron1);
	}
  
    if(++nElectron>=njetmx) break;                                                                                                                      
  }
 
  } //store_electrons
  
  // AK8 jets //
  
  nPFJetAK8 = 0;
  nPFJetAK8_cons = 0;
  
  edm::Handle<edm::View<pat::Jet>> pfjetAK8s;
  iEvent.getByToken(tok_pfjetAK8s_, pfjetAK8s);	
  
  if(pfjetAK8s.isValid() && store_ak8jets){
    
    for (unsigned jet = 0; jet< pfjetAK8s->size(); jet++) {
      
      const auto &ak8jet = (*pfjetAK8s)[jet];

      TLorentzVector pfjetAK8_4v(ak8jet.correctedP4("Uncorrected").px(),ak8jet.correctedP4("Uncorrected").py(),ak8jet.correctedP4("Uncorrected").pz(), ak8jet.correctedP4("Uncorrected").energy());
     
	  if(subtractLepton_fromAK8){
		pfjetAK8_4v = LeptonJet_subtraction(tlvmu,ak8jet,pfjetAK8_4v);
		pfjetAK8_4v = LeptonJet_subtraction(tlvel,ak8jet,pfjetAK8_4v);
	  }
	  
      double tmprecpt = pfjetAK8_4v.Pt();
      
      double total_cor =1;
      Read_JEC(total_cor,tmprecpt,pfjetAK8_4v,Rho,isData,ak8jet,jecL1FastAK8,jecL2RelativeAK8,jecL3AbsoluteAK8,jecL2L3ResidualAK8);
      PFJetAK8_JEC[nPFJetAK8] = total_cor;
            
      if(tmprecpt<min_pt_AK8jet) continue;
      if(abs(ak8jet.eta())>max_eta) continue;
      
      // JEC corrected 4-momentum //
      TLorentzVector pfjetAK8_4v_jecor;
      pfjetAK8_4v_jecor = pfjetAK8_4v * PFJetAK8_JEC[nPFJetAK8];
       
      PFJetAK8_pt[nPFJetAK8] = 	pfjetAK8_4v.Pt();
      PFJetAK8_y[nPFJetAK8] = pfjetAK8_4v.Rapidity();
      PFJetAK8_eta[nPFJetAK8] = pfjetAK8_4v.Eta();
      PFJetAK8_phi[nPFJetAK8] = pfjetAK8_4v.Phi();
      PFJetAK8_mass[nPFJetAK8] = ak8jet.correctedP4("Uncorrected").mass();
      //PFJetAK8_btag_DeepCSV[nPFJetAK8] = ak8jet.bDiscriminator("pfDeepCSVJetTags:probb")+ak8jet.bDiscriminator("pfDeepCSVJetTags:probbb");
      
      // DNN-based tagger info //
      
      PFJetAK8_DeepTag_DAK8_TvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(toptagger_DAK8);
      PFJetAK8_DeepTag_DAK8_WvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Wtagger_DAK8);
      PFJetAK8_DeepTag_DAK8_ZvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Ztagger_DAK8);
      PFJetAK8_DeepTag_DAK8_HvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Htagger_DAK8);
      PFJetAK8_DeepTag_DAK8_bbvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(bbtagger_DAK8);
      
      //mass-correlated PNet taggers 
      PFJetAK8_DeepTag_PNet_TvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(toptagger_PNet);
      PFJetAK8_DeepTag_PNet_WvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Wtagger_PNet);
      PFJetAK8_DeepTag_PNet_ZvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Ztagger_PNet);
      PFJetAK8_DeepTag_PNet_HbbvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Hbbtagger_PNet);
      PFJetAK8_DeepTag_PNet_HccvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Hcctagger_PNet);
      PFJetAK8_DeepTag_PNet_H4qvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(H4qtagger_PNet);
      
      //mass-decorrelated PNet taggers
      PFJetAK8_DeepTag_PNet_XbbvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Xbbtagger_PNet);
      PFJetAK8_DeepTag_PNet_XccvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Xcctagger_PNet);
      PFJetAK8_DeepTag_PNet_XqqvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Xqqtagger_PNet);
      PFJetAK8_DeepTag_PNet_XggvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Xggtagger_PNet);
      PFJetAK8_DeepTag_PNet_XtevsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Xtetagger_PNet);
      PFJetAK8_DeepTag_PNet_XtmvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Xtmtagger_PNet);
      PFJetAK8_DeepTag_PNet_XttvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(Xtttagger_PNet);
      //Run2//
      PFJetAK8_DeepTag_PNet_QCD[nPFJetAK8] = (ak8jet.bDiscriminator("pfMassDecorrelatedParticleNetJetTags:probQCDbb")+ak8jet.bDiscriminator("pfMassDecorrelatedParticleNetJetTags:probQCDcc")+
										    ak8jet.bDiscriminator("pfMassDecorrelatedParticleNetJetTags:probQCDb")+ak8jet.bDiscriminator("pfMassDecorrelatedParticleNetJetTags:probQCDc")+
										    ak8jet.bDiscriminator("pfMassDecorrelatedParticleNetJetTags:probQCDothers"));
      //Run3//
      PFJetAK8_DeepTag_PNet_QCD[nPFJetAK8] = (PFJetAK8_DeepTag_PNet_QCD0HF[nPFJetAK8]+PFJetAK8_DeepTag_PNet_QCD1HF[nPFJetAK8]+PFJetAK8_DeepTag_PNet_QCD2HF[nPFJetAK8]);
      PFJetAK8_DeepTag_PNet_QCD0HF[nPFJetAK8] = ak8jet.bDiscriminator(QCD0HFtagger_PNet);
      PFJetAK8_DeepTag_PNet_QCD1HF[nPFJetAK8] = ak8jet.bDiscriminator(QCD1HFtagger_PNet);
      PFJetAK8_DeepTag_PNet_QCD2HF[nPFJetAK8] = ak8jet.bDiscriminator(QCD2HFtagger_PNet);
      
      //Global PartT taggers
      //X->bb/cc/cs/qq
      PFJetAK8_DeepTag_PartT_Xbb[nPFJetAK8] = ak8jet.bDiscriminator(Xbbtagger_PartT);
      PFJetAK8_DeepTag_PartT_Xcc[nPFJetAK8] = ak8jet.bDiscriminator(Xcctagger_PartT);
      PFJetAK8_DeepTag_PartT_Xcs[nPFJetAK8] = ak8jet.bDiscriminator(Xcstagger_PartT);
      PFJetAK8_DeepTag_PartT_Xqq[nPFJetAK8] = ak8jet.bDiscriminator(Xqqtagger_PartT);
      //t->bqq/bq/bev/bmv/btauv
      PFJetAK8_DeepTag_PartT_TopbWqq[nPFJetAK8] = ak8jet.bDiscriminator(TopbWqqtagger_PartT);
      PFJetAK8_DeepTag_PartT_TopbWq[nPFJetAK8] = ak8jet.bDiscriminator(TopbWqtagger_PartT);
      PFJetAK8_DeepTag_PartT_TopbWev[nPFJetAK8] = ak8jet.bDiscriminator(TopbWevtagger_PartT);
      PFJetAK8_DeepTag_PartT_TopbWmv[nPFJetAK8] = ak8jet.bDiscriminator(TopbWmvtagger_PartT);
      PFJetAK8_DeepTag_PartT_TopbWtauv[nPFJetAK8] = ak8jet.bDiscriminator(TopbWtauvtagger_PartT);
      //QCD
      PFJetAK8_DeepTag_PartT_QCD[nPFJetAK8] = ak8jet.bDiscriminator(QCDtagger_PartT);
      //H->WW
      PFJetAK8_DeepTag_PartT_XWW4q[nPFJetAK8] = ak8jet.bDiscriminator(XWW4qtagger_PartT);
      PFJetAK8_DeepTag_PartT_XWW3q[nPFJetAK8] = ak8jet.bDiscriminator(XWW3qtagger_PartT);
      PFJetAK8_DeepTag_PartT_XWWqqev[nPFJetAK8] = ak8jet.bDiscriminator(XWWqqevtagger_PartT);
      PFJetAK8_DeepTag_PartT_XWWqqmv[nPFJetAK8] = ak8jet.bDiscriminator(XWWqqmvtagger_PartT);
      //mass-correlated Top, W, Z vs QCD
      PFJetAK8_DeepTag_PartT_TvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(TvsQCDtagger_PartT);
      PFJetAK8_DeepTag_PartT_WvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(WvsQCDtagger_PartT);
      PFJetAK8_DeepTag_PartT_ZvsQCD[nPFJetAK8] = ak8jet.bDiscriminator(ZvsQCDtagger_PartT);
      
      //ParticleNet mass correction//
      PFJetAK8_particleNet_massCorr[nPFJetAK8] = ak8jet.bDiscriminator(mass_cor_PNet);
      //ParticleTransformer based mass correction//
      PFJetAK8_partT_massCorr_generic[nPFJetAK8] = ak8jet.bDiscriminator(mass_cor_PartT_genertic);
      PFJetAK8_partT_massCorr_twoprong[nPFJetAK8] = ak8jet.bDiscriminator(mass_cor_PartT_twoprong);
            
      //  JER //
       
      if(isMC){
	
		TLorentzVector tmp4v;
		tmp4v = pfjetAK8_4v_jecor;
		
      	vector<double> SFs;
      	Read_JER(mPtResoFileAK8, mPtSFFileAK8, tmprecpt, tmp4v, Rho, genjetAK8s, 0.5*0.8, SFs);
      	
      	PFJetAK8_reso[nPFJetAK8] = SFs[0];
      	PFJetAK8_resoup[nPFJetAK8] = SFs[1];
      	PFJetAK8_resodn[nPFJetAK8] = SFs[2];
      	      	
      }//isMC
      
      // JES uncertainty //
      
      for(int isrc =0 ; isrc<njecmcmx; isrc++){
	
        double sup = 1.0 ;
	
		if((isrc>0)&&(isrc<=nsrc)){
		  
		  JetCorrectionUncertainty *jecUnc = vsrcAK8[isrc-1];
		  jecUnc->setJetEta(pfjetAK8_4v_jecor.Eta());
		  jecUnc->setJetPt(pfjetAK8_4v_jecor.Pt());
		  
		  sup += jecUnc->getUncertainty(true);         
		  if(isrc==1){ PFJetAK8_jesup_AbsoluteStat[nPFJetAK8] = sup; }
		  if(isrc==2){ PFJetAK8_jesup_AbsoluteScale[nPFJetAK8] = sup; }
		  if(isrc==3){ PFJetAK8_jesup_AbsoluteMPFBias[nPFJetAK8] = sup; }
		  if(isrc==4){ PFJetAK8_jesup_FlavorQCD[nPFJetAK8] = sup; }
		  if(isrc==5){ PFJetAK8_jesup_Fragmentation[nPFJetAK8] = sup; }
		  if(isrc==6){ PFJetAK8_jesup_PileUpDataMC[nPFJetAK8] = sup; }
		  if(isrc==7){ PFJetAK8_jesup_PileUpPtBB[nPFJetAK8] = sup; }
		  if(isrc==8){ PFJetAK8_jesup_PileUpPtEC1[nPFJetAK8] = sup; }
		  if(isrc==9){ PFJetAK8_jesup_PileUpPtEC2[nPFJetAK8] = sup; }
		  if(isrc==10){ PFJetAK8_jesup_PileUpPtRef[nPFJetAK8] = sup; }
		  if(isrc==11){ PFJetAK8_jesup_RelativeFSR[nPFJetAK8] = sup; }
		  if(isrc==12){ PFJetAK8_jesup_RelativeJEREC1[nPFJetAK8] = sup; }
		  if(isrc==13){ PFJetAK8_jesup_RelativeJEREC2[nPFJetAK8] = sup; }
		  if(isrc==14){ PFJetAK8_jesup_RelativePtBB[nPFJetAK8] = sup; }
		  if(isrc==15){ PFJetAK8_jesup_RelativePtEC1[nPFJetAK8] = sup; }
		  if(isrc==16){ PFJetAK8_jesup_RelativePtEC2[nPFJetAK8] = sup; }
		  if(isrc==17){ PFJetAK8_jesup_RelativeBal[nPFJetAK8] = sup; }
		  if(isrc==18){ PFJetAK8_jesup_RelativeSample[nPFJetAK8] = sup; }
		  if(isrc==19){ PFJetAK8_jesup_RelativeStatEC[nPFJetAK8] = sup; }
		  if(isrc==20){ PFJetAK8_jesup_RelativeStatFSR[nPFJetAK8] = sup; }
		  if(isrc==21){ PFJetAK8_jesup_SinglePionECAL[nPFJetAK8] = sup; }
		  if(isrc==22){ PFJetAK8_jesup_SinglePionHCAL[nPFJetAK8] = sup; }
		  if(isrc==23){ PFJetAK8_jesup_TimePtEta[nPFJetAK8] = sup; }
		  if(isrc==24){ PFJetAK8_jesup_Total[nPFJetAK8] = sup; }
		
		}
		else if(isrc>nsrc){
		  
		  JetCorrectionUncertainty *jecUnc = vsrcAK8[isrc-1-nsrc];
		  jecUnc->setJetEta(pfjetAK8_4v_jecor.Eta());
		  jecUnc->setJetPt(pfjetAK8_4v_jecor.Pt());
		  
		  sup -= jecUnc->getUncertainty(false);
		  if(isrc==(nsrc+1)){ PFJetAK8_jesdn_AbsoluteStat[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+2)){ PFJetAK8_jesdn_AbsoluteScale[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+3)){ PFJetAK8_jesdn_AbsoluteMPFBias[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+4)){ PFJetAK8_jesdn_FlavorQCD[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+5)){ PFJetAK8_jesdn_Fragmentation[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+6)){ PFJetAK8_jesdn_PileUpDataMC[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+7)){ PFJetAK8_jesdn_PileUpPtBB[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+8)){ PFJetAK8_jesdn_PileUpPtEC1[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+9)){ PFJetAK8_jesdn_PileUpPtEC2[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+10)){ PFJetAK8_jesdn_PileUpPtRef[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+11)){ PFJetAK8_jesdn_RelativeFSR[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+12)){ PFJetAK8_jesdn_RelativeJEREC1[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+13)){ PFJetAK8_jesdn_RelativeJEREC2[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+14)){ PFJetAK8_jesdn_RelativePtBB[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+15)){ PFJetAK8_jesdn_RelativePtEC1[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+16)){ PFJetAK8_jesdn_RelativePtEC2[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+17)){ PFJetAK8_jesdn_RelativeBal[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+18)){ PFJetAK8_jesdn_RelativeSample[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+19)){ PFJetAK8_jesdn_RelativeStatEC[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+20)){ PFJetAK8_jesdn_RelativeStatFSR[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+21)){ PFJetAK8_jesdn_SinglePionECAL[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+22)){ PFJetAK8_jesdn_SinglePionHCAL[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+23)){ PFJetAK8_jesdn_TimePtEta[nPFJetAK8] = sup; }
		  if(isrc==(nsrc+24)){ PFJetAK8_jesdn_Total[nPFJetAK8] = sup; }
		}
		
      }
      
      // Jet id //
      if(store_jet_id_variables && ak8jet.isPFJet()){
		   
      PFJetAK8_CHF[nPFJetAK8] = ak8jet.chargedHadronEnergyFraction();
      PFJetAK8_NHF[nPFJetAK8] = ak8jet.neutralHadronEnergyFraction();
      PFJetAK8_CEMF[nPFJetAK8] = ak8jet.chargedEmEnergyFraction();
      PFJetAK8_NEMF[nPFJetAK8] = ak8jet.neutralEmEnergyFraction();
      PFJetAK8_MUF[nPFJetAK8] = ak8jet.muonEnergyFraction();
      PFJetAK8_PHF[nPFJetAK8] = ak8jet.photonEnergyFraction();
      PFJetAK8_EEF[nPFJetAK8] = ak8jet.electronEnergyFraction();
      PFJetAK8_HFHF[nPFJetAK8] = ak8jet.HFHadronEnergyFraction();
      
      PFJetAK8_CHM[nPFJetAK8] = ak8jet.chargedHadronMultiplicity();
      PFJetAK8_NHM[nPFJetAK8] = ak8jet.neutralHadronMultiplicity();
      PFJetAK8_MUM[nPFJetAK8] = ak8jet.muonMultiplicity();
      PFJetAK8_PHM[nPFJetAK8] = ak8jet.photonMultiplicity();
      PFJetAK8_EEM[nPFJetAK8] = ak8jet.electronMultiplicity();
      PFJetAK8_HFHM[nPFJetAK8] = ak8jet.HFHadronMultiplicity();
      
      PFJetAK8_Chcons[nPFJetAK8] = ak8jet.chargedMultiplicity();
      PFJetAK8_Neucons[nPFJetAK8] = ak8jet.neutralMultiplicity();
      
	  }
      
      JetIDVars idvars; 
      idvars.NHF = (ak8jet.isPFJet())?ak8jet.neutralHadronEnergyFraction():1.;
      idvars.NEMF = (ak8jet.isPFJet())?ak8jet.neutralEmEnergyFraction():1;
      idvars.MUF = (ak8jet.isPFJet())?ak8jet.muonEnergyFraction():1;
      idvars.CHF = (ak8jet.isPFJet())?ak8jet.chargedHadronEnergyFraction():1;
      idvars.CEMF = (ak8jet.isPFJet())?ak8jet.chargedEmEnergyFraction():1;
      idvars.NumConst = (ak8jet.isPFJet())?(ak8jet.chargedMultiplicity()+ak8jet.neutralMultiplicity()):1;
      idvars.NumNeutralParticle = (ak8jet.isPFJet())?ak8jet.neutralMultiplicity():1;
      idvars.CHM = (ak8jet.isPFJet())?ak8jet.chargedHadronMultiplicity():1;
      
      PFJetAK8_jetID[nPFJetAK8] = getJetID(idvars,"PUPPI",year,PFJetAK8_eta[nPFJetAK8],false,isUltraLegacy,isRun3);
      PFJetAK8_jetID_tightlepveto[nPFJetAK8] = getJetID(idvars,"PUPPI",year,PFJetAK8_eta[nPFJetAK8],true,isUltraLegacy,isRun3);  
      
      // veto map //
      
      PFJetAK8_jetveto_Flag[nPFJetAK8] = PFJetAK8_jetveto_eep_Flag[nPFJetAK8] = false;
      
	  PFJetAK8_jetveto_Flag[nPFJetAK8] = Assign_JetVeto(pfjetAK8_4v_jecor,PFJetAK8_jetID[nPFJetAK8],idvars,muons,h_jetvetomap);
	  if(year=="2022EE") { PFJetAK8_jetveto_eep_Flag[nPFJetAK8] = Assign_JetVeto(pfjetAK8_4v_jecor,PFJetAK8_jetID[nPFJetAK8],idvars,muons,h_jetvetomap_eep,30.); }
	   
	  // number of B and C hadrons //
	 
	  PFJetAK8_nBHadrons[nPFJetAK8] = int(ak8jet.jetFlavourInfo().getbHadrons().size());
	  PFJetAK8_nCHadrons[nPFJetAK8] = int(ak8jet.jetFlavourInfo().getcHadrons().size());
     
      // subjet info & classical observables //
     
      PFJetAK8_sub1pt[nPFJetAK8] = PFJetAK8_sub1eta[nPFJetAK8] = PFJetAK8_sub1phi[nPFJetAK8] = PFJetAK8_sub1mass[nPFJetAK8] = PFJetAK8_sub1btag[nPFJetAK8] = -100;              
      PFJetAK8_sub2pt[nPFJetAK8] = PFJetAK8_sub2eta[nPFJetAK8] = PFJetAK8_sub2phi[nPFJetAK8] = PFJetAK8_sub2mass[nPFJetAK8] = PFJetAK8_sub2btag[nPFJetAK8] = -100;                                                        
      PFJetAK8_sdmass[nPFJetAK8] = PFJetAK8_tau1[nPFJetAK8] = PFJetAK8_tau2[nPFJetAK8] = PFJetAK8_tau3[nPFJetAK8] = -100;                                                                      
      
      if(isSoftDrop){
	
		PFJetAK8_tau1[nPFJetAK8] = ak8jet.userFloat(Nsubjettiness_tau1);
		PFJetAK8_tau2[nPFJetAK8] = ak8jet.userFloat(Nsubjettiness_tau2);
		PFJetAK8_tau3[nPFJetAK8] = ak8jet.userFloat(Nsubjettiness_tau3);
		
		PFJetAK8_sdmass[nPFJetAK8] = (ak8jet.groomedMass(subjets) > 0)? ak8jet.groomedMass(subjets) : 0;
		
		for(unsigned int isub=0; isub<((ak8jet.subjets(subjets)).size()); isub++){
		
			const auto ak8subjet = (ak8jet.subjets(subjets))[isub];
	    
			if(isub==0){
				PFJetAK8_sub1pt[nPFJetAK8] = ak8subjet->correctedP4("Uncorrected").pt();
				PFJetAK8_sub1eta[nPFJetAK8] = ak8subjet->eta();
				PFJetAK8_sub1phi[nPFJetAK8] = ak8subjet->phi();
				PFJetAK8_sub1mass[nPFJetAK8] = ak8subjet->correctedP4("Uncorrected").mass();	 
				PFJetAK8_sub1JEC[nPFJetAK8] = ak8subjet->pt()*1./ak8subjet->correctedP4("Uncorrected").pt();
				PFJetAK8_sub1btag[nPFJetAK8] = ak8subjet->bDiscriminator("pfDeepCSVJetTags:probb")+ak8subjet->bDiscriminator("pfDeepCSVJetTags:probbb");
			}
			else if(isub==1){
				PFJetAK8_sub2pt[nPFJetAK8] = ak8subjet->correctedP4("Uncorrected").pt();
				PFJetAK8_sub2eta[nPFJetAK8] = ak8subjet->eta();
				PFJetAK8_sub2phi[nPFJetAK8] = ak8subjet->phi();
				PFJetAK8_sub2mass[nPFJetAK8] = ak8subjet->correctedP4("Uncorrected").mass();	
				PFJetAK8_sub2JEC[nPFJetAK8] = ak8subjet->pt()*1./ak8subjet->correctedP4("Uncorrected").pt(); 
				PFJetAK8_sub2btag[nPFJetAK8] = ak8subjet->bDiscriminator("pfDeepCSVJetTags:probb")+ak8subjet->bDiscriminator("pfDeepCSVJetTags:probbb");
			}
		}
		
	
      }//isSoftDrop
      
      // Storing 4-momenta of jet constituents//
      if(store_fatjet_constituents && nGenJetAK8<njetconsmax)     {
        for(unsigned int ic = 0 ; ic < ak8jet.numberOfSourceCandidatePtrs() ; ++ic) {  
          if(ak8jet.sourceCandidatePtr(ic).isNonnull() && ak8jet.sourceCandidatePtr(ic).isAvailable()){
            
            if(nPFJetAK8_cons>= nconsmax) break;	
            const reco::Candidate* jcand = ak8jet.sourceCandidatePtr(ic).get();
            PFJetAK8_cons_pt[nPFJetAK8_cons] = jcand->pt();
            PFJetAK8_cons_eta[nPFJetAK8_cons] = jcand->eta();
            PFJetAK8_cons_phi[nPFJetAK8_cons] = jcand->phi();
            PFJetAK8_cons_mass[nPFJetAK8_cons] = jcand->mass();
            PFJetAK8_cons_pdgId[nPFJetAK8_cons] = jcand->pdgId();
            PFJetAK8_cons_jetIndex[nPFJetAK8_cons] = nPFJetAK8;   
            nPFJetAK8_cons++;
          }
        }
      }
            
      // end of candidate storage //
            
      nPFJetAK8++;	
      if(nPFJetAK8 >= njetmxAK8) { break;}
      
    }
  }
  
  // AK4 jets //
  
  nPFJetAK4 = 0;
  edm::Handle<edm::View<pat::Jet>> pfjetAK4s;
  iEvent.getByToken(tok_pfjetAK4s_, pfjetAK4s);
  
  if(pfjetAK4s.isValid() && store_ak4jets){
    
  for (unsigned jet = 0; jet< pfjetAK4s->size(); jet++) {
      
	const auto &ak4jet = (*pfjetAK4s)[jet];
    TLorentzVector pfjetAK4_4v(ak4jet.correctedP4("Uncorrected").px(),ak4jet.correctedP4("Uncorrected").py(),ak4jet.correctedP4("Uncorrected").pz(), ak4jet.correctedP4("Uncorrected").energy());
    
    if(subtractLepton_fromAK4){
		pfjetAK4_4v = LeptonJet_subtraction(tlvmu,ak4jet,pfjetAK4_4v);
		pfjetAK4_4v = LeptonJet_subtraction(tlvel,ak4jet,pfjetAK4_4v);
	}
    
    double tmprecpt = pfjetAK4_4v.Pt();
    
    double total_cor =1;
    Read_JEC(total_cor,tmprecpt,pfjetAK4_4v,Rho,isData,ak4jet,jecL1FastAK4,jecL2RelativeAK4,jecL3AbsoluteAK4,jecL2L3ResidualAK4);  
    PFJetAK4_JEC[nPFJetAK4] = total_cor;
        
    // basic selection conditions on JEC-corrected jets //
    if(tmprecpt<min_pt_AK4jet) continue;
    if(abs(ak4jet.eta())>max_eta) continue;
    
    // JEC corrected 4-momentum //
    TLorentzVector pfjetAK4_4v_jecor;
    pfjetAK4_4v_jecor = pfjetAK4_4v * PFJetAK4_JEC[nPFJetAK4];
       
    PFJetAK4_pt[nPFJetAK4] = 	pfjetAK4_4v.Pt();
    PFJetAK4_eta[nPFJetAK4] = 	pfjetAK4_4v.Eta();
    PFJetAK4_y[nPFJetAK4] = pfjetAK4_4v.Rapidity();
    PFJetAK4_phi[nPFJetAK4] = pfjetAK4_4v.Phi();
    PFJetAK4_mass[nPFJetAK4] = pfjetAK4_4v.M(); 
    PFJetAK4_area[nPFJetAK4] = ak4jet.jetArea();
    
    //cout<<"pt: "<<ak4jet.pt()<<" uncorrected pt "<<pfjetAK4_4v.Pt()<<" textJEC*uncorpt "<<PFJetAK4_JEC[nPFJetAK4]*PFJetAK4_pt[nPFJetAK4]<<endl;
    
    // JER //
     
    if(isMC){
		
		TLorentzVector tmp4v;
		tmp4v = pfjetAK4_4v_jecor;
		
		vector<double> SFs;
      	Read_JER(mPtResoFileAK4, mPtSFFileAK4, tmprecpt, tmp4v, Rho, genjetAK4s, 0.5*0.4, SFs);
      	
      	PFJetAK4_reso[nPFJetAK4] = SFs[0];
      	PFJetAK4_resoup[nPFJetAK4] = SFs[1];
      	PFJetAK4_resodn[nPFJetAK4] = SFs[2];
		
    }//isMC
      
     // JES uncertainty //
      
    for(int isrc =0 ; isrc<njecmcmx; isrc++){
	
		double sup = 1.0 ;
	
		if((isrc>0)&&(isrc<=nsrc)){
	  
			JetCorrectionUncertainty *jecUnc = vsrc[isrc-1];
			jecUnc->setJetEta(pfjetAK4_4v_jecor.Eta());
			jecUnc->setJetPt(pfjetAK4_4v_jecor.Pt());
	  
			sup += jecUnc->getUncertainty(true);         
			if(isrc==1){ PFJetAK4_jesup_AbsoluteStat[nPFJetAK4] = sup; }
			if(isrc==2){ PFJetAK4_jesup_AbsoluteScale[nPFJetAK4] = sup; }
			if(isrc==3){ PFJetAK4_jesup_AbsoluteMPFBias[nPFJetAK4] = sup; }
			if(isrc==4){ PFJetAK4_jesup_FlavorQCD[nPFJetAK4] = sup; }
			if(isrc==5){ PFJetAK4_jesup_Fragmentation[nPFJetAK4] = sup; }
			if(isrc==6){ PFJetAK4_jesup_PileUpDataMC[nPFJetAK4] = sup; }
			if(isrc==7){ PFJetAK4_jesup_PileUpPtBB[nPFJetAK4] = sup; }
			if(isrc==8){ PFJetAK4_jesup_PileUpPtEC1[nPFJetAK4] = sup; }
			if(isrc==9){ PFJetAK4_jesup_PileUpPtEC2[nPFJetAK4] = sup; }
			if(isrc==10){ PFJetAK4_jesup_PileUpPtRef[nPFJetAK4] = sup; }
			if(isrc==11){ PFJetAK4_jesup_RelativeFSR[nPFJetAK4] = sup; }
			if(isrc==12){ PFJetAK4_jesup_RelativeJEREC1[nPFJetAK4] = sup; }
			if(isrc==13){ PFJetAK4_jesup_RelativeJEREC2[nPFJetAK4] = sup; }
			if(isrc==14){ PFJetAK4_jesup_RelativePtBB[nPFJetAK4] = sup; }
			if(isrc==15){ PFJetAK4_jesup_RelativePtEC1[nPFJetAK4] = sup; }
			if(isrc==16){ PFJetAK4_jesup_RelativePtEC2[nPFJetAK4] = sup; }
			if(isrc==17){ PFJetAK4_jesup_RelativeBal[nPFJetAK4] = sup; }
			if(isrc==18){ PFJetAK4_jesup_RelativeSample[nPFJetAK4] = sup; }
			if(isrc==19){ PFJetAK4_jesup_RelativeStatEC[nPFJetAK4] = sup; }
			if(isrc==20){ PFJetAK4_jesup_RelativeStatFSR[nPFJetAK4] = sup; }
			if(isrc==21){ PFJetAK4_jesup_SinglePionECAL[nPFJetAK4] = sup; }
			if(isrc==22){ PFJetAK4_jesup_SinglePionHCAL[nPFJetAK4] = sup; }
			if(isrc==23){ PFJetAK4_jesup_TimePtEta[nPFJetAK4] = sup; }
			if(isrc==24){ PFJetAK4_jesup_Total[nPFJetAK4] = sup; }
		}
	
		else if(isrc>nsrc){
	  
			JetCorrectionUncertainty *jecUnc = vsrc[isrc-1-nsrc];
		    jecUnc->setJetEta(pfjetAK4_4v_jecor.Eta());
		    jecUnc->setJetPt(pfjetAK4_4v_jecor.Pt());
	  
			sup -= jecUnc->getUncertainty(false);
			if(isrc==(nsrc+1)){ PFJetAK4_jesdn_AbsoluteStat[nPFJetAK4] = sup; }
			if(isrc==(nsrc+2)){ PFJetAK4_jesdn_AbsoluteScale[nPFJetAK4] = sup; }
			if(isrc==(nsrc+3)){ PFJetAK4_jesdn_AbsoluteMPFBias[nPFJetAK4] = sup; }
			if(isrc==(nsrc+4)){ PFJetAK4_jesdn_FlavorQCD[nPFJetAK4] = sup; }
			if(isrc==(nsrc+5)){ PFJetAK4_jesdn_Fragmentation[nPFJetAK4] = sup; }
			if(isrc==(nsrc+6)){ PFJetAK4_jesdn_PileUpDataMC[nPFJetAK4] = sup; }
			if(isrc==(nsrc+7)){ PFJetAK4_jesdn_PileUpPtBB[nPFJetAK4] = sup; }
			if(isrc==(nsrc+8)){ PFJetAK4_jesdn_PileUpPtEC1[nPFJetAK4] = sup; }
			if(isrc==(nsrc+9)){ PFJetAK4_jesdn_PileUpPtEC2[nPFJetAK4] = sup; }
			if(isrc==(nsrc+10)){ PFJetAK4_jesdn_PileUpPtRef[nPFJetAK4] = sup; }
			if(isrc==(nsrc+11)){ PFJetAK4_jesdn_RelativeFSR[nPFJetAK4] = sup; }
			if(isrc==(nsrc+12)){ PFJetAK4_jesdn_RelativeJEREC1[nPFJetAK4] = sup; }
			if(isrc==(nsrc+13)){ PFJetAK4_jesdn_RelativeJEREC2[nPFJetAK4] = sup; }
			if(isrc==(nsrc+14)){ PFJetAK4_jesdn_RelativePtBB[nPFJetAK4] = sup; }
			if(isrc==(nsrc+15)){ PFJetAK4_jesdn_RelativePtEC1[nPFJetAK4] = sup; }
			if(isrc==(nsrc+16)){ PFJetAK4_jesdn_RelativePtEC2[nPFJetAK4] = sup; }
			if(isrc==(nsrc+17)){ PFJetAK4_jesdn_RelativeBal[nPFJetAK4] = sup; }
			if(isrc==(nsrc+18)){ PFJetAK4_jesdn_RelativeSample[nPFJetAK4] = sup; }
			if(isrc==(nsrc+19)){ PFJetAK4_jesdn_RelativeStatEC[nPFJetAK4] = sup; }
			if(isrc==(nsrc+20)){ PFJetAK4_jesdn_RelativeStatFSR[nPFJetAK4] = sup; }
			if(isrc==(nsrc+21)){ PFJetAK4_jesdn_SinglePionECAL[nPFJetAK4] = sup; }
			if(isrc==(nsrc+22)){ PFJetAK4_jesdn_SinglePionHCAL[nPFJetAK4] = sup; }
			if(isrc==(nsrc+23)){ PFJetAK4_jesdn_TimePtEta[nPFJetAK4] = sup; }
			if(isrc==(nsrc+24)){ PFJetAK4_jesdn_Total[nPFJetAK4] = sup; }
			
		}
	
    }
      
    // JES uncertainty Ends //
    
    // Jet id //
      
    JetIDVars AK4idvars;
      
    AK4idvars.NHF = (ak4jet.isPFJet())?ak4jet.neutralHadronEnergyFraction():1;
    AK4idvars.NEMF = (ak4jet.isPFJet())?ak4jet.neutralEmEnergyFraction():1;
    AK4idvars.MUF = (ak4jet.isPFJet())?ak4jet.muonEnergyFraction():1;
    AK4idvars.CHF = (ak4jet.isPFJet())?ak4jet.chargedHadronEnergyFraction():1;
    AK4idvars.CEMF = (ak4jet.isPFJet())?ak4jet.chargedEmEnergyFraction():1;
    AK4idvars.NumConst = (ak4jet.isPFJet())?(ak4jet.chargedMultiplicity()+ak4jet.neutralMultiplicity()):1;
    AK4idvars.NumNeutralParticle = (ak4jet.isPFJet())?ak4jet.neutralMultiplicity():1;
    AK4idvars.CHM = (ak4jet.isPFJet())?ak4jet.chargedHadronMultiplicity():1;
     
    PFJetAK4_jetID[nPFJetAK4] = getJetID(AK4idvars,"CHS",year,PFJetAK4_eta[nPFJetAK4],false,isUltraLegacy,isRun3);
    PFJetAK4_jetID_tightlepveto[nPFJetAK4] = getJetID(AK4idvars,"CHS",year,PFJetAK4_eta[nPFJetAK4],true,isUltraLegacy,isRun3);
    
    // veto map //
      
    PFJetAK4_jetveto_Flag[nPFJetAK4] = PFJetAK4_jetveto_eep_Flag[nPFJetAK4] = false;
  
	PFJetAK4_jetveto_Flag[nPFJetAK4] = Assign_JetVeto(pfjetAK4_4v,PFJetAK4_jetID[nPFJetAK4],AK4idvars,muons,h_jetvetomap);
	if(year=="2022EE") { PFJetAK4_jetveto_eep_Flag[nPFJetAK4] = Assign_JetVeto(pfjetAK4_4v,PFJetAK4_jetID[nPFJetAK4],AK4idvars,muons,h_jetvetomap_eep,30.); }
	
	// flavor, QGL, PU ID //
		
    PFJetAK4_hadronflav[nPFJetAK4] = ak4jet.hadronFlavour();
    PFJetAK4_partonflav[nPFJetAK4] = ak4jet.partonFlavour();
      
    PFJetAK4_qgl[nPFJetAK4] = ak4jet.userFloat("qgLikelihood");//"QGTagger:qgLikelihood");
    PFJetAK4_PUID[nPFJetAK4] = ak4jet.userFloat("pileupJetId_fullDiscriminant");//"pileupJetId:fullDiscriminant");
    
    std::vector<reco::CandidatePtr> daughters(ak4jet.daughterPtrVector());
    std::sort(daughters.begin(), daughters.end(), [](const reco::CandidatePtr &p1, const reco::CandidatePtr &p2)
    { return p1->pt() > p2->pt(); });

    float sumptchg_kp3 = 0; float sumptchg_kp6 = 0; float sumptchg_k1 = 0;
    float chg_ptsum = 0;

    for (unsigned int i2 = 0; i2< daughters.size(); ++i2) {
        sumptchg_kp3 += pow(daughters[i2]->pt(),0.3)*daughters[i2]->charge();   
        sumptchg_kp6 += pow(daughters[i2]->pt(),0.6)*daughters[i2]->charge();    
        sumptchg_k1  += pow(daughters[i2]->pt(),1.0)*daughters[i2]->charge();   
        if(abs(daughters[i2]->charge())>0) {    chg_ptsum += daughters[i2]->pt(); }  
    }

	PFJetAK4_charge_kappa_0p3[nPFJetAK4] = (daughters.size()>0)? (sumptchg_kp3*1./pow(PFJetAK4_pt[nPFJetAK4],0.3)): 0;
	PFJetAK4_charge_kappa_0p6[nPFJetAK4] = (daughters.size()>0)? (sumptchg_kp6*1./pow(PFJetAK4_pt[nPFJetAK4],0.6)): 0;
	PFJetAK4_charge_kappa_1p0[nPFJetAK4] = (daughters.size()>0)? (sumptchg_k1 *1./PFJetAK4_pt[nPFJetAK4]): 0;
	
	PFJetAK4_charged_ptsum[nPFJetAK4] = chg_ptsum;
    
    // B tagging stuffs //
    
    PFJetAK4_btag_DeepCSV[nPFJetAK4] = ak4jet.bDiscriminator("pfDeepCSVJetTags:probb")+ak4jet.bDiscriminator("pfDeepCSVJetTags:probbb");
    
    PFJetAK4_btag_DeepFlav[nPFJetAK4] = ak4jet.bDiscriminator("pfDeepFlavourJetTags:probb") + ak4jet.bDiscriminator("pfDeepFlavourJetTags:probbb")+ak4jet.bDiscriminator("pfDeepFlavourJetTags:problepb");
   
    PFJetAK4_btagDeepFlavB[nPFJetAK4] = ak4jet.bDiscriminator("pfDeepFlavourJetTags:probb") + ak4jet.bDiscriminator("pfDeepFlavourJetTags:probbb")+ak4jet.bDiscriminator("pfDeepFlavourJetTags:problepb");
    PFJetAK4_btagDeepFlavCvB[nPFJetAK4] = (ak4jet.bDiscriminator("pfDeepFlavourJetTags:probc"))*1./(ak4jet.bDiscriminator("pfDeepFlavourJetTags:probc") + ak4jet.bDiscriminator("pfDeepFlavourJetTags:probb") + ak4jet.bDiscriminator("pfDeepFlavourJetTags:probbb")+ak4jet.bDiscriminator("pfDeepFlavourJetTags:problepb"));
    PFJetAK4_btagDeepFlavCvL[nPFJetAK4] = (ak4jet.bDiscriminator("pfDeepFlavourJetTags:probc"))*1./(ak4jet.bDiscriminator("pfDeepFlavourJetTags:probc") + ak4jet.bDiscriminator("pfDeepFlavourJetTags:probuds") + ak4jet.bDiscriminator("pfDeepFlavourJetTags:probg"));
	PFJetAK4_btagDeepFlavQG[nPFJetAK4] = (ak4jet.bDiscriminator("pfDeepFlavourJetTags:probg"))*1./(ak4jet.bDiscriminator("pfDeepFlavourJetTags:probg") + ak4jet.bDiscriminator("pfDeepFlavourJetTags:probuds"));

    PFJetAK4_btagPNetB[nPFJetAK4] = ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:BvsAll");
    PFJetAK4_btagPNetCvNotB[nPFJetAK4] = ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probc")*1./(1.- ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probb"));
    PFJetAK4_btagPNetCvB[nPFJetAK4] = ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:CvsB");
    PFJetAK4_btagPNetCvL[nPFJetAK4] = ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:CvsL");
    PFJetAK4_btagPNetQG[nPFJetAK4] = ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:QvsG");
    
    PFJetAK4_btagRobustParTAK4B[nPFJetAK4] =   (ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probb") + ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probbb")+ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:problepb"))>0 ? (ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probb") + ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probbb")+ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:problepb")):-10;
    PFJetAK4_btagRobustParTAK4CvB[nPFJetAK4] = (ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probc")+ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probb") + ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probbb")+ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:problepb"))>0 ? (ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probc")*1./(ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probc")+ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probb") + ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probbb")+ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:problepb"))) : -10;
    PFJetAK4_btagRobustParTAK4CvL[nPFJetAK4] = (ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probc")+ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probuds")+ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probg"))>0 ? (ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probc")*1./(ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probc")+ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probuds")+ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probg"))) : -10;
    PFJetAK4_btagRobustParTAK4QG[nPFJetAK4] =  (ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probg")+ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probuds"))>0 ? (ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probg")*1./(ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probg") + ak4jet.bDiscriminator("pfParticleTransformerAK4JetTags:probuds"))) : -10;

	if(read_btagSF){

		BTagEntry::JetFlavor btv_flav;
		if(abs(PFJetAK4_hadronflav[nPFJetAK4])==5){ btv_flav = BTagEntry::FLAV_B; }
		else if (abs(PFJetAK4_hadronflav[nPFJetAK4])==4){ btv_flav = BTagEntry::FLAV_C; }
		else { btv_flav = BTagEntry::FLAV_UDSG; }
    
		PFJetAK4_btag_DeepFlav_SF[nPFJetAK4] = reader_deepflav.eval_auto_bounds("central",btv_flav,fabs(pfjetAK4_4v_jecor.Eta()),pfjetAK4_4v_jecor.Pt()); 
		PFJetAK4_btag_DeepFlav_SF_up[nPFJetAK4] = reader_deepflav.eval_auto_bounds("up",btv_flav,fabs(pfjetAK4_4v_jecor.Eta()),pfjetAK4_4v_jecor.Pt());
		PFJetAK4_btag_DeepFlav_SF_dn[nPFJetAK4] = reader_deepflav.eval_auto_bounds("down",btv_flav,fabs(pfjetAK4_4v_jecor.Eta()),pfjetAK4_4v_jecor.Pt());
	}
	
	// b jet energy regression //
	
	PFJetAK4_PNetRegPtRawCorr[nPFJetAK4] = ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptcorr");
	PFJetAK4_PNetRegPtRawCorrNeutrino[nPFJetAK4] = ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptnu");
	PFJetAK4_PNetRegPtRawRes[nPFJetAK4]  = (ak4jet.pt()>15. && fabs(ak4jet.eta())<2.5)? (0.5*(ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptreshigh") - ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptreslow"))):(0.5*(ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiForwardJetTags:ptreshigh") - ak4jet.bDiscriminator("pfParticleNetFromMiniAODAK4PuppiForwardJetTags:ptreslow")));
	 
	// Note that btag SF is derived after applying JEC //
     
    nPFJetAK4++;	
    if(nPFJetAK4 >= njetmx) { break;}
    
  }
  
  }
  
  // Tau leptons //
 
  nTau = 0;
  
  if(store_taus){
  
  for(const auto& tau1 : iEvent.get(tok_taus_) ) {

	if (tau1.pt()<min_pt_tau) continue;
    if(fabs(tau1.eta())>max_eta_tau) continue;

    Tau_pt[nTau] = tau1.pt();
    Tau_eta[nTau] = tau1.eta();
    Tau_phi[nTau] = tau1.phi();
    Tau_e[nTau] = tau1.energy();
    Tau_charge[nTau] = tau1.charge();
    Tau_isPF[nTau] = tau1.isPFTau();
    Tau_dxy[nTau] = tau1.dxy();
    Tau_dz[nTau] = (tau1.leadTrack().isNonnull())?tau1.leadTrack()->dz():0;
    
    // Id & iso variables //
   
    // DeepTau 2017v2p1 //
    
    if(store_tau_id_variables){
    
    Tau_jetiso_deeptau2017v2p1_raw[nTau] = tau1.tauID("byDeepTau2017v2p1VSjetraw");
    Tau_jetiso_deeptau2017v2p1[nTau] = (0 + (int(tau1.tauID("byVVVLooseDeepTau2017v2p1VSjet"))) + (1<<(1*int(tau1.tauID("byVVLooseDeepTau2017v2p1VSjet")))) + (1<<(2*int(tau1.tauID("byVLooseDeepTau2017v2p1VSjet")))) + (1<<(3*int(tau1.tauID("byLooseDeepTau2017v2p1VSjet")))) + (1<<(4*int(tau1.tauID("byMediumDeepTau2017v2p1VSjet")))) + (1<<(5*int(tau1.tauID("byTightDeepTau2017v2p1VSjet")))) + (1<<(6*int(tau1.tauID("byVTightDeepTau2017v2p1VSjet")))) + (1<<(7*int(tau1.tauID("byVVTightDeepTau2017v2p1VSjet")))) );
	
    Tau_eiso_deeptau2017v2p1_raw[nTau] = tau1.tauID("byDeepTau2017v2p1VSeraw");
    Tau_eiso_deeptau2017v2p1[nTau] = (0 + (int(tau1.tauID("byVVVLooseDeepTau2017v2p1VSe"))) + (1<<(1*int(tau1.tauID("byVVLooseDeepTau2017v2p1VSe")))) + (1<<(2*int(tau1.tauID("byVLooseDeepTau2017v2p1VSe")))) + (1<<(3*int(tau1.tauID("byLooseDeepTau2017v2p1VSe")))) + (1<<(4*int(tau1.tauID("byMediumDeepTau2017v2p1VSe")))) + (1<<(5*int(tau1.tauID("byTightDeepTau2017v2p1VSe")))) + (1<<(6*int(tau1.tauID("byVTightDeepTau2017v2p1VSe")))) + (1<<(7*int(tau1.tauID("byVVTightDeepTau2017v2p1VSe")))) );

    Tau_muiso_deeptau2017v2p1_raw[nTau] = tau1.tauID("byDeepTau2017v2p1VSmuraw");
    Tau_muiso_deeptau2017v2p1[nTau] = (0 + (int(tau1.tauID("byVLooseDeepTau2017v2p1VSmu"))) + (1<<(1*int(tau1.tauID("byLooseDeepTau2017v2p1VSmu")))) + (1<<(2*int(tau1.tauID("byMediumDeepTau2017v2p1VSmu")))) + (1<<(3*int(tau1.tauID("byTightDeepTau2017v2p1VSmu")))) );
    
	}
    
    // DeepTau 2018v2p5 //
    
    Tau_jetiso_deeptau2018v2p5_raw[nTau] = tau1.tauID("byDeepTau2018v2p5VSjetraw");
    Tau_jetiso_deeptau2018v2p5[nTau] = (0 + (int(tau1.tauID("byVVVLooseDeepTau2018v2p5VSjet"))) + (1<<(1*int(tau1.tauID("byVVLooseDeepTau2018v2p5VSjet")))) + (1<<(2*int(tau1.tauID("byVLooseDeepTau2018v2p5VSjet")))) + (1<<(3*int(tau1.tauID("byLooseDeepTau2018v2p5VSjet")))) + (1<<(4*int(tau1.tauID("byMediumDeepTau2018v2p5VSjet")))) + (1<<(5*int(tau1.tauID("byTightDeepTau2018v2p5VSjet")))) + (1<<(6*int(tau1.tauID("byVTightDeepTau2018v2p5VSjet")))) + (1<<(7*int(tau1.tauID("byVVTightDeepTau2018v2p5VSjet")))) );
	
    Tau_eiso_deeptau2018v2p5_raw[nTau] = tau1.tauID("byDeepTau2018v2p5VSeraw");
    Tau_eiso_deeptau2018v2p5[nTau] = (0 + (int(tau1.tauID("byVVVLooseDeepTau2018v2p5VSe"))) + (1<<(1*int(tau1.tauID("byVVLooseDeepTau2018v2p5VSe")))) + (1<<(2*int(tau1.tauID("byVLooseDeepTau2018v2p5VSe")))) + (1<<(3*int(tau1.tauID("byLooseDeepTau2018v2p5VSe")))) + (1<<(4*int(tau1.tauID("byMediumDeepTau2018v2p5VSe")))) + (1<<(5*int(tau1.tauID("byTightDeepTau2018v2p5VSe")))) + (1<<(6*int(tau1.tauID("byVTightDeepTau2018v2p5VSe")))) + (1<<(7*int(tau1.tauID("byVVTightDeepTau2018v2p5VSe")))) );

    Tau_muiso_deeptau2018v2p5_raw[nTau] = tau1.tauID("byDeepTau2018v2p5VSmuraw");
    Tau_muiso_deeptau2018v2p5[nTau] = (0 + (int(tau1.tauID("byVLooseDeepTau2018v2p5VSmu"))) + (1<<(1*int(tau1.tauID("byLooseDeepTau2018v2p5VSmu")))) + (1<<(2*int(tau1.tauID("byMediumDeepTau2018v2p5VSmu")))) + (1<<(3*int(tau1.tauID("byTightDeepTau2018v2p5VSmu")))) );
    
    //cout<<"pnet "<<tau1.tauID("byUTagCHSVSeraw")<<endl;
    
    if(store_tau_id_variables){
		
		 /*
		Tau_eiso2018_raw[nTau] = tau1.tauID("againstElectronMVA6Raw2018");
		Tau_eiso2018[nTau] = (0 + (int(tau1.tauID("againstElectronVLooseMVA62018"))) + (1<<(1*int(tau1.tauID("againstElectronLooseMVA62018")))) + (1<<(2*int(tau1.tauID("againstElectronMediumMVA62018")))) + (1<<(3*int(tau1.tauID("againstElectronTightMVA62018")))) + (1<<(4*int(tau1.tauID("againstElectronVTightMVA62018")))));
		*/
		
		if(!tau1.leadTrack().isNull()){
			Tau_leadtrkdxy[nTau] = tau1.leadTrack()->dxy(vertex.position());
			Tau_leadtrkdz[nTau] = tau1.leadTrack()->dz(vertex.position());
		}
    
		Tau_decayMode[nTau] = tau1.decayMode();
		Tau_decayModeinding[nTau] = tau1.tauID("decayModeFinding");
		Tau_decayModeindingNewDMs[nTau] = tau1.tauID("decayModeFindingNewDMs");

		Tau_rawiso[nTau] = tau1.tauID("byCombinedIsolationDeltaBetaCorrRaw3Hits");
		Tau_rawisodR03[nTau] = (tau1.tauID("chargedIsoPtSumdR03")+TMath::Max(0.,tau1.tauID("neutralIsoPtSumdR03")-0.072*tau1.tauID("puCorrPtSum")));
		Tau_puCorr[nTau] = tau1.tauID("puCorrPtSum");

		if(!tau1.leadChargedHadrCand().isNull()){

		//  Tau_dxy[nTau] = tau1.leadChargedHadrCand()->dxy(vertex.position());
		//  Tau_dz[nTau] = tau1.leadChargedHadrCand()->dz(vertex.position());
	
			Tau_leadtrkpt[nTau] = tau1.leadChargedHadrCand()->pt();
			Tau_leadtrketa[nTau] = tau1.leadChargedHadrCand()->eta();
			Tau_leadtrkphi[nTau] = tau1.leadChargedHadrCand()->phi();
		}
    
    }//store_tau_id_variables
    
    if (++nTau>=njetmx) break;

  }
  
  }//store_taus
  
  // Photons //
  
  nPhoton = 0;
  edm::Handle<edm::View<pat::Photon>> photons;
  iEvent.getByToken(tok_photons_, photons);
  
  // for raw MVA score info //
  edm::Handle <edm::ValueMap <float> > mvaPhoID_FallV2_raw;
  iEvent.getByToken(tok_mvaPhoID_FallV2_raw, mvaPhoID_FallV2_raw);

  if(store_photons){

  for(const auto& gamma1 : photons->ptrs() ) {

	if(gamma1->pt() < min_pt_gamma) continue;

    Photon_e[nPhoton] = gamma1->energy();
    Photon_eta[nPhoton] = gamma1->eta();
    Photon_phi[nPhoton] = gamma1->phi();
  
	// MVA id //

	Photon_mvaid_RunIIIWinter22V1_WP90[nPhoton] =  gamma1->photonID(mPhoID_RunIIIWinter22V1_WP90);
	Photon_mvaid_RunIIIWinter22V1_WP80[nPhoton] =  gamma1->photonID(mPhoID_RunIIIWinter22V1_WP80);

    Photon_mvaid_Fall17V2_WP90[nPhoton] =  gamma1->photonID(mPhoID_FallV2_WP90);
    Photon_mvaid_Fall17V2_WP80[nPhoton] = gamma1->photonID(mPhoID_FallV2_WP80);
//    Photon_mvaid_Fall17V2_raw[nPhoton] = (*mvaPhoID_FallV2_raw)[gamma1];
	
	Photon_mvaid_Spring16V1_WP90[nPhoton] = gamma1->photonID(mPhoID_SpringV1_WP90);
    Photon_mvaid_Spring16V1_WP80[nPhoton] = gamma1->photonID(mPhoID_SpringV1_WP80);

	// Isolation variables //

	if(store_photon_id_variables){

	Photon_e1by9[nPhoton] = gamma1->maxEnergyXtal()/max(float(1),gamma1->e3x3());
    if (gamma1->hasConversionTracks()) { Photon_e1by9[nPhoton] *= -1; }
	Photon_e9by25[nPhoton] = gamma1->r9();
    Photon_hadbyem[nPhoton] = gamma1->hadronicOverEm();
    Photon_ietaieta[nPhoton] = gamma1->sigmaIetaIeta();

    Photon_trkiso[nPhoton] = gamma1->trkSumPtSolidConeDR04();
    Photon_emiso[nPhoton] = gamma1->ecalRecHitSumEtConeDR04();
    Photon_hadiso[nPhoton] = gamma1->hcalTowerSumEtConeDR04();
    Photon_phoiso[nPhoton] = gamma1->photonIso() ;
    Photon_chhadiso[nPhoton] = gamma1->chargedHadronIso();
    Photon_neuhadiso[nPhoton] = gamma1->neutralHadronIso();
    
	}
    
    if (++nPhoton>=njetmx) break;

  }
  
  }//store_photons
       
  //cout<<"done!"<<endl;
  
  T2->Fill(); // filling the tree used to get sumofweights
  
  // Skimming condition //
  
  //if(nPFJetAK8>=1 && (nMuon+nElectron)>=1){ // for X->YH->bbWW (1- or 2-lepton) analysis
  if( int(tlvmu.size()+tlvel.size())<1 && nPFJetAK4>=2 ){ // for X->YH->4b analysis
	T1->Fill(); // filling the main tree
  }
  
  // End of skimming 
  
  logMemoryUsage("After processing event");
  
}


// ------------ method called once each job just before starting event loop  ------------
void 
Leptop::beginJob()
{
  
  Nevt = 0;
  
  ////JEC /////
  
  cout<<"mJECL1FastFileAK8 "<<mJECL1FastFileAK4<<endl;
  
  L1FastAK4       = new JetCorrectorParameters(mJECL1FastFileAK4.c_str());
  L2RelativeAK4   = new JetCorrectorParameters(mJECL2RelativeFileAK4.c_str());
  L3AbsoluteAK4   = new JetCorrectorParameters(mJECL3AbsoluteFileAK4.c_str());
  L2L3ResidualAK4 = new JetCorrectorParameters(mJECL2L3ResidualFileAK4.c_str());
  
  vecL1FastAK4.push_back(*L1FastAK4);
  vecL2RelativeAK4.push_back(*L2RelativeAK4);
  vecL3AbsoluteAK4.push_back(*L3AbsoluteAK4);
  vecL2L3ResidualAK4.push_back(*L2L3ResidualAK4);
  
  jecL1FastAK4       = new FactorizedJetCorrector(vecL1FastAK4);
  jecL2RelativeAK4   = new FactorizedJetCorrector(vecL2RelativeAK4);
  jecL3AbsoluteAK4   = new FactorizedJetCorrector(vecL3AbsoluteAK4);
  jecL2L3ResidualAK4 = new FactorizedJetCorrector(vecL2L3ResidualAK4);
  
  cout<<"mJECL1FastFileAK8 "<<mJECL1FastFileAK8<<endl;
  
  L1FastAK8       = new JetCorrectorParameters(mJECL1FastFileAK8.c_str());
  L2RelativeAK8   = new JetCorrectorParameters(mJECL2RelativeFileAK8.c_str());
  L3AbsoluteAK8   = new JetCorrectorParameters(mJECL3AbsoluteFileAK8.c_str());
  L2L3ResidualAK8 = new JetCorrectorParameters(mJECL2L3ResidualFileAK8.c_str());
  
  vecL1FastAK8.push_back(*L1FastAK8);
  vecL2RelativeAK8.push_back(*L2RelativeAK8);
  vecL3AbsoluteAK8.push_back(*L3AbsoluteAK8);
  vecL2L3ResidualAK8.push_back(*L2L3ResidualAK8);
  
  jecL1FastAK8       = new FactorizedJetCorrector(vecL1FastAK8);
  jecL2RelativeAK8   = new FactorizedJetCorrector(vecL2RelativeAK8);
  jecL3AbsoluteAK8   = new FactorizedJetCorrector(vecL3AbsoluteAK8);
  jecL2L3ResidualAK8 = new FactorizedJetCorrector(vecL2L3ResidualAK8);
  
  for (int isrc = 0; isrc < nsrc; isrc++) {
    const char *name = jecsrcnames[isrc];
    JetCorrectorParameters *pAK4 = new JetCorrectorParameters(mJECUncFileAK4.c_str(), name) ;
    JetCorrectionUncertainty *uncAK4 = new JetCorrectionUncertainty(*pAK4);
    vsrc.push_back(uncAK4);
    JetCorrectorParameters *pAK8 = new JetCorrectorParameters(mJECUncFileAK8.c_str(), name) ;
    JetCorrectionUncertainty *uncAK8 = new JetCorrectionUncertainty(*pAK8);
    vsrcAK8.push_back(uncAK8);
  }
  
  if(read_btagSF){
	calib_deepflav = BTagCalibration("DeepJet", mBtagSF_DeepFlav.c_str(),true);
	reader_deepflav = BTagCalibrationReader(BTagEntry::OP_MEDIUM, "central", {"up", "down"}); 
	reader_deepflav.load(calib_deepflav, BTagEntry::FLAV_B, "comb");
	reader_deepflav.load(calib_deepflav, BTagEntry::FLAV_C, "comb");
	reader_deepflav.load(calib_deepflav, BTagEntry::FLAV_UDSG, "incl");
  } 
  
  if(isUltraLegacy)
  {
	   
	if(year=="2018"){
		roch_cor.init((mRochcorFolder+"RoccoR2018UL.txt").c_str()); 
	}
	if(year=="2017"){
		roch_cor.init((mRochcorFolder+"RoccoR2017UL.txt").c_str()); 
	}
	if(year=="2016"){
		roch_cor.init((mRochcorFolder+"RoccoR2016aUL.txt").c_str()); 
	}
	
  }
  else{
		roch_cor.init((mRochcorFolder+"RoccoR2017.txt").c_str()); 
	  }
  
  //**Important**//
  //For precision top physics, change "comb" to "mujets" in BTagCalibrationReader above //
  //https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18#Additional_information
  
   file_jetvetomap = new TFile(mJetVetoMap.c_str(),"read");
   h_jetvetomap = (TH2D*)file_jetvetomap->Get("jetvetomap");
   h_jetvetomap_eep = (TH2D*)file_jetvetomap->Get("jetvetomap_eep");
	
}

// ------------ method called once each job just after ending the event loop  ------------
void 
Leptop::endJob() 
{
  theFile->cd();
  theFile->Write();
  theFile->Close();
  
  //delete L1FastAK4;
}

// ------------ method called when starting to processes a run  ------------
void 
Leptop::beginRun(edm::Run const& iRun, edm::EventSetup const& pset)
{	
  bool changed(true);
  if(!isFastSIM){
	//hltPrescaleProvider_.init(iRun,pset,theHLTTag,changed);
	//HLTConfigProvider const&  hltConfig_ = hltPrescaleProvider_.hltConfigProvider();
  }
  //hltConfig_.dump("Triggers");
  //hltConfig_.dump("PrescaleTable");
}

// ------------ method called when ending the processing of a run  ------------
void 
Leptop::endRun(edm::Run const&, edm::EventSetup const&)
{
}
// ------------ method called when starting to processes a luminosity block  ------------
void 
Leptop::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
Leptop::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

void 
Leptop::InitializeBranches(){
	
	Generator_weight = Generator_qscale = Generator_x1 = Generator_x2 = Generator_id1 = Generator_id2 = Generator_xpdf1 = Generator_xpdf2 = Generator_scalePDF = -10000;
	
	nprim = -100;
    npvert = -100;
    PV_npvsGood = -100;
    PV_ndof = -100;
    PV_chi2 = PV_x = PV_y = PV_z = -1000;
    
    LHEWeights.clear();
    nLHEWeights = 0;
    
    genmiset = genmisphi = genmisetsig = -1000;
    
    miset = misphi = misetsig = sumEt = -1000 ;
	miset_covXX = miset_covXY = miset_covYY = -100;
	miset_UnclusEup = miset_UnclusEdn = -100;
	misphi_UnclusEup = misphi_UnclusEdn = -100;
	
	miset_PUPPI = misphi_PUPPI = misetsig_PUPPI = sumEt_PUPPI = -100;
	miset_PUPPI_JESup = miset_PUPPI_JESdn = miset_PUPPI_JERup = miset_PUPPI_JERdn = miset_PUPPI_UnclusEup = miset_PUPPI_UnclusEdn = -100;
	misphi_PUPPI_JESup = misphi_PUPPI_JESdn = misphi_PUPPI_JERup = misphi_PUPPI_JERdn = misphi_PUPPI_UnclusEup = misphi_PUPPI_UnclusEdn = -100;
  
	trig_bits.clear();
	trig_paths.clear();
	
	TrigObj_HLTname.clear();
		
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
Leptop::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  
  desc.setUnknown();
  
  desc.add<std::string>("processName", "HLT");
  desc.add<std::string>("triggerName", "@");
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("triggerEvent", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"));
  desc.add<unsigned int>("stageL1Trigger", 2);
  
  descriptions.addDefault(desc);
  //descriptions.add("hltEventAnalyzerAODDefault", desc);
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(Leptop);
