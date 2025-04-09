// -*- C++ -*-
//
// Package:    Run2_2016/TopplusB
// Class:      TopplusB
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

// system include files
#include <memory>
// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace edm;
using namespace reco;  
using namespace CLHEP;
using namespace math;

const float mu_mass = 0.105658;
const float el_mass = 0.000511;
const float pival = acos(-1.);

//class declaration
//
class LeptonLessPFProducer : public edm::stream::EDProducer<> {
public:
  explicit LeptonLessPFProducer(const edm::ParameterSet&);
  ~LeptonLessPFProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  
private:
  virtual void beginStream(edm::StreamID) override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endStream() override;

  // ----------member data ---------------------------
     
  edm::EDGetTokenT<reco::VertexCollection> tok_primaryVertices_;
  edm::EDGetTokenT<pat::PackedCandidateCollection>tok_pfcands_;
  edm::EDGetTokenT<edm::View<pat::Muon>> tok_muons_;
  edm::EDGetTokenT<edm::View<pat::Electron>> tok_electrons_;
  
  //std::unique_ptr<EffectiveAreas> ea_miniiso_;
  
  // object cuts //
  int iTag;
  int iTagMET;
   
  double maxEta;
  double minmuPt;
  double minePt;
  
  // Root file & tree //
 
  unsigned ievt;
   
  // Electron MVA ID //
  
  std::string melectronID_isowp90, melectronID_noisowp90;
  std::string melectronID_isowp80, melectronID_noisowp80;
  
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

LeptonLessPFProducer::LeptonLessPFProducer(const edm::ParameterSet& pset)
{
  //now do what ever initialization is needed
      
  minmuPt = pset.getUntrackedParameter<double>("minmuPt",10.);
  minePt = pset.getUntrackedParameter<double>("minePt",10.);
 
  maxEta = pset.getUntrackedParameter<double>("maxEta",3.);
  
  tok_primaryVertices_ =consumes<reco::VertexCollection>( pset.getParameter<edm::InputTag>("PrimaryVertices"));     
  tok_pfcands_ = consumes<pat::PackedCandidateCollection>( pset.getParameter<edm::InputTag>("pfCands"));
  tok_muons_ = consumes<edm::View<pat::Muon>> ( pset.getParameter<edm::InputTag>("Muons"));
  tok_electrons_ = consumes<edm::View<pat::Electron>> ( pset.getParameter<edm::InputTag>("Electrons"));
  
  melectronID_isowp90       = pset.getParameter<std::string>("electronID_isowp90");
  melectronID_noisowp90     = pset.getParameter<std::string>("electronID_noisowp90");
  melectronID_isowp80       = pset.getParameter<std::string>("electronID_isowp80");
  melectronID_noisowp80     = pset.getParameter<std::string>("electronID_noisowp80");
    
  produces< pat::PackedCandidateCollection >(        );
  
}


LeptonLessPFProducer::~LeptonLessPFProducer()
{
 
  // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
LeptonLessPFProducer::produce(edm::Event& iEvent, const edm::EventSetup& pset) {
  
  using namespace edm;
    
  // Primary vertex info //
  
  Handle<VertexCollection> primaryVertices;
  iEvent.getByToken(tok_primaryVertices_, primaryVertices);
  reco::Vertex vertex;
  
  auto vertex_pos = primaryVertices->size() > 0 ?  primaryVertices->at(0).position() : reco::Vertex::Point();
  vertex = primaryVertices->at(0);
 
  // ====== RECO-objects now  ==========//

  // Muons //
    
  std::vector<pat::Muon> tlvmu;
      
  for(const auto& muon1 : iEvent.get(tok_muons_) ) {                                                                                 

	if (((muon1.isTrackerMuon() || muon1.isGlobalMuon()) && (muon1.isPFMuon())) && (muon1.pt()>=minmuPt) && (fabs(muon1.eta())<=maxEta)) {                                                              
			                                                                                                                                                                                                                                      
		TrackRef trkglb =muon1.globalTrack();                                                                                                       
		if ((!muon1.isGlobalMuon())) {                                                                                                              
			if (muon1.isTrackerMuon()) {                                                                                                              
				trkglb =muon1.innerTrack();                                                                                                             
			} else {                                                                                                                                   
				trkglb =muon1.outerTrack();                                                                                                             
			}                                                                                                                                          
		}
		/*	
		bool mu_id = Muon_TightID(muon1.isGlobalMuon(),muon1.isPFMuon(),
				    trkglb->normalizedChi2(),trkglb->hitPattern().numberOfValidMuonHits(),muon1.numberOfMatchedStations(),
				    muon1.muonBestTrack()->dxy(vertex_pos),muon1.muonBestTrack()->dz(vertex_pos),
				    trktrk->hitPattern().numberOfValidPixelHits(),trktrk->hitPattern().trackerLayersWithMeasurement());
		*/
		bool mu_id = (muon::isTightMuon(muon1,vertex));
		if (muon1.pt()>15 && fabs(muon1.eta())<2.5 && mu_id && muon1.muonBestTrack()->dxy(vertex_pos)<0.2 && muon1.muonBestTrack()->dz(vertex_pos)<0.5) {
			tlvmu.push_back(muon1);
		}                                                                                                             
		
	}                                                                                                                                              
                                                                                                                                                     
  }// muon loop 
  
  // Electrons //
      
  std::vector<pat::Electron> tlvel;
    
  for(const auto& electron1 : iEvent.get(tok_electrons_) ) {                                                                                          
 
	GsfTrackRef gsftrk1 = electron1.gsfTrack();   	
 
	if ((!gsftrk1.isNull()) && (electron1.pt()>=minePt) && (fabs(electron1.eta())<=maxEta) && (gsftrk1->ndof()>=9)){
  
		bool impact_pass = 	((fabs(electron1.superCluster()->eta())<1.4442 && fabs(gsftrk1->dxy(vertex_pos))<0.05 && fabs(gsftrk1->dz(vertex_pos))<0.1)
					   ||(fabs(electron1.superCluster()->eta())>1.5660 && fabs(gsftrk1->dxy(vertex_pos))<(2*0.05) && fabs(gsftrk1->dz(vertex_pos))<(2*0.1)));

		if(electron1.pt()>15 && fabs(electron1.superCluster()->eta())<2.5 && electron1.electronID(melectronID_noisowp90) && impact_pass){
				tlvel.push_back(electron1);
		}
	
	}
  }
   
  std::vector<unsigned int> filteredCandidateList; 
  for (unsigned int ilep = 0; ilep<tlvmu.size(); ilep++) {
	  for(unsigned int jd = 0 ; jd < tlvmu[ilep].numberOfSourceCandidatePtrs() ; ++jd) {
		  if(tlvmu[ilep].sourceCandidatePtr(jd).isNonnull() && tlvmu[ilep].sourceCandidatePtr(jd).isAvailable()){
			filteredCandidateList.push_back(tlvmu[ilep].sourceCandidatePtr(jd).key()); 
		}
	}
  }
  for (unsigned int ilep = 0; ilep<tlvel.size(); ilep++) {
	  for(unsigned int jd = 0 ; jd < tlvel[ilep].numberOfSourceCandidatePtrs() ; ++jd) {
		  if(tlvel[ilep].sourceCandidatePtr(jd).isNonnull() && tlvel[ilep].sourceCandidatePtr(jd).isAvailable()){
			filteredCandidateList.push_back(tlvel[ilep].sourceCandidatePtr(jd).key()); 
		}
	}
  }
    // other option: for(unsigned int jd = 0 ; jd < tlvel[ilep].associatedPackedPFCandidates().size() ; ++jd) 
    
  // PF candidates //
  
  edm::Handle<pat::PackedCandidateCollection> pfs;
  iEvent.getByToken(tok_pfcands_, pfs);  
  
  std::unique_ptr<pat::PackedCandidateCollection> filteredCands;
  filteredCands.reset( new pat::PackedCandidateCollection );
  filteredCands->reserve(pfs->size());
  
  for(unsigned int iP = 0; iP < pfs->size(); ++iP){
	const pat::PackedCandidate *cand = &pfs->at(iP);
	bool found = false;
	for(const auto& filtIdx : filteredCandidateList){
		if (iP == filtIdx){ found = true; break;}
	}
		if(found){  continue; }
		filteredCands->push_back(*cand);
	}

  iEvent.put(std::move(filteredCands));

  //cout<<"done!"<<endl;
    
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
LeptonLessPFProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
LeptonLessPFProducer::endStream() {
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
LeptonLessPFProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LeptonLessPFProducer);
