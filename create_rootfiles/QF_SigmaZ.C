#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include <TROOT.h>
#include <TRandom.h>
#include <TLorentzVector.h>
#include <TGenPhaseSpace.h>
#include <TMath.h>
#include <TH1.h>
#include <TFile.h>
#include <TF1.h>
#include <TVector3.h>
#include "/home/had/yudai/private/work/kinema/Cusp/SigmaN_Cusp/QFBG/FermiMotion.hh"




//#include <TRandom.h>

void QF_SigmaZ(){
  gRandom->SetSeed(time(NULL));
  //TFile * fout = new TFile("train_QF_SigmaZ.root","RECREATE");
  //TFile * fout = new TFile("test_QF_SigmaZ.root","RECREATE");
  TFile * fout = new TFile("testHigh_QF_SigmaZ.root","RECREATE");


 const double M_SigmaZ =1.192642;
 const double M_SigmaP = 1.18937; 
 const double M_Lambda = 1.115683; 
 const double M_Proton = 0.93827203;
 const double M_Neutron = 0.939573;
 const double M_Pi = 0.13957018;
 const double M_G = 0.;
 // const double M_Pi = 0.1349766;
 const double M_Kaon = 0.493677;
 const double M_Deuteron     = 1.875612859;
 
 TLorentzVector LvH(0.0, 0.0, 0.0, M_Proton);
 TLorentzVector LvD(0.0, 0.0, 0.0, M_Deuteron);
 double beam_mom = 1.4;
 double E_mom = sqrt(M_Kaon*M_Kaon + beam_mom*beam_mom);

 TLorentzVector beam(0.0, 0.0,beam_mom ,E_mom );
 TLorentzVector W1;
 TLorentzVector W2,W3;

 TLorentzVector *pPi_mom = new TLorentzVector();
 TLorentzVector *pY_mom = new TLorentzVector();
 TLorentzVector *pSpec_mom = new TLorentzVector();
 TLorentzVector *DecayL_mom = new TLorentzVector();
 TLorentzVector *DecayG_mom = new TLorentzVector();
 TLorentzVector *DecayP_mom = new TLorentzVector();
 TLorentzVector *DecayPi_mom = new TLorentzVector();
 
 double mm_d, mm_d_res;
 double mm_p, mm_p_res;
 double theta;

 double masses1[2] = {M_Pi,M_SigmaZ};
 double masses2[2] = {M_Lambda,M_G};
 double masses3[2] = {M_Proton,M_Pi};

 double px_Pi, py_Pi, pz_Pi;
 double px_DP, py_DP, pz_DP;
 double px_DPi, py_DPi, pz_DPi;
 double px_SP, py_SP, pz_SP;

 double p_Pi, pth_Pi, pphi_Pi;
 double p_DP, pth_DP, pphi_DP;
 double p_DPi, pth_DPi, pphi_DPi;
 double p_SP, pth_SP, pphi_SP;

 double reaction = 2.;

 TTree * tree = new TTree("tree","tree of data");
 //tree -> Branch("P_K_Lab",&P_K_Lab,"P_K_Lab/D"); 
 tree->Branch("mm_d",&mm_d, "mm_d/D");
 tree->Branch("mm_d_res",&mm_d_res, "mm_d_res/D");
 tree->Branch("theta",&theta, "theta/D");
  tree->Branch("theta",&theta, "theta/D");
 tree->Branch("px_Pi",&px_Pi, "px_Pi/D");
 tree->Branch("py_Pi",&py_Pi, "py_Pi/D");
 tree->Branch("pz_Pi",&pz_Pi, "pz_Pi/D");
 tree->Branch("px_DP",&px_DP, "px_DP/D");
 tree->Branch("py_DP",&py_DP, "py_DP/D");
 tree->Branch("pz_DP",&pz_DP, "pz_DP/D");
 tree->Branch("px_DPi",&px_DPi, "px_DPi/D");
 tree->Branch("py_DPi",&py_DPi, "py_DPi/D");
 tree->Branch("pz_DPi",&pz_DPi, "pz_DPi/D");
 tree->Branch("px_SP",&px_SP, "px_SP/D");
 tree->Branch("py_SP",&py_SP, "py_SP/D");
 tree->Branch("pz_SP",&pz_SP, "pz_SP/D");
 
 tree->Branch("p_Pi",&p_Pi, "p_Pi/D");
 tree->Branch("pth_Pi",&pth_Pi, "pth_Pi/D");
 tree->Branch("pphi_Pi",&pphi_Pi, "pphi_Pi/D");
 tree->Branch("p_DP",&p_DP, "p_DP/D");
 tree->Branch("pth_DP",&pth_DP, "pth_DP/D");
 tree->Branch("pphi_DP",&pphi_DP, "pphi_DP/D");
 tree->Branch("p_DPi",&p_DPi, "p_DPi/D");
 tree->Branch("pth_DPi",&pth_DPi, "pth_DPi/D");
 tree->Branch("pphi_DPi",&pphi_DPi, "pphi_DPi/D");
 tree->Branch("p_SP",&p_SP, "p_SP/D");
 tree->Branch("pth_SP",&pth_SP, "pth_SP/D");
 tree->Branch("pphi_SP",&pphi_SP, "pphi_SP/D");

 tree->Branch("reaction", &reaction, "reaction/D");


 std::cout<<"getchar "<<std::endl;
 getchar();
 int i =0;
 while(1){
   // for(int i = 0; i < 10000000; i++){
   
   if(i %1000 ==0){
     std::cout<<"Now Event is "<<i <<std::endl;
   }
//---------------pi-n_reaction--Sigma_production---------------------------

   TVector3 VFermi;
   VFermi = FermiMotion::GetMomentum();
   TVector3 VSpec = -VFermi;
   pSpec_mom->SetVectM(VSpec, M_Proton);
   double Md = M_Deuteron;
   double Mp = M_Proton;
   double MSpec = sqrt(Md*Md + Mp*Mp - 2.*sqrt(Mp*Mp + VFermi.Mag2())*Md);
    
   double E_Proton = sqrt(VFermi.Mag2() + MSpec*MSpec);

   TLorentzVector target;
   target.SetVectM(VFermi, MSpec);//target = proton
   W1 = beam + target;

   TGenPhaseSpace event1;
   event1.SetDecay(W1, 2, masses1);
   
   Double_t weight1 =   event1.Generate();
   pPi_mom = event1.GetDecay(0);
   pY_mom = event1.GetDecay(1);
   
   theta = pPi_mom->Theta()*180./acos(-1);
   TVector3 V_Pi(pPi_mom->Px(), pPi_mom->Py(), pPi_mom->Pz());
   TLorentzVector LvPi;
   LvPi.SetVectM(V_Pi, M_Pi);

   TLorentzVector LvRcH = beam + LvH - LvPi;
   TLorentzVector LvRcD = beam + LvD - LvPi;
   mm_p = LvRcH.M();
   mm_d = LvRcD.M();
   double FWHM_sigma = 2.*sqrt(2.*log(2.));
   // 1 MeV in FWHM
   mm_p_res = LvRcH.M()+gRandom->Gaus(0., 0.001/FWHM_sigma);
   mm_d_res = LvRcD.M()+gRandom->Gaus(0., 0.001/FWHM_sigma);

   //--------------SimgaZ decay to LG ---------------------------
   W2=*pY_mom;
   TGenPhaseSpace event2;
   event2.SetDecay(W2, 2, masses2);
 
   Double_t weight2 =   event2.Generate();
   DecayL_mom = event2.GetDecay(0);
   DecayG_mom = event2.GetDecay(1);

   //--------------Lambda decay to ppi ---------------------------
   W3=*DecayL_mom;
   TGenPhaseSpace event3;
   event3.SetDecay(W3, 2, masses3);
 
   Double_t weight3 =   event3.Generate();
   DecayP_mom = event3.GetDecay(0);
   DecayPi_mom = event3.GetDecay(1);

   px_Pi=pPi_mom->Px(), py_Pi=pPi_mom->Py(), pz_Pi=pPi_mom->Pz();
   px_DP=DecayP_mom->Px(), py_DP=DecayP_mom->Py(), pz_DP=DecayP_mom->Pz();
   px_DPi=DecayPi_mom->Px(), py_DPi=DecayPi_mom->Py(), pz_DPi=DecayPi_mom->Pz();
   px_SP=pSpec_mom->Px(), py_SP=pSpec_mom->Py(), pz_SP=pSpec_mom->Pz();

   p_Pi=pPi_mom->P(), pth_Pi=pPi_mom->Theta(), pphi_Pi=pPi_mom->Phi()+acos(-1);
   p_DP=DecayP_mom->P(), pth_DP=DecayP_mom->Theta(), pphi_DP=DecayP_mom->Phi()+acos(-1);
   p_DPi=DecayPi_mom->P(), pth_DPi=DecayPi_mom->Theta(), pphi_DPi=DecayPi_mom->Phi()+acos(-1);
   p_SP=pSpec_mom->P(), pth_SP=pSpec_mom->Theta(), pphi_SP=pSpec_mom->Phi()+acos(-1);



   //   if(theta<18.)
   if(theta<10.){
     tree->Fill();
     ++i;
     //if(i>=10000)
     if(i>=3333)
     //if(i>=333)
       break;
   }
 }

 fout->Write();
 fout->Close();
 delete fout;
}
