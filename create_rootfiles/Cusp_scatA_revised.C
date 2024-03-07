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
#include "/home/had/yudai/private/work/kinema/Cusp/SigmaN_Cusp/FdComplex.hh"


void Cusp_scatA_revised(){
  gRandom->SetSeed(time(NULL));
  //TFile * fout = new TFile("train_Cusp.root","RECREATE");
  // TFile * fout = new TFile("test_Cusp.root","RECREATE");
  TFile * fout = new TFile("testHigh_Cusp.root","RECREATE");
  
  //TFile * fout = new TFile("Cusp_scatA_param2_revised.root","RECREATE");
  //TFile * fout = new TFile("Cusp_scatA_param2_revised_Badalyan.root","RECREATE");

  // double scat_a[3] = {-2.47, -3.0, 2.06};//Julich ModelA, Dalitz, ND
  // double scat_b[3] = {3.74, 1.8, 4.64};

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


  const double scat_a = 2.06;
  const double scat_b = 4.64;
  //  TF1 *fcusp=new TF1("fcusp", "f(x,[0],[1],[2])", 
  TF1 *fcusp=new TF1("fcusp", "FdComplex::f_single(x,[0],[1],[2])", 
		     //TF1 *fcusp=new TF1("fcusp", "FdComplex::f(x,[0],[1],[2])", 
		     (M_SigmaP + M_Neutron)*1000. - 100.,
		     (M_SigmaP + M_Neutron)*1000. + 100.);
  fcusp->SetNpx(1000);
  fcusp->SetParameter(0, scat_a);
  fcusp->SetParameter(1, scat_b);
  fcusp->SetParameter(2, 186.);
  double fcusp_max = fcusp->Eval(fcusp->GetMaximumX());
    
		    
  TLorentzVector LvH(0.0, 0.0, 0.0, M_Proton);
  TLorentzVector LvD(0.0, 0.0, 0.0, M_Deuteron);
  double beam_mom = 1.4;
  double E_mom = sqrt(M_Kaon*M_Kaon + beam_mom*beam_mom);

  TLorentzVector beam(0.0, 0.0,beam_mom ,E_mom );
 
  TLorentzVector target(0.0, 0.0, 0.0, M_Deuteron);//target = deuteron
  TLorentzVector W1;
  TLorentzVector W2,W3;

  TLorentzVector *pPi_mom = new TLorentzVector();
  TLorentzVector *pCusp_mom = new TLorentzVector();
  TLorentzVector *pL_mom = new TLorentzVector();
  TLorentzVector *pP1_mom = new TLorentzVector();
  TLorentzVector *pPi1_mom = new TLorentzVector();
  TLorentzVector *pP2_mom = new TLorentzVector();

 
  double mm_d, mm_d_res;
  double mm_p, mm_p_res;
  double theta;

  double masses1[2];// = {M_Pi,M_Lambda};
  double masses2[2] = {M_Lambda,M_Proton};
  double masses3[2] = {M_Proton,M_Pi};

  double px_Pi, py_Pi, pz_Pi;
  double px_DP, py_DP, pz_DP;
  double px_DPi, py_DPi, pz_DPi;
  double px_SP, py_SP, pz_SP;

  double p_Pi, pth_Pi, pphi_Pi;
  double p_DP, pth_DP, pphi_DP;
  double p_DPi, pth_DPi, pphi_DPi;
  double p_SP, pth_SP, pphi_SP;

  double reaction = 0.;
  
  TTree * tree = new TTree("tree","tree of data");
  tree->Branch("mm_d",&mm_d, "mm_d/D");
  // tree->Branch("mm_p_res",&mm_p_res, "mm_p_res/D");
  tree->Branch("mm_d_res",&mm_d_res, "mm_d_res/D");
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

  //for(int i = 0; i < 1000000; i++){
  //for(int i = 0; i < 3000000; i++){
  //  for(int i = 0; i < 1000000; i++){
  int i =0;
  while(1){
      
    //---------------pi-n_reaction--Sigma_production---------------------------

    double CuspM=0.;
    while(1){
      double MM = gRandom->Uniform((M_SigmaZ + M_Neutron)*1000. - 100.,
				   (M_SigmaZ + M_Neutron)*1000. + 100.);
      double ds_MM = fcusp->Eval(MM);
      double Rand = gRandom->Uniform(0., fcusp_max*1.3);
      if(Rand<=ds_MM){
	//	double MM_res = gRandom->Gaus(MM, resolution[0]/FWHM_sigma);
	//h[i][j]->Fill(MM);
	CuspM = MM*0.001;
	break;
      }
    }
    W1 = beam + target;
    masses1[0] = M_Pi;
    masses1[1] = CuspM;
    TGenPhaseSpace event1;
    event1.SetDecay(W1, 2, masses1);
   
    Double_t weight1 =   event1.Generate();
    pPi_mom = event1.GetDecay(0);
    pCusp_mom = event1.GetDecay(1);
   
    theta = pPi_mom->Theta()*180./acos(-1);
    if(theta>=10.)
      continue;

    //    if(i %1000 ==0){
    if(i %10 ==0){
      std::cout<<"Now Event is "<<i <<std::endl;
    }


    
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

    if(pCusp_mom->Mag()<=masses2[0]+masses2[1])
      continue;
    //--------------Lambda decay to ppi ---------------------------
    W2=*pCusp_mom;
    TGenPhaseSpace event2;
    event2.SetDecay(W2, 2, masses2);
    Double_t weight2 =   event2.Generate();
    pL_mom = event2.GetDecay(0);
    pP2_mom = event2.GetDecay(1);

    if(pL_mom->Mag()<=masses3[0]+masses3[1])
      continue;
    //-------------- L to p pi ---------------------------
    W3=*pL_mom;
    TGenPhaseSpace event3;
    event3.SetDecay(W3, 2, masses3);
 
    Double_t weight3 =   event3.Generate();
    pP1_mom = event3.GetDecay(0);
    pPi1_mom = event3.GetDecay(1);
    
    px_Pi=pPi_mom->Px(), py_Pi=pPi_mom->Py(), pz_Pi=pPi_mom->Pz();
    px_DP=pP1_mom->Px(), py_DP=pP1_mom->Py(), pz_DP=pP1_mom->Pz();
    px_DPi=pPi1_mom->Px(), py_DPi=pPi1_mom->Py(), pz_DPi=pPi1_mom->Pz();
    px_SP=pP2_mom->Px(), py_SP=pP2_mom->Py(), pz_SP=pP2_mom->Pz();
    
    p_Pi=pPi_mom->P(), pth_Pi=pPi_mom->Theta(), pphi_Pi=pPi_mom->Phi()+acos(-1);
    p_DP=pP1_mom->P(), pth_DP=pP1_mom->Theta(), pphi_DP=pP1_mom->Phi()+acos(-1);
    p_DPi=pPi1_mom->P(), pth_DPi=pPi1_mom->Theta(), pphi_DPi=pPi1_mom->Phi()+acos(-1);
    p_SP=pP2_mom->P(), pth_SP=pP2_mom->Theta(), pphi_SP=pP2_mom->Phi()+acos(-1);

   
    tree->Fill();
    ++i;
    //if(i>=10000)
    if(i>=3333)
      //if(i>=333)
      break;

  }

  fout->Write();
  fout->Close();
  delete fout;
}
