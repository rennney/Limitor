void TOsc::Set_oscillation_base(TString default_eventlist_dir)
{
  /// Eventlist generator
  /// /home/xji/data0/work/work_oscillation/301_Framework_for_Osc/wcp-uboone-bdt/apps/convert_checkout_hist.cxx, winxp
  
  cout<<endl;
  cout<<" ---> Set_oscillation_base"<<endl;

  matrix_oscillation_base_oldworld_pred = matrix_default_oldworld_pred;

  TString str_dirbase = default_eventlist_dir;
  
  ///////////////////
#define OTHERRUNS
  if( flag_NuMI_nueCC_from_intnue ) {
    cout<<endl<<"      ---> flag_NuMI_nueCC_from_intnue"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_intrinsic_nue_overlay.root ";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_intrinsic.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_intnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_intnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_intnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_intnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_intnue_PC_eventinfo);
    }
#ifdef OTHERRUNS
    {// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_intrinsic_nue_overlay.root ";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_intrinsic.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_intnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_intnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_intnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_intnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_intnue_PC_eventinfo);
    }

    {// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_intrinsic_nue_overlay.root ";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_intrinsic.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_intnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_intnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_intnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_intnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_intnue_PC_eventinfo);
    }

    {// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_intrinsic_nue_overlay.root ";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_intrinsic.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_intnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_intnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_intnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_intnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_intnue_PC_eventinfo);
    }

    {// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_intrinsic_nue_overlay.root ";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_intrinsic.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_intnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_intnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_intnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_intnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_intnue_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_nueCC_from_intnue_scaleFPOT, &vector_vector_NuMI_nueCC_from_intnue_FC_eventinfo, 22, "nue2nue");// hack
    Set_oscillation_base_minus(&vector_NuMI_nueCC_from_intnue_scaleFPOT, &vector_vector_NuMI_nueCC_from_intnue_PC_eventinfo, 23, "nue2nue");// hack
  
  }// if( flag_NuMI_nueCC_from_intnue )

  ///////////////////
  ///////////////////

  if( flag_NuMI_nueCC_from_overlaynumu ) {
    cout<<endl<<"      ---> flag_NuMI_nueCC_from_overlaynumu"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumu_PC_eventinfo);
    }
#ifdef OTHERRUNS
    {// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumu_PC_eventinfo);
    }
    {// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumu_PC_eventinfo);
    }
    {// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumu_PC_eventinfo);
    }
    {// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumu_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_nueCC_from_overlaynumu_scaleFPOT, &vector_vector_NuMI_nueCC_from_overlaynumu_FC_eventinfo, 22, "numu2numu");// hack
    Set_oscillation_base_minus(&vector_NuMI_nueCC_from_overlaynumu_scaleFPOT, &vector_vector_NuMI_nueCC_from_overlaynumu_PC_eventinfo, 23, "numu2numu");// hack
  
  }// if( flag_NuMI_nueCC_from_overlaynumu )
  
  ///////////////////

  if( flag_NuMI_nueCC_from_overlaynueNC ) {
    cout<<endl<<"      ---> flag_NuMI_nueCC_from_overlaynueNC"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynueNC_PC_eventinfo);
    }
#ifdef OTHERRUNS
    {// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynueNC_PC_eventinfo);
    }
    {// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynueNC_PC_eventinfo);
    }
    {// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynueNC_PC_eventinfo);
    }
    {// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynueNC_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT, &vector_vector_NuMI_nueCC_from_overlaynueNC_FC_eventinfo, 22, "nueNC");// hack
    Set_oscillation_base_minus(&vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT, &vector_vector_NuMI_nueCC_from_overlaynueNC_PC_eventinfo, 23, "nueNC");// hack
  
  }// if( flag_NuMI_nueCC_from_overlaynueNC )
   
  ///////////////////

  if( flag_NuMI_nueCC_from_overlaynumuNC ) {
    cout<<endl<<"      ---> flag_NuMI_nueCC_from_overlaynumuNC"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumuNC_PC_eventinfo);
    }
#ifdef OTHERRUNS
    {// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumuNC_PC_eventinfo);
    }
    {// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumuNC_PC_eventinfo);
    }
    {// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumuNC_PC_eventinfo);
    }
    {// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_overlaynumuNC_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT, &vector_vector_NuMI_nueCC_from_overlaynumuNC_FC_eventinfo, 22, "numuNC");// hack
    Set_oscillation_base_minus(&vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT, &vector_vector_NuMI_nueCC_from_overlaynumuNC_PC_eventinfo, 23, "numuNC");// hack
  
  }// if( flag_NuMI_nueCC_from_overlaynumuNC )
  
  ///////////////////

  if( flag_NuMI_numuCC_from_overlaynumu ) {
    cout<<endl<<"      ---> flag_NuMI_numuCC_from_overlaynumu"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                           strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumu_PC_eventinfo);
    }
#ifdef OTHERRUNS
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                           strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumu_PC_eventinfo);
    }
    {// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                           strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumu_PC_eventinfo);
    }
    {// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                           strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumu_PC_eventinfo);
    }
    {// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                           strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumu_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_numuCC_from_overlaynumu_scaleFPOT, &vector_vector_NuMI_numuCC_from_overlaynumu_FC_eventinfo, 24, "numu2numu");// hack
    Set_oscillation_base_minus(&vector_NuMI_numuCC_from_overlaynumu_scaleFPOT, &vector_vector_NuMI_numuCC_from_overlaynumu_PC_eventinfo, 25, "numu2numu");// hack
  
  }// if( flag_NuMI_numuCC_from_overlaynumu )
    
  ///////////////////

  if( flag_NuMI_numuCC_from_overlaynueNC ) {
    cout<<endl<<"      ---> flag_NuMI_numuCC_from_overlaynueNC"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynueNC_PC_eventinfo);
    }
#ifdef OTHERRUNS
   {// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynueNC_PC_eventinfo);
    }
   {// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynueNC_PC_eventinfo);
    }
   {// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynueNC_PC_eventinfo);
    }
   {// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynueNC_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT, &vector_vector_NuMI_numuCC_from_overlaynueNC_FC_eventinfo, 24, "nueNC");// hack
    Set_oscillation_base_minus(&vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT, &vector_vector_NuMI_numuCC_from_overlaynueNC_PC_eventinfo, 25, "nueNC");// hack
  
  }// if( flag_NuMI_numuCC_from_overlaynueNC )
   
  ///////////////////

  if( flag_NuMI_numuCC_from_overlaynumuNC ) {
    cout<<endl<<"      ---> flag_NuMI_numuCC_from_overlaynumuNC"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumuNC_PC_eventinfo);
    }
#ifdef OTHERRUNS
    {// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumuNC_PC_eventinfo);
    }
    {// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumuNC_PC_eventinfo);
    }
    {// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumuNC_PC_eventinfo);
    }
    {// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_overlaynumuNC_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT, &vector_vector_NuMI_numuCC_from_overlaynumuNC_FC_eventinfo, 24, "numuNC");// hack
    Set_oscillation_base_minus(&vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT, &vector_vector_NuMI_numuCC_from_overlaynumuNC_PC_eventinfo, 25, "numuNC");// hack
  
  }// if( flag_NuMI_numuCC_from_overlaynumuNC )
  
  ///////////////////

  if( flag_NuMI_CCpi0_from_overlaynumu ) {
    cout<<endl<<"      ---> flag_NuMI_CCpi0_from_overlaynumu"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumu_PC_eventinfo);
    }
#ifdef OTHERRUNS
 {// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumu_PC_eventinfo);
    }
 {// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumu_PC_eventinfo);
    }
 {// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumu_PC_eventinfo);
    }
 {// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumu_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT, &vector_vector_NuMI_CCpi0_from_overlaynumu_FC_eventinfo, 26, "numu2numu");// hack
    Set_oscillation_base_minus(&vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT, &vector_vector_NuMI_CCpi0_from_overlaynumu_PC_eventinfo, 27, "numu2numu");// hack
  
  }// if( flag_NuMI_CCpi0_from_overlaynumu )
      
  ///////////////////

  if( flag_NuMI_CCpi0_from_overlaynueNC ) {
    cout<<endl<<"      ---> flag_NuMI_CCpi0_from_overlaynueNC"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynueNC_PC_eventinfo);
    }
#ifdef OTHERRUNS
{// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynueNC_PC_eventinfo);
    }
{// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynueNC_PC_eventinfo);
    }
{// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynueNC_PC_eventinfo);
    }
{// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynueNC_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT, &vector_vector_NuMI_CCpi0_from_overlaynueNC_FC_eventinfo, 26, "nueNC");// hack
    Set_oscillation_base_minus(&vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT, &vector_vector_NuMI_CCpi0_from_overlaynueNC_PC_eventinfo, 27, "nueNC");// hack
  
  }// if( flag_NuMI_CCpi0_from_overlaynueNC )
   
  ///////////////////

  if( flag_NuMI_CCpi0_from_overlaynumuNC ) {
    cout<<endl<<"      ---> flag_NuMI_CCpi0_from_overlaynumuNC"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_PC_eventinfo);
    }
#ifdef OTHERRUNS
    {// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_PC_eventinfo);
    }
    {// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_PC_eventinfo);
    }
    {// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_PC_eventinfo);
    }
    {// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_FC_eventinfo, 26, "numuNC");// hack
    Set_oscillation_base_minus(&vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT, &vector_vector_NuMI_CCpi0_from_overlaynumuNC_PC_eventinfo, 27, "numuNC");// hack
  
  }// if( flag_NuMI_CCpi0_from_overlaynumuNC )
  
  ///////////////////

  if( flag_NuMI_NCpi0_from_overlaynumu ) {
    cout<<endl<<"      ---> flag_NuMI_NCpi0_from_overlaynumu"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumu";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynumu_eventinfo);
    }
#ifdef OTHERRUNS
 {// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumu";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynumu_eventinfo);
    }
 {// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumu";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynumu_eventinfo);
    }
 {// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumu";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynumu_eventinfo);
    }
 {// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumu";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynumu_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_NCpi0_from_overlaynumu_scaleFPOT, &vector_vector_NuMI_NCpi0_from_overlaynumu_eventinfo, 28, "numu2numu");// hack
  
  }// if( flag_NuMI_NCpi0_from_overlaynumu )
    
  ///////////////////

  if( flag_NuMI_NCpi0_from_overlaynueNC ) {
    cout<<endl<<"      ---> flag_NuMI_NCpi0_from_overlaynueNC"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynueNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynueNC_eventinfo);
    }
#ifdef OTHERRUNS
{// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynueNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynueNC_eventinfo);
    }
{// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynueNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynueNC_eventinfo);
    }
{// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynueNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynueNC_eventinfo);
    }
{// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynueNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynueNC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_NCpi0_from_overlaynueNC_scaleFPOT, &vector_vector_NuMI_NCpi0_from_overlaynueNC_eventinfo, 28, "nueNC");// hack
  
  }// if( flag_NuMI_NCpi0_from_overlaynueNC )
          
  ///////////////////

  if( flag_NuMI_NCpi0_from_overlaynumuNC ) {
    cout<<endl<<"      ---> flag_NuMI_NCpi0_from_overlaynumuNC"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumuNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynumuNC_eventinfo);
    }
#ifdef OTHERRUNS
{// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumuNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynumuNC_eventinfo);
    }
{// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumuNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynumuNC_eventinfo);
    }
{// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumuNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynumuNC_eventinfo);
    }
{// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_nu_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumuNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_overlaynumuNC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_NCpi0_from_overlaynumuNC_scaleFPOT, &vector_vector_NuMI_NCpi0_from_overlaynumuNC_eventinfo, 28, "numuNC");// hack
  
  }// if( flag_NuMI_NCpi0_from_overlaynumuNC )
      
  ///////////////////

  if( flag_NuMI_nueCC_from_appnue || 1 ) {
    cout<<endl<<"      ---> flag_NuMI_nueCC_from_appnue"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_appnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_appnue_PC_eventinfo);
    }
#ifdef OTHERRUNS
{// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_appnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_appnue_PC_eventinfo);
    }
{// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_appnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_appnue_PC_eventinfo);
    }
{// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_appnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_appnue_PC_eventinfo);
    }
{// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_nueCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_appnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_nueCC_from_appnue_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_nueCC_from_appnue_scaleFPOT, &vector_vector_NuMI_nueCC_from_appnue_FC_eventinfo, 36, "numu2nue");// hack
    Set_oscillation_base_minus(&vector_NuMI_nueCC_from_appnue_scaleFPOT, &vector_vector_NuMI_nueCC_from_appnue_PC_eventinfo, 37, "numu2nue");// hack
    
  }// if( flag_NuMI_nueCC_from_appnue )
       
  ///////////////////

  if( flag_NuMI_numuCC_from_appnue || 1 ) {
    cout<<endl<<"      ---> flag_NuMI_numuCC_from_appnue"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_appnue_FC_eventinfo);
      str_treename = "tree_numuCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                      strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_appnue_PC_eventinfo);
    }
#ifdef OTHERRUNS
{// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_appnue_FC_eventinfo);
      str_treename = "tree_numuCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                      strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_appnue_PC_eventinfo);
    }
{// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_appnue_FC_eventinfo);
      str_treename = "tree_numuCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                      strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_appnue_PC_eventinfo);
    }
{// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_appnue_FC_eventinfo);
      str_treename = "tree_numuCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                      strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_appnue_PC_eventinfo);
    }
{// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_numuCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_appnue_FC_eventinfo);
      str_treename = "tree_numuCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                      strfile_mc_e2e, str_treename, &vector_vector_NuMI_numuCC_from_appnue_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_numuCC_from_appnue_scaleFPOT, &vector_vector_NuMI_numuCC_from_appnue_FC_eventinfo, 38, "numu2nue");// hack
    Set_oscillation_base_minus(&vector_NuMI_numuCC_from_appnue_scaleFPOT, &vector_vector_NuMI_numuCC_from_appnue_PC_eventinfo, 39, "numu2nue");// hack
    
  }// if( flag_NuMI_numuCC_from_appnue )
         
  ///////////////////

  if( flag_NuMI_CCpi0_from_appnue || 1 ) {
    cout<<endl<<"      ---> flag_NuMI_CCpi0_from_appnue"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_appnue_FC_eventinfo);
      str_treename = "tree_CCpi0_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_appnue_PC_eventinfo);
    }
#ifdef OTHERRUNS
{// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_appnue_FC_eventinfo);
      str_treename = "tree_CCpi0_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_appnue_PC_eventinfo);
    }
{// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_appnue_FC_eventinfo);
      str_treename = "tree_CCpi0_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_appnue_PC_eventinfo);
    }
{// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_appnue_FC_eventinfo);
      str_treename = "tree_CCpi0_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_appnue_PC_eventinfo);
    }
{// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_CCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_appnue_FC_eventinfo);
      str_treename = "tree_CCpi0_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                     strfile_mc_e2e, str_treename, &vector_vector_NuMI_CCpi0_from_appnue_PC_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_CCpi0_from_appnue_scaleFPOT, &vector_vector_NuMI_CCpi0_from_appnue_FC_eventinfo, 40, "numu2nue");// hack
    Set_oscillation_base_minus(&vector_NuMI_CCpi0_from_appnue_scaleFPOT, &vector_vector_NuMI_CCpi0_from_appnue_PC_eventinfo, 41, "numu2nue");// hack
    
  }// if( flag_NuMI_CCpi0_from_appnue )
           
  ///////////////////

  if( flag_NuMI_NCpi0_from_appnue || 1 ) {
    cout<<endl<<"      ---> flag_NuMI_NCpi0_from_appnue"<<endl;
    {// run1 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_fhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_FHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_appnue";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_appnue_eventinfo);
    }
#ifdef OTHERRUNS
{// run1 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run1_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run1_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run1_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_appnue";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_appnue_eventinfo);
    }
{// run2 FHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_fhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_fhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_FHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_appnue";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_appnue_eventinfo);
    }
{// run2 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run2_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run2_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run2_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_appnue";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_appnue_eventinfo);
    }
{// run3 RHC
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_run3_rhc_fullosc_overlay.root";
      TString strfile_dataPOT = str_dirbase + "run3_rhc_data_numi.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_NuMI_run3_RHC_appnue.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_appnue";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_NuMI_NCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_NuMI_NCpi0_from_appnue_eventinfo);
    }
#endif
    Set_oscillation_base_minus(&vector_NuMI_NCpi0_from_appnue_scaleFPOT, &vector_vector_NuMI_NCpi0_from_appnue_eventinfo, 42, "numu2nue");// hack
    
  }// if( flag_NuMI_NCpi0_from_appnue )
         
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////

  if( flag_BNB_nueCC_from_intnue ) {
    cout<<endl<<"      ---> flag_BNB_nueCC_from_intnue"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_intrinsic_nue_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_intrinsic.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_intnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_intnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_intnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_intnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_intnue_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_intrinsic_nue_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_intrinsic.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_intnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_intnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_intnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_intnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_intnue_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_intrinsic_nue_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_intrinsic.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_intnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_intnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_intnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_intnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_intnue_PC_eventinfo);
    }

    Set_oscillation_base_minus(&vector_BNB_nueCC_from_intnue_scaleFPOT, &vector_vector_BNB_nueCC_from_intnue_FC_eventinfo, 1, "nue2nue");// hack
    Set_oscillation_base_minus(&vector_BNB_nueCC_from_intnue_scaleFPOT, &vector_vector_BNB_nueCC_from_intnue_PC_eventinfo, 2, "nue2nue");// hack
    
  }// if( flag_BNB_nueCC_from_intnue )

  ///////////////////
  ///////////////////

  if( flag_BNB_nueCC_from_overlaynumu ) {
    cout<<endl<<"      ---> flag_BNB_nueCC_from_overlaynumu"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumu_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumu_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumu_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_nueCC_from_overlaynumu_scaleFPOT, &vector_vector_BNB_nueCC_from_overlaynumu_FC_eventinfo, 1, "numu2numu");// hack
    Set_oscillation_base_minus(&vector_BNB_nueCC_from_overlaynumu_scaleFPOT, &vector_vector_BNB_nueCC_from_overlaynumu_PC_eventinfo, 2, "numu2numu");// hack
    
  }// if( flag_BNB_numuCC_from_overlaynumu )
  
  ///////////////////

  if( flag_BNB_nueCC_from_overlaynueNC ) {
    cout<<endl<<"      ---> flag_BNB_nueCC_from_overlaynueNC"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynueNC_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynueNC_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynueNC_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_nueCC_from_overlaynueNC_scaleFPOT, &vector_vector_BNB_nueCC_from_overlaynueNC_FC_eventinfo, 1, "nueNC");// hack
    Set_oscillation_base_minus(&vector_BNB_nueCC_from_overlaynueNC_scaleFPOT, &vector_vector_BNB_nueCC_from_overlaynueNC_PC_eventinfo, 2, "nueNC");// hack
    
  }// if( flag_BNB_nueCC_from_overlaynueNC )
  
  ///////////////////

  if( flag_BNB_nueCC_from_overlaynumuNC ) {
    cout<<endl<<"      ---> flag_BNB_nueCC_from_overlaynumuNC"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumuNC_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumuNC_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_nueCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_overlaynumuNC_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_nueCC_from_overlaynumuNC_scaleFPOT, &vector_vector_BNB_nueCC_from_overlaynumuNC_FC_eventinfo, 1, "numuNC");// hack
    Set_oscillation_base_minus(&vector_BNB_nueCC_from_overlaynumuNC_scaleFPOT, &vector_vector_BNB_nueCC_from_overlaynumuNC_PC_eventinfo, 2, "numuNC");// hack
    
  }// if( flag_BNB_nueCC_from_overlaynumuNC )
  
  ///////////////////

  if( flag_BNB_numuCC_from_overlaynumu ) {
    cout<<endl<<"      ---> flag_BNB_numuCC_from_overlaynumu"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumu_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumu_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                          strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumu_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_numuCC_from_overlaynumu_scaleFPOT, &vector_vector_BNB_numuCC_from_overlaynumu_FC_eventinfo, 3, "numu2numu");// hack
    Set_oscillation_base_minus(&vector_BNB_numuCC_from_overlaynumu_scaleFPOT, &vector_vector_BNB_numuCC_from_overlaynumu_PC_eventinfo, 4, "numu2numu");// hack
    
  }// if( flag_BNB_numuCC_from_overlaynumu )
  
  ///////////////////

  if( flag_BNB_numuCC_from_overlaynueNC ) {
    cout<<endl<<"      ---> flag_BNB_numuCC_from_overlaynueNC"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynueNC_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynueNC_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynueNC_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_numuCC_from_overlaynueNC_scaleFPOT, &vector_vector_BNB_numuCC_from_overlaynueNC_FC_eventinfo, 3, "nueNC");// hack
    Set_oscillation_base_minus(&vector_BNB_numuCC_from_overlaynueNC_scaleFPOT, &vector_vector_BNB_numuCC_from_overlaynueNC_PC_eventinfo, 4, "nueNC");// hack
    
  }// if( flag_BNB_numuCC_from_overlaynueNC )
  
  ///////////////////

  if( flag_BNB_numuCC_from_overlaynumuNC ) {
    cout<<endl<<"      ---> flag_BNB_numuCC_from_overlaynumuNC"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumuNC_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumuNC_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_numuCC_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_overlaynumuNC_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_numuCC_from_overlaynumuNC_scaleFPOT, &vector_vector_BNB_numuCC_from_overlaynumuNC_FC_eventinfo, 3, "numuNC");// hack
    Set_oscillation_base_minus(&vector_BNB_numuCC_from_overlaynumuNC_scaleFPOT, &vector_vector_BNB_numuCC_from_overlaynumuNC_PC_eventinfo, 4, "numuNC");// hack
    
  }// if( flag_BNB_numuCC_from_overlaynumuNC )
  
  ///////////////////

  if( flag_BNB_CCpi0_from_overlaynumu ) {
    cout<<endl<<"      ---> flag_BNB_CCpi0_from_overlaynumu"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumu_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumu_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumu_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumu_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumu_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumu_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_CCpi0_from_overlaynumu_scaleFPOT, &vector_vector_BNB_CCpi0_from_overlaynumu_FC_eventinfo, 5, "numu2numu");// hack
    Set_oscillation_base_minus(&vector_BNB_CCpi0_from_overlaynumu_scaleFPOT, &vector_vector_BNB_CCpi0_from_overlaynumu_PC_eventinfo, 6, "numu2numu");// hack
    
  }// if( flag_BNB_CCpi0_from_overlaynumu )
    
  ///////////////////

  if( flag_BNB_CCpi0_from_overlaynueNC ) {
    cout<<endl<<"      ---> flag_BNB_CCpi0_from_overlaynueNC"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynueNC_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynueNC_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynueNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynueNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynueNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynueNC_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_CCpi0_from_overlaynueNC_scaleFPOT, &vector_vector_BNB_CCpi0_from_overlaynueNC_FC_eventinfo, 5, "nueNC");// hack
    Set_oscillation_base_minus(&vector_BNB_CCpi0_from_overlaynueNC_scaleFPOT, &vector_vector_BNB_CCpi0_from_overlaynueNC_PC_eventinfo, 6, "nueNC");// hack
    
  }// if( flag_BNB_CCpi0_from_overlaynueNC )
  
  ///////////////////

  if( flag_BNB_CCpi0_from_overlaynumuNC ) {
    cout<<endl<<"      ---> flag_BNB_CCpi0_from_overlaynumuNC"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumuNC_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumuNC_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_overlaynumuNC_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumuNC_FC_eventinfo);
      str_treename = "tree_CCpi0_from_overlaynumuNC_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                         strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_overlaynumuNC_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_CCpi0_from_overlaynumuNC_scaleFPOT, &vector_vector_BNB_CCpi0_from_overlaynumuNC_FC_eventinfo, 5, "numuNC");// hack
    Set_oscillation_base_minus(&vector_BNB_CCpi0_from_overlaynumuNC_scaleFPOT, &vector_vector_BNB_CCpi0_from_overlaynumuNC_PC_eventinfo, 6, "numuNC");// hack
    
  }// if( flag_BNB_CCpi0_from_overlaynumuNC )
  
  ///////////////////

  if( flag_BNB_NCpi0_from_overlaynumu ) {
    cout<<endl<<"      ---> flag_BNB_NCpi0_from_overlaynumu"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumu";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_overlaynumu_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumu";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_overlaynumu_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumu";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_overlaynumu_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_overlaynumu_eventinfo);
    }

    Set_oscillation_base_minus(&vector_BNB_NCpi0_from_overlaynumu_scaleFPOT, &vector_vector_BNB_NCpi0_from_overlaynumu_eventinfo, 7, "numu2numu");// hack
    
  }// if( flag_BNB_NCpi0_from_overlaynumu )
      
  ///////////////////

  if( flag_BNB_NCpi0_from_overlaynueNC ) {
    cout<<endl<<"      ---> flag_BNB_NCpi0_from_overlaynueNC xx"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynueNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_overlaynueNC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynueNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_overlaynueNC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynueNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_overlaynueNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_overlaynueNC_eventinfo);
    }

    Set_oscillation_base_minus(&vector_BNB_NCpi0_from_overlaynueNC_scaleFPOT, &vector_vector_BNB_NCpi0_from_overlaynueNC_eventinfo, 7, "nueNC");// hack
    
  }// if( flag_BNB_NCpi0_from_overlaynueNC )
         
  ///////////////////

  if( flag_BNB_NCpi0_from_overlaynumuNC ) {
    cout<<endl<<"      ---> flag_BNB_NCpi0_from_overlaynumuNC"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumuNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_overlaynumuNC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumuNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_overlaynumuNC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_nu_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_nu_overlay.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_overlaynumuNC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_overlaynumuNC_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_overlaynumuNC_eventinfo);
    }

    Set_oscillation_base_minus(&vector_BNB_NCpi0_from_overlaynumuNC_scaleFPOT, &vector_vector_BNB_NCpi0_from_overlaynumuNC_eventinfo, 7, "numuNC");// hack
    
  }// if( flag_BNB_NCpi0_from_overlaynumuNC )
    
  ///////////////////

  if( flag_BNB_nueCC_from_appnue || 1 ) {
    cout<<endl<<"      ---> flag_BNB_nueCC_from_appnue"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_appnue.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_appnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_appnue_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_appnue.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_appnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_appnue_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_appnue.root";
      TString str_treename = "";
      str_treename = "tree_nueCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_nueCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_appnue_FC_eventinfo);
      str_treename = "tree_nueCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_nueCC_from_appnue_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_nueCC_from_appnue_scaleFPOT, &vector_vector_BNB_nueCC_from_appnue_FC_eventinfo, 15, "numu2nue");// hack
    Set_oscillation_base_minus(&vector_BNB_nueCC_from_appnue_scaleFPOT, &vector_vector_BNB_nueCC_from_appnue_PC_eventinfo, 16, "numu2nue");// hack
    
  }// if( flag_BNB_nueCC_from_appnue )
    
  ///////////////////

  if( flag_BNB_numuCC_from_appnue || 1 ) {
    cout<<endl<<"      ---> flag_BNB_numuCC_from_appnue"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_appnue.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_appnue_FC_eventinfo);
      str_treename = "tree_numuCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_appnue_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_appnue.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_appnue_FC_eventinfo);
      str_treename = "tree_numuCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_appnue_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_appnue.root";
      TString str_treename = "";
      str_treename = "tree_numuCC_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_numuCC_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_appnue_FC_eventinfo);
      str_treename = "tree_numuCC_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_numuCC_from_appnue_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_numuCC_from_appnue_scaleFPOT, &vector_vector_BNB_numuCC_from_appnue_FC_eventinfo, 17, "numu2nue");// hack
    Set_oscillation_base_minus(&vector_BNB_numuCC_from_appnue_scaleFPOT, &vector_vector_BNB_numuCC_from_appnue_PC_eventinfo, 18, "numu2nue");// hack
    
  }// if( flag_BNB_numuCC_from_appnue )
    
  ///////////////////

  if( flag_BNB_CCpi0_from_appnue || 1 ) {
    cout<<endl<<"      ---> flag_BNB_CCpi0_from_appnue"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_appnue.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_appnue_FC_eventinfo);
      str_treename = "tree_CCpi0_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_appnue_PC_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_appnue.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_appnue_FC_eventinfo);
      str_treename = "tree_CCpi0_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_appnue_PC_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_appnue.root";
      TString str_treename = "";
      str_treename = "tree_CCpi0_from_appnue_FC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_CCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_appnue_FC_eventinfo);
      str_treename = "tree_CCpi0_from_appnue_PC";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, NULL,                                    strfile_mc_e2e, str_treename, &vector_vector_BNB_CCpi0_from_appnue_PC_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_CCpi0_from_appnue_scaleFPOT, &vector_vector_BNB_CCpi0_from_appnue_FC_eventinfo, 19, "numu2nue");// hack
    Set_oscillation_base_minus(&vector_BNB_CCpi0_from_appnue_scaleFPOT, &vector_vector_BNB_CCpi0_from_appnue_PC_eventinfo, 20, "numu2nue");// hack
    
  }// if( flag_BNB_CCpi0_from_appnue )
    
  ///////////////////

  if( flag_BNB_NCpi0_from_appnue || 1 ) {
    cout<<endl<<"      ---> flag_BNB_NCpi0_from_appnue"<<endl;
    {// run1
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run1.root";
      TString strfile_dataPOT = str_dirbase + "run1_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run1_appnue.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_appnue";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_appnue_eventinfo);
    }
    {// run2
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run2.root";
      TString strfile_dataPOT = str_dirbase + "run2_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run2_appnue.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_appnue";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_appnue_eventinfo);
    }
    {// run3
      TString strfile_mcPOT   = str_dirbase + "checkout_prodgenie_bnb_numu2nue_overlay_run3.root";
      TString strfile_dataPOT = str_dirbase + "run3_data_bnb.root";
      TString strfile_mc_e2e  = str_dirbase + "roofile_obj_BNB_run3_appnue.root";
      TString str_treename = "";
      str_treename = "tree_NCpi0_from_appnue";
      Set_oscillation_base_subfunc(strfile_mcPOT, strfile_dataPOT, &vector_BNB_NCpi0_from_appnue_scaleFPOT, strfile_mc_e2e, str_treename, &vector_vector_BNB_NCpi0_from_appnue_eventinfo);
    }
    
    Set_oscillation_base_minus(&vector_BNB_NCpi0_from_appnue_scaleFPOT, &vector_vector_BNB_NCpi0_from_appnue_eventinfo, 21, "numu2nue");// hack
    
  }// if( flag_BNB_NCpi0_from_appnue )

  
  cout<<endl;
}
