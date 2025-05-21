from dataclasses import dataclass

@dataclass
class OscConfig:
    # ===== Default file paths =====
    cv_file: str = "/nfs/data/1/renney/python_TOsc/BNBNuMI_TOsc_input_numifluxpatched/merge.root"
    dirtadd_file: str = "/nfs/data/1/renney/python_TOsc/BNBNuMI_TOsc_input_numifluxpatched/merge.root"
    mcstat_file: str = "/nfs/data/1/renney/python_TOsc/BNBNuMI_TOsc_input_numifluxpatched/0.log"
    fluxXs_dir: str = "/nfs/data/1/renney/python_TOsc/BNBNuMI_TOsc_input_numifluxpatched/XsFlux_edit/"
    detector_dir: str = "/nfs/data/1/renney/python_TOsc/BNBNuMI_TOsc_input_numifluxpatched/DetVar_edit/"
    eventlist_dir: str = "/nfs/data/1/renney/python_TOsc/BNBNuMI_TOsc_input_numifluxpatched/hist_rootfiles/"

    # ===== Systematic flags =====
    flag_syst_dirt: bool = True
    flag_syst_mcstat: bool = True
    flag_syst_flux: bool = True
    flag_syst_geant: bool = True
    flag_syst_Xs: bool = True
    flag_syst_det: bool = True

    # ===== NuMI nueCC =====
    flag_NuMI_nueCC_from_intnue: bool = True
    flag_NuMI_nueCC_from_overlaynumu: bool = True
    flag_NuMI_nueCC_from_appnue: bool = True
    flag_NuMI_nueCC_from_appnumu: bool = False
    flag_NuMI_nueCC_from_dirtnue: bool = False
    flag_NuMI_nueCC_from_dirtnumu: bool = False
    flag_NuMI_nueCC_from_overlaynueNC: bool = True
    flag_NuMI_nueCC_from_overlaynumuNC: bool = True

    # ===== NuMI numuCC =====
    flag_NuMI_numuCC_from_overlaynumu: bool = True
    flag_NuMI_numuCC_from_overlaynue: bool = False
    flag_NuMI_numuCC_from_appnue: bool = True
    flag_NuMI_numuCC_from_appnumu: bool = False
    flag_NuMI_numuCC_from_dirtnue: bool = False
    flag_NuMI_numuCC_from_dirtnumu: bool = False
    flag_NuMI_numuCC_from_overlaynumuNC: bool = True
    flag_NuMI_numuCC_from_overlaynueNC: bool = True

    # ===== NuMI CCpi0 =====
    flag_NuMI_CCpi0_from_overlaynumu: bool = True
    flag_NuMI_CCpi0_from_overlaynue: bool = False
    flag_NuMI_CCpi0_from_appnue: bool = True
    flag_NuMI_CCpi0_from_appnumu: bool = False
    flag_NuMI_CCpi0_from_dirtnue: bool = False
    flag_NuMI_CCpi0_from_dirtnumu: bool = False
    flag_NuMI_CCpi0_from_overlaynumuNC: bool = True
    flag_NuMI_CCpi0_from_overlaynueNC: bool = True

    # ===== NuMI NCpi0 =====
    flag_NuMI_NCpi0_from_overlaynumu: bool = True
    flag_NuMI_NCpi0_from_overlaynue: bool = False
    flag_NuMI_NCpi0_from_appnue: bool = True
    flag_NuMI_NCpi0_from_appnumu: bool = False
    flag_NuMI_NCpi0_from_dirtnue: bool = False
    flag_NuMI_NCpi0_from_dirtnumu: bool = False
    flag_NuMI_NCpi0_from_overlaynumuNC: bool = True
    flag_NuMI_NCpi0_from_overlaynueNC: bool = True

    # ===== BNB nueCC =====
    flag_BNB_nueCC_from_intnue: bool = True
    flag_BNB_nueCC_from_overlaynumu: bool = True
    flag_BNB_nueCC_from_appnue: bool = True
    flag_BNB_nueCC_from_appnumu: bool = False
    flag_BNB_nueCC_from_dirtnue: bool = False
    flag_BNB_nueCC_from_dirtnumu: bool = False
    flag_BNB_nueCC_from_overlaynueNC: bool = True
    flag_BNB_nueCC_from_overlaynumuNC: bool = True

    # ===== BNB numuCC =====
    flag_BNB_numuCC_from_overlaynumu: bool = True
    flag_BNB_numuCC_from_overlaynue: bool = False
    flag_BNB_numuCC_from_appnue: bool = True
    flag_BNB_numuCC_from_appnumu: bool = False
    flag_BNB_numuCC_from_dirtnue: bool = False
    flag_BNB_numuCC_from_dirtnumu: bool = False
    flag_BNB_numuCC_from_overlaynumuNC: bool = True
    flag_BNB_numuCC_from_overlaynueNC: bool = True

    # ===== BNB CCpi0 =====
    flag_BNB_CCpi0_from_overlaynumu: bool = True
    flag_BNB_CCpi0_from_overlaynue: bool = False
    flag_BNB_CCpi0_from_appnue: bool = True
    flag_BNB_CCpi0_from_appnumu: bool = False
    flag_BNB_CCpi0_from_dirtnue: bool = False
    flag_BNB_CCpi0_from_dirtnumu: bool = False
    flag_BNB_CCpi0_from_overlaynumuNC: bool = True
    flag_BNB_CCpi0_from_overlaynueNC: bool = True

    # ===== BNB NCpi0 =====
    flag_BNB_NCpi0_from_overlaynumu: bool = True
    flag_BNB_NCpi0_from_overlaynue: bool = False
    flag_BNB_NCpi0_from_appnue: bool = True
    flag_BNB_NCpi0_from_appnumu: bool = False
    flag_BNB_NCpi0_from_dirtnue: bool = False
    flag_BNB_NCpi0_from_dirtnumu: bool = False
    flag_BNB_NCpi0_from_overlaynumuNC: bool = True
    flag_BNB_NCpi0_from_overlaynueNC: bool = True
