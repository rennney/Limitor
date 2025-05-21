import sys
import json
import click
import limitor
from .core import TOsc
from .configs.defaults import OscConfig as Configure_Osc


@click.group()
@click.option("-s","--store",type=click.Path(),
              envvar="Some_STORE",
              help="File for primary data storage (input/output)")
@click.option("-o","--outstore",type=click.Path(),
              help="File for output (primary only input)")
@click.pass_context
def cli(ctx, store, outstore):
    '''
    Limitor command line interface
    '''
    if not store:
        store = "."
    ctx.obj = limitor.main.Main(store, outstore)


@cli.command()
@click.option("-i", "--input", type=str, required=False, help="Optional input file")
@click.option("--idm2", type=int, default=0)
@click.option("--it14", type=int, default=0)
@click.option("--it24", type=int, default=0)
@click.option("--device", type=str, default="cpu")
@click.pass_context
def run_tosc(ctx, input, idm2, it14, it24, device):
    import torch
    """
    Run TOsc and perform FC scan
    """
    # === Instantiate and configure TOsc ===
    osc = TOsc()
    osc.scaleF_POT_BNB = 1.0
    osc.scaleF_POT_NuMI = 1.0

    # Systematic flags
    for flag_name in [
        "flag_syst_dirt", "flag_syst_mcstat", "flag_syst_flux",
        "flag_syst_geant", "flag_syst_Xs", "flag_syst_det"
    ]:
        setattr(osc, flag_name, getattr(Configure_Osc, flag_name))

    # Event category flags
    for flag in Configure_Osc.__dict__:
        if flag.startswith("flag_"):
            setattr(osc, flag, getattr(Configure_Osc, flag))

    # === Load data ===
    osc.Set_default_cv_cov(
        Configure_Osc.cv_file,
        Configure_Osc.dirtadd_file,
        Configure_Osc.mcstat_file,
        Configure_Osc.fluxXs_dir,
        Configure_Osc.detector_dir,
        device=osc.device
    )
    osc.set_oscillation_base(Configure_Osc.eventlist_dir)
    

    # === FC grid setup ===
    h1_dm2 = torch.linspace(-2, 2, 80)
    h1_sin2_theta14 = torch.linspace(-4, 0, 60)
    val_dm2 = 0.1
    val_sin2_theta14 = 0.13

    # === Load profile map ===
    # You must replace this with loading ROOT TH2F or similar from file
    #from some_module import load_profile_map  # placeholder
    #map_t24 = load_profile_map("/path/to/smoothe_appearance_pearson.root")
    theta24_prof = 1.0 #map_t24.lookup(log10(val_sin2_theta14), log10(val_dm2))

    osc.set_oscillation_pars(val_dm2, val_sin2_theta14, theta24_prof, 0)
    osc.apply_oscillation()
    osc.set_apply_POT()

    # === Generate pseudo-data ===
    osc.set_toy_variations(1)
    osc.set_toy2fitdata(1)

    pars_grid = torch.tensor([val_dm2, val_sin2_theta14, theta24_prof], device=device)
    chi2_grid = osc.FCN(pars_grid)

    # === Grid search for minimum ===
    chi2_min = 1e6
    dm2_min, s2_tmue_min, s2_t24_min = 0., 0., 0.
    for dm_bin in range(80):
        dm2_v = 10 ** h1_dm2[dm_bin]
        for j in range(60):
            s2_tmue_v = 10 ** h1_sin2_theta14[j]
            for i in range(60):
                s2_t24_v = 10 ** h1_sin2_theta14[i]
                if s2_tmue_v > s2_t24_v:
                    continue
                pars = torch.tensor([dm2_v, s2_tmue_v, s2_t24_v], device=device)
                chi2 = osc.FCN(pars)
                print(i,j,chi2)
                if chi2 < chi2_min:
                    chi2_min = chi2
                    dm2_min, s2_tmue_min, s2_t24_min = dm2_v, s2_tmue_v, s2_t24_v

    print("Grid scan result:", dm2_min, s2_tmue_min, s2_t24_min)

    # === Fit ===
    for retry in range(5):
        osc.Minimization_OscPars_FullCov(dm2_min, s2_tmue_min, s2_t24_min, 0, "")
        if osc.minimization_status == 0:
            break
        step = 0.06
        if retry < 2:
            dm2_min *= 1 + step
            s2_tmue_min *= 1 + step
            s2_t24_min *= 1 + step
        else:
            dm2_min *= 1 - step
            s2_tmue_min *= 1 - step
            s2_t24_min *= 1 - step

    # === Final chi2 and delta ===
    if osc.minimization_status == 0:
        final_pars = torch.tensor([
            osc.minimization_dm2_41_val,
            osc.minimization_sin2_2theta_14_val,
            osc.minimization_sin2_theta_24_val
        ], device=device)
        chi2_min = osc.FCN(final_pars).item()

    delta_chi2 = chi2_grid.item() - chi2_min

    # === Output ===
    print(f"Result: converged={osc.minimization_status == 0}, "
          f"chi2_grid={chi2_grid.item():.4f}, chi2_min={chi2_min:.4f}, delta_chi2={delta_chi2:.4f}")



    

def main():
    cli(obj=None)


if '__main__' == __name__:
    main()
