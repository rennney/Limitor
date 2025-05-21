import torch
import numpy as np
from .event_info import EventInfo
from ..config.defaults import default_config
from .oscillation import prob_oscillation
import uproot
import numpy as np
import os

class TOsc:
    def __init__(self):
        print("\n ---> Hello TOsc\n")

        # -----------------------------------------------------------
        # General setup
        # -----------------------------------------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rng = np.random.default_rng()

        # -----------------------------------------------------------
        # Oscillation parameters
        # -----------------------------------------------------------
        self.scaleF_POT_BNB = 1.0
        self.scaleF_POT_NuMI = 1.0
        self.dm2_41 = 0.0
        self.sin2_2theta_14 = 0.0
        self.sin2_theta_24 = 0.0
        self.sin2_theta_34 = 0.0

        # -----------------------------------------------------------
        # Minimization results
        # -----------------------------------------------------------
        self.minimization_status = -1
        self.minimization_chi2 = -1
        self.minimization_dm2_41_val = -1
        self.minimization_sin2_2theta_14_val = -1
        self.minimization_sin2_theta_24_val = -1
        self.minimization_sin2_theta_34_val = -1
        self.minimization_dm2_41_err = -1
        self.minimization_sin2_2theta_14_err = -1
        self.minimization_sin2_theta_24_err = -1
        self.minimization_sin2_theta_34_err = -1

        # -----------------------------------------------------------
        # Configuration flags (loaded from config module)
        # -----------------------------------------------------------
        for attr in vars(default_config):
            setattr(self, attr, getattr(default_config, attr))

        # -----------------------------------------------------------
        # Matrix and histogram metadata
        # -----------------------------------------------------------
        self.matrix_transform = None

        self.matrix_default_newworld_meas = None
        self.matrix_default_oldworld_pred = None
        self.matrix_oscillation_base_oldworld_pred = None
        self.matrix_oscillation_oldworld_pred = None

        self.matrix_default_oldworld_abs_syst_addi = None
        self.matrix_default_oldworld_rel_syst_flux = None
        self.matrix_default_oldworld_rel_syst_geant = None
        self.matrix_default_oldworld_rel_syst_Xs = None
        self.matrix_default_oldworld_rel_syst_det = None
        self.matrix_default_newworld_abs_syst_mcstat = None

        self.matrix_eff_newworld_abs_syst_addi = None
        self.matrix_eff_newworld_abs_syst_mcstat = None
        self.matrix_eff_newworld_abs_syst_flux = None
        self.matrix_eff_newworld_abs_syst_geant = None
        self.matrix_eff_newworld_abs_syst_Xs = None
        self.matrix_eff_newworld_abs_syst_det = None
        self.matrix_eff_newworld_abs_syst_total = None

        self.matrix_eff_newworld_meas = None
        self.matrix_eff_newworld_pred = None
        self.matrix_eff_newworld_noosc = None
        self.matrix_fitdata_newworld = None

        # -----------------------------------------------------------
        # Histogram bin metadata
        # -----------------------------------------------------------
        self.map_default_h1d_pred_bins = {}
        self.map_default_h1d_pred_xlow = {}
        self.map_default_h1d_pred_xhgh = {}
        self.map_default_newworld_meas_bins = {}
        self.map_default_oldworld_pred_bins = {}

        self.vector_default_newworld_meas = None
        self.vector_default_oldworld_pred = None

        # -----------------------------------------------------------
        # Toy generation storage
        # -----------------------------------------------------------
        self.NUM_TOYS = 0
        self.map_matrix_toy_pred = {}

        # -----------------------------------------------------------
        # Placeholder: Event info containers (these will be set externally)
        # Full list of these will match the C++ class exactly
        # You can initialize them to [] or use a helper dict if needed
        # -----------------------------------------------------------

        self.vector_BNB_nueCC_from_intnue_scaleFPOT = []
        self.vector_vector_BNB_nueCC_from_intnue_FC_eventinfo = []
        self.vector_vector_BNB_nueCC_from_intnue_PC_eventinfo = []
        self.vector_BNB_nueCC_from_overlaynumu_scaleFPOT = []
        self.vector_vector_BNB_nueCC_from_overlaynumu_FC_eventinfo = []
        self.vector_vector_BNB_nueCC_from_overlaynumu_PC_eventinfo = []
        self.vector_BNB_nueCC_from_appnue_scaleFPOT = []
        self.vector_vector_BNB_nueCC_from_appnue_FC_eventinfo = []
        self.vector_vector_BNB_nueCC_from_appnue_PC_eventinfo = []
        self.vector_BNB_nueCC_from_appnumu_scaleFPOT = []
        self.vector_vector_BNB_nueCC_from_appnumu_FC_eventinfo = []
        self.vector_vector_BNB_nueCC_from_appnumu_PC_eventinfo = []
        self.vector_BNB_nueCC_from_dirtnue_scaleFPOT = []
        self.vector_vector_BNB_nueCC_from_dirtnue_eventinfo = []
        self.vector_BNB_nueCC_from_dirtnumu_scaleFPOT = []
        self.vector_vector_BNB_nueCC_from_dirtnumu_eventinfo = []
        self.vector_BNB_nueCC_from_overlaynueNC_scaleFPOT = []
        self.vector_vector_BNB_nueCC_from_overlaynueNC_FC_eventinfo = []
        self.vector_vector_BNB_nueCC_from_overlaynueNC_PC_eventinfo = []
        self.vector_BNB_nueCC_from_overlaynumuNC_scaleFPOT = []
        self.vector_vector_BNB_nueCC_from_overlaynumuNC_FC_eventinfo = []
        self.vector_vector_BNB_nueCC_from_overlaynumuNC_PC_eventinfo = []
        self.vector_BNB_numuCC_from_overlaynumu_scaleFPOT = []
        self.vector_vector_BNB_numuCC_from_overlaynumu_FC_eventinfo = []
        self.vector_vector_BNB_numuCC_from_overlaynumu_PC_eventinfo = []
        self.vector_BNB_numuCC_from_overlaynue_scaleFPOT = []
        self.vector_vector_BNB_numuCC_from_overlaynue_FC_eventinfo = []
        self.vector_vector_BNB_numuCC_from_overlaynue_PC_eventinfo = []
        self.vector_BNB_numuCC_from_appnue_scaleFPOT = []
        self.vector_vector_BNB_numuCC_from_appnue_FC_eventinfo = []
        self.vector_vector_BNB_numuCC_from_appnue_PC_eventinfo = []
        self.vector_BNB_numuCC_from_appnumu_scaleFPOT = []
        self.vector_vector_BNB_numuCC_from_appnumu_FC_eventinfo = []
        self.vector_vector_BNB_numuCC_from_appnumu_PC_eventinfo = []
        self.vector_BNB_numuCC_from_dirtnue_scaleFPOT = []
        self.vector_vector_BNB_numuCC_from_dirtnue_eventinfo = []
        self.vector_BNB_numuCC_from_dirtnumu_scaleFPOT = []
        self.vector_vector_BNB_numuCC_from_dirtnumu_eventinfo = []
        self.vector_BNB_numuCC_from_overlaynumuNC_scaleFPOT = []
        self.vector_vector_BNB_numuCC_from_overlaynumuNC_FC_eventinfo = []
        self.vector_vector_BNB_numuCC_from_overlaynumuNC_PC_eventinfo = []
        self.vector_BNB_numuCC_from_overlaynueNC_scaleFPOT = []
        self.vector_vector_BNB_numuCC_from_overlaynueNC_FC_eventinfo = []
        self.vector_vector_BNB_numuCC_from_overlaynueNC_PC_eventinfo = []
        self.vector_BNB_CCpi0_from_overlaynumu_scaleFPOT = []
        self.vector_vector_BNB_CCpi0_from_overlaynumu_FC_eventinfo = []
        self.vector_vector_BNB_CCpi0_from_overlaynumu_PC_eventinfo = []
        self.vector_BNB_CCpi0_from_overlaynue_scaleFPOT = []
        self.vector_vector_BNB_CCpi0_from_overlaynue_eventinfo = []
        self.vector_BNB_CCpi0_from_appnue_scaleFPOT = []
        self.vector_vector_BNB_CCpi0_from_appnue_FC_eventinfo = []
        self.vector_vector_BNB_CCpi0_from_appnue_PC_eventinfo = []
        self.vector_BNB_CCpi0_from_appnumu_scaleFPOT = []
        self.vector_vector_BNB_CCpi0_from_appnumu_eventinfo = []
        self.vector_BNB_CCpi0_from_dirtnue_scaleFPOT = []
        self.vector_vector_BNB_CCpi0_from_dirtnue_eventinfo = []
        self.vector_BNB_CCpi0_from_dirtnumu_scaleFPOT = []
        self.vector_vector_BNB_CCpi0_from_dirtnumu_eventinfo = []
        self.vector_BNB_CCpi0_from_overlaynumuNC_scaleFPOT = []
        self.vector_vector_BNB_CCpi0_from_overlaynumuNC_FC_eventinfo = []
        self.vector_vector_BNB_CCpi0_from_overlaynumuNC_PC_eventinfo = []
        self.vector_BNB_CCpi0_from_overlaynueNC_scaleFPOT = []
        self.vector_vector_BNB_CCpi0_from_overlaynueNC_FC_eventinfo = []
        self.vector_vector_BNB_CCpi0_from_overlaynueNC_PC_eventinfo = []
        self.vector_BNB_NCpi0_from_overlaynumu_scaleFPOT = []
        self.vector_vector_BNB_NCpi0_from_overlaynumu_eventinfo = []
        self.vector_BNB_NCpi0_from_overlaynue_scaleFPOT = []
        self.vector_vector_BNB_NCpi0_from_overlaynue_eventinfo = []
        self.vector_BNB_NCpi0_from_appnue_scaleFPOT = []
        self.vector_vector_BNB_NCpi0_from_appnue_eventinfo = []
        self.vector_BNB_NCpi0_from_appnumu_scaleFPOT = []
        self.vector_vector_BNB_NCpi0_from_appnumu_eventinfo = []
        self.vector_BNB_NCpi0_from_dirtnue_scaleFPOT = []
        self.vector_vector_BNB_NCpi0_from_dirtnue_eventinfo = []
        self.vector_BNB_NCpi0_from_dirtnumu_scaleFPOT = []
        self.vector_vector_BNB_NCpi0_from_dirtnumu_eventinfo = []
        self.vector_BNB_NCpi0_from_overlaynumuNC_scaleFPOT = []
        self.vector_vector_BNB_NCpi0_from_overlaynumuNC_eventinfo = []
        self.vector_BNB_NCpi0_from_overlaynueNC_scaleFPOT = []
        self.vector_vector_BNB_NCpi0_from_overlaynueNC_eventinfo = []

        # === Repeat for NuMI ===

        self.vector_NuMI_nueCC_from_intnue_scaleFPOT = []
        self.vector_vector_NuMI_nueCC_from_intnue_FC_eventinfo = []
        self.vector_vector_NuMI_nueCC_from_intnue_PC_eventinfo = []
        self.vector_NuMI_nueCC_from_overlaynumu_scaleFPOT = []
        self.vector_vector_NuMI_nueCC_from_overlaynumu_FC_eventinfo = []
        self.vector_vector_NuMI_nueCC_from_overlaynumu_PC_eventinfo = []
        self.vector_NuMI_nueCC_from_appnue_scaleFPOT = []
        self.vector_vector_NuMI_nueCC_from_appnue_FC_eventinfo = []
        self.vector_vector_NuMI_nueCC_from_appnue_PC_eventinfo = []
        self.vector_NuMI_nueCC_from_appnumu_scaleFPOT = []
        self.vector_vector_NuMI_nueCC_from_appnumu_FC_eventinfo = []
        self.vector_vector_NuMI_nueCC_from_appnumu_PC_eventinfo = []
        self.vector_NuMI_nueCC_from_dirtnue_scaleFPOT = []
        self.vector_vector_NuMI_nueCC_from_dirtnue_eventinfo = []
        self.vector_NuMI_nueCC_from_dirtnumu_scaleFPOT = []
        self.vector_vector_NuMI_nueCC_from_dirtnumu_eventinfo = []
        self.vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT = []
        self.vector_vector_NuMI_nueCC_from_overlaynueNC_FC_eventinfo = []
        self.vector_vector_NuMI_nueCC_from_overlaynueNC_PC_eventinfo = []
        self.vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT = []
        self.vector_vector_NuMI_nueCC_from_overlaynumuNC_FC_eventinfo = []
        self.vector_vector_NuMI_nueCC_from_overlaynumuNC_PC_eventinfo = []
        self.vector_NuMI_numuCC_from_overlaynumu_scaleFPOT = []
        self.vector_vector_NuMI_numuCC_from_overlaynumu_FC_eventinfo = []
        self.vector_vector_NuMI_numuCC_from_overlaynumu_PC_eventinfo = []
        self.vector_NuMI_numuCC_from_overlaynue_scaleFPOT = []
        self.vector_vector_NuMI_numuCC_from_overlaynue_FC_eventinfo = []
        self.vector_vector_NuMI_numuCC_from_overlaynue_PC_eventinfo = []
        self.vector_NuMI_numuCC_from_appnue_scaleFPOT = []
        self.vector_vector_NuMI_numuCC_from_appnue_FC_eventinfo = []
        self.vector_vector_NuMI_numuCC_from_appnue_PC_eventinfo = []
        self.vector_NuMI_numuCC_from_appnumu_scaleFPOT = []
        self.vector_vector_NuMI_numuCC_from_appnumu_FC_eventinfo = []
        self.vector_vector_NuMI_numuCC_from_appnumu_PC_eventinfo = []
        self.vector_NuMI_numuCC_from_dirtnue_scaleFPOT = []
        self.vector_vector_NuMI_numuCC_from_dirtnue_eventinfo = []
        self.vector_NuMI_numuCC_from_dirtnumu_scaleFPOT = []
        self.vector_vector_NuMI_numuCC_from_dirtnumu_eventinfo = []
        self.vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT = []
        self.vector_vector_NuMI_numuCC_from_overlaynumuNC_FC_eventinfo = []
        self.vector_vector_NuMI_numuCC_from_overlaynumuNC_PC_eventinfo = []
        self.vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT = []
        self.vector_vector_NuMI_numuCC_from_overlaynueNC_FC_eventinfo = []
        self.vector_vector_NuMI_numuCC_from_overlaynueNC_PC_eventinfo = []
        self.vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT = []
        self.vector_vector_NuMI_CCpi0_from_overlaynumu_FC_eventinfo = []
        self.vector_vector_NuMI_CCpi0_from_overlaynumu_PC_eventinfo = []
        self.vector_NuMI_CCpi0_from_overlaynue_scaleFPOT = []
        self.vector_vector_NuMI_CCpi0_from_overlaynue_eventinfo = []
        self.vector_NuMI_CCpi0_from_appnue_scaleFPOT = []
        self.vector_vector_NuMI_CCpi0_from_appnue_FC_eventinfo = []
        self.vector_vector_NuMI_CCpi0_from_appnue_PC_eventinfo = []
        self.vector_NuMI_CCpi0_from_appnumu_scaleFPOT = []
        self.vector_vector_NuMI_CCpi0_from_appnumu_eventinfo = []
        self.vector_NuMI_CCpi0_from_dirtnue_scaleFPOT = []
        self.vector_vector_NuMI_CCpi0_from_dirtnue_eventinfo = []
        self.vector_NuMI_CCpi0_from_dirtnumu_scaleFPOT = []
        self.vector_vector_NuMI_CCpi0_from_dirtnumu_eventinfo = []
        self.vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT = []
        self.vector_vector_NuMI_CCpi0_from_overlaynumuNC_FC_eventinfo = []
        self.vector_vector_NuMI_CCpi0_from_overlaynumuNC_PC_eventinfo = []
        self.vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT = []
        self.vector_vector_NuMI_CCpi0_from_overlaynueNC_FC_eventinfo = []
        self.vector_vector_NuMI_CCpi0_from_overlaynueNC_PC_eventinfo = []
        self.vector_NuMI_NCpi0_from_overlaynumu_scaleFPOT = []
        self.vector_vector_NuMI_NCpi0_from_overlaynumu_eventinfo = []
        self.vector_NuMI_NCpi0_from_overlaynue_scaleFPOT = []
        self.vector_vector_NuMI_NCpi0_from_overlaynue_eventinfo = []
        self.vector_NuMI_NCpi0_from_appnue_scaleFPOT = []
        self.vector_vector_NuMI_NCpi0_from_appnue_eventinfo = []
        self.vector_NuMI_NCpi0_from_appnumu_scaleFPOT = []
        self.vector_vector_NuMI_NCpi0_from_appnumu_eventinfo = []
        self.vector_NuMI_NCpi0_from_dirtnue_scaleFPOT = []
        self.vector_vector_NuMI_NCpi0_from_dirtnue_eventinfo = []
        self.vector_NuMI_NCpi0_from_dirtnumu_scaleFPOT = []
        self.vector_vector_NuMI_NCpi0_from_dirtnumu_eventinfo = []
        self.vector_NuMI_NCpi0_from_overlaynumuNC_scaleFPOT = []
        self.vector_vector_NuMI_NCpi0_from_overlaynumuNC_eventinfo = []
        self.vector_NuMI_NCpi0_from_overlaynueNC_scaleFPOT = []
        self.vector_vector_NuMI_NCpi0_from_overlaynueNC_eventinfo = []

        # -----------------------------------------------------------
        # Matrix shape info
        # -----------------------------------------------------------
        self.default_oldworld_rows = 0
        self.default_newworld_rows = 0

    def FCN(self, par: torch.Tensor) -> torch.Tensor:
        """
        Fully GPU-compatible Feldman-Cousins chi² function.

        Parameters:
            par (Tensor): shape (3,), [dm2_41, sin2_2theta_14, sin2_theta_24]
            mode (str): 'full', 'bnb_only', 'numi_only', or 'nue_both'

        Returns:
            chi2 (Tensor): scalar chi² value
        """

        # 1. Set oscillation
        self.Set_oscillation_pars(par[0], par[1], par[2], 0)
        self.Apply_oscillation()
        self.Set_apply_POT()

        # 2. Choose slice
        data = self.matrix_fitdata_newworld[0]
        pred = self.matrix_eff_newworld_pred[0]
        cov = self.matrix_eff_newworld_abs_syst_total
        
        mode = 'full'
        
        if mode == 'bnb_only':
            idx = slice(0, 26*7)
        elif mode == 'numi_only':
            idx = slice(26*7, 26*14)
        elif mode == 'nue_both':
            # nue bins: [0..25] and [26*7..26*7+25]
            idxs = torch.cat([torch.arange(26), torch.arange(26*7, 26*7+26)]).to(data.device)
            data = data[idxs]
            pred = pred[idxs]
            cov = cov[idxs][:, idxs]
        else:
            idx = slice(None)  # full

        if mode in {'bnb_only', 'numi_only'}:
            data = data[idx]
            pred = pred[idx]
            cov = cov[idx, idx]

        N = data.shape[0]

        # 3. Statistical uncertainties
        val_stat_cov = torch.zeros(N, device=pred.device)
        mask_data_zero = data == 0
        mask_pred_nonzero = pred != 0
        combined = (~mask_data_zero) & mask_pred_nonzero

        val_stat_cov[mask_data_zero] = pred[mask_data_zero] / 2
        val_stat_cov[combined] = 3. / (1. / data[combined] + 2. / pred[combined])
        val_stat_cov[~mask_data_zero & ~mask_pred_nonzero] = data[~mask_data_zero & ~mask_pred_nonzero]

        val_stat_cov = torch.clamp(val_stat_cov, min=1e-6)

        diag_cov = torch.diag(val_stat_cov)
        cov = cov.clone()
        cov[torch.diag_indices(N)] = torch.clamp(cov.diagonal(), min=1e-6)
        cov_total = cov + diag_cov

        # 4. Chi² = delta.T @ inv(C) @ delta
        delta = pred - data
        L = torch.linalg.cholesky(cov_total)
        z = torch.cholesky_solve(delta.unsqueeze(1), L)  # shape (N,1)
        chi2 = (delta.unsqueeze(1).T @ z).squeeze()  # scalar

        return chi2


    def Minimization_OscPars_FullCov(self, init_dm2_41, init_sin2_2theta_14, init_sin2_theta_24, init_sin2_theta_34, flag_fixpar, device='cuda'):

        print("\n ---> Minimization_OscPars_FullCov (Torch-GPU)")

        # Init parameter vector
        init_vals = torch.tensor([init_dm2_41, init_sin2_2theta_14, init_sin2_theta_24], dtype=torch.float32, device=device)

        # Which parameters are fixed
        fix_mask = torch.tensor([
            'dm2' in flag_fixpar,
            't14' in flag_fixpar,
            't24' in flag_fixpar
        ], dtype=torch.bool, device=device)

        # Select free parameters
        free_init_vals = init_vals[~fix_mask].clone().detach().requires_grad_(True)

        # Optimizer (LBFGS is second-order, like Minuit2)
        optimizer = torch.optim.LBFGS([free_init_vals], max_iter=200, tolerance_grad=1e-6, tolerance_change=1e-9, line_search_fn='strong_wolfe')

        # Closure function
        def closure():
            optimizer.zero_grad()
            full_params = init_vals.clone()
            full_params[~fix_mask] = free_init_vals
            chi2 = self.FCN(full_params)
            chi2.backward()
            return chi2

        # Run optimizer
        try:
            optimizer.step(closure)
            full_params = init_vals.clone()
            full_params[~fix_mask] = free_init_vals.detach()

            self.minimization_status = 0
            self.minimization_chi2 = self.FCN(full_params).item()
        except Exception as e:
            print(f"Minimization failed: {e}")
            self.minimization_status = 999
            self.minimization_chi2 = float('nan')
            full_params = torch.tensor([float('nan')] * 3, device=device)

        # Store final values (errors are not estimated here)
        self.minimization_dm2_41_val = full_params[0].item()
        self.minimization_sin2_2theta_14_val = full_params[1].item()
        self.minimization_sin2_theta_24_val = full_params[2].item()

        self.minimization_dm2_41_err = float('nan')
        self.minimization_sin2_2theta_14_err = float('nan')
        self.minimization_sin2_theta_24_err = float('nan')

        # Check for NaNs
        if torch.isnan(full_params).any():
            self.minimization_status = 123

        # Print results
        print(f" ---> minimization, status {self.minimization_status}, chi2 {self.minimization_chi2:.4f}, "
              f"dm2 {full_params[0].item():.4f}, s22t14 {full_params[1].item():.4f}, s2t24 {full_params[2].item():.4f}")

    def Set_toy_variations(self, num_toys, device='cuda'):
        print(f"\n ---> Set_toy_variations with {num_toys} toys")

        self.map_matrix_toy_pred.clear()
        self.NUM_TOYS = num_toys

        # Step 1: Covariance matrix (Symmetric)
        cov = self.matrix_eff_newworld_abs_syst_total.clone()

        # Step 2: Eigen-decomposition
        eigenval, eigenvec = torch.linalg.eigh(cov)
        eigenval = torch.clamp(eigenval, min=0.0)  # Ensure non-negative

        # Step 3: Prepare CV prediction
        pred_cv = self.matrix_eff_newworld_pred.clone()[0]  # (N,)

        # Step 4: Precompute sqrt(eigenval) to use in sampling
        sqrt_eigenval = eigenval.sqrt()
        N = self.default_newworld_rows

        # Step 5: Generate toys (loop for rejection)
        i = 1
        while i <= num_toys:
            # Draw from standard normal and scale by sqrt(eigenval)
            z = torch.randn(N, device=device)
            z_scaled = z * sqrt_eigenval

            # Rotate to new basis: toy variation
            toy_var = eigenvec @ z_scaled  # (N,)
            toy_full = pred_cv + toy_var

            # Constraint: all toy bins must be non-negative
            if (toy_full < 0).any():
                # Check hack condition
                hack_mask = (pred_cv < 0.8) & (toy_full < 0)
                toy_var[hack_mask] *= -1
                toy_full = pred_cv + toy_var
                if (toy_full < 0).any():
                    continue  # Reject toy and retry

            # Poisson sampling
            toy_smeared = torch.poisson(toy_full)

            # Store toy prediction as a 1xN matrix
            toy_matrix = toy_smeared.unsqueeze(0)
            self.map_matrix_toy_pred[i] = toy_matrix
            i += 1


    def Set_apply_POT(self):
        # Clone tensors (no inplace mutation issues)
        self.matrix_eff_newworld_meas = self.matrix_default_newworld_meas.clone()
        self.matrix_eff_newworld_pred.zero_()
        self.matrix_eff_newworld_abs_syst_total.zero_()

        # Guard against negative predictions
        mask_neg = self.matrix_oscillation_oldworld_pred[0] < 0
        if torch.any(mask_neg):
            clipped = self.matrix_oscillation_oldworld_pred[0].clone()
            bad_vals = clipped[mask_neg & (clipped > -0.1)]
            clipped[mask_neg] = torch.where(bad_vals.bool(), torch.tensor(0.0, device=clipped.device), clipped[mask_neg])
            self.matrix_oscillation_oldworld_pred[0] = clipped
            if torch.any(clipped < 0):
                print(" -------> WARNING: matrix_oscillation_oldworld_pred still < 0")

        # POT scaling
        scaleF_POT_BNB, scaleF_POT_NuMI = 1.0, 1.0
        cutoff_old = 26 * 21
        cutoff_new = 26 * 7

        pred = self.matrix_oscillation_oldworld_pred.clone()
        addi = self.matrix_default_oldworld_abs_syst_addi.clone()

        mask_old = torch.arange(self.default_oldworld_rows) < cutoff_old
        mask_new = torch.arange(self.default_newworld_rows) < cutoff_new

        pred[0, mask_old] *= scaleF_POT_BNB
        pred[0, ~mask_old] *= scaleF_POT_NuMI

        addi[mask_old, mask_old] *= scaleF_POT_BNB ** 2
        addi[~mask_old, ~mask_old] *= scaleF_POT_NuMI ** 2

        self.matrix_eff_newworld_meas[0, mask_new] *= scaleF_POT_BNB
        self.matrix_eff_newworld_meas[0, ~mask_new] *= scaleF_POT_NuMI

        mcstat = self.matrix_default_newworld_abs_syst_mcstat.clone()
        idx = torch.arange(self.default_newworld_rows)
        mcstat[idx[mask_new], idx[mask_new]] *= scaleF_POT_BNB ** 2
        mcstat[idx[~mask_new], idx[~mask_new]] *= scaleF_POT_NuMI ** 2

        # Prediction projection
        self.matrix_eff_newworld_pred = pred @ self.matrix_transform

        # Construct systematics from relative ones
        cv = pred[0]
        cv_outer = torch.ger(cv, cv)  # outer product
        flux_abs  = cv_outer * self.matrix_default_oldworld_rel_syst_flux
        geant_abs = cv_outer * self.matrix_default_oldworld_rel_syst_geant
        Xs_abs    = cv_outer * self.matrix_default_oldworld_rel_syst_Xs
        det_abs   = cv_outer * self.matrix_default_oldworld_rel_syst_det

        # Efficient projection to newworld
        A = self.matrix_transform
        At = A.T
        syst_total_oldworld = torch.zeros_like(addi)
        if getattr(self, 'flag_syst_dirt', False):  syst_total_oldworld += addi
        if getattr(self, 'flag_syst_flux', False):  syst_total_oldworld += flux_abs
        if getattr(self, 'flag_syst_geant', False): syst_total_oldworld += geant_abs
        if getattr(self, 'flag_syst_Xs', False):    syst_total_oldworld += Xs_abs
        if getattr(self, 'flag_syst_det', False):   syst_total_oldworld += det_abs

        # Block optimization (SpeedUp section)
        n = self.default_oldworld_rows
        block = n // 6
        def blk(i, j): return syst_total_oldworld[i*block:(i+1)*block, j*block:(j+1)*block]
        BB = sum([blk(i, j) for i in range(3) for j in range(3)])
        NN = sum([blk(i+3, j+3) for i in range(3) for j in range(3)])
        BN = sum([blk(i, j+3) for i in range(3) for j in range(3)])
        NB = BN.T

        total = torch.zeros((block*2, block*2), dtype=torch.float32, device=BB.device)
        total[:block, :block] = BB
        total[block:, block:] = NN
        total[:block, block:] = BN
        total[block:, :block] = NB

        self.matrix_eff_newworld_abs_syst_total = total

        # Add MCstat last
        if getattr(self, 'flag_syst_mcstat', False):
            self.matrix_eff_newworld_abs_syst_total += mcstat


    def set_oscillation_base_minus(self, vec_ratioPOT, vec_vec_eventinfo, pred_channel_index, osc_mode):
        """
        Subtracts contribution to matrix_oscillation_base_oldworld_pred
        using vectorized histogramming of event reco energies.

        Args:
            vec_ratioPOT (list of float)
            vec_vec_eventinfo (list of list of EventInfo)
            pred_channel_index (int): index into self.map_default_h1d_pred
            osc_mode (str): not used directly, but kept for compatibility/logging
        """
        total_pred_chs = len(self.map_default_h1d_pred)
        if pred_channel_index > total_pred_chs:
            raise IndexError(f"pred_channel_index({pred_channel_index}) > total_pred_chs({total_pred_chs})")

        print(f"\n ---> (self-check) work on pred_channel: {pred_channel_index:3d},  oscillation mode: {osc_mode}")

        for isize, event_group in enumerate(vec_vec_eventinfo):
            # get histogram metadata
            h1d_ref = self.map_default_h1d_pred[pred_channel_index]
            nbins = self.map_default_h1d_pred_bins[pred_channel_index]
            xlow = self.map_default_h1d_pred_xlow[pred_channel_index]
            xhigh = self.map_default_h1d_pred_xhgh[pred_channel_index]
            bin_width = (xhigh - xlow) / nbins

            # vectorize event fill
            ereco = torch.tensor([ev.e2e_Ereco for ev in event_group], device=self.device)
            weights = torch.tensor([ev.e2e_weight_xs for ev in event_group], device=self.device)

            bin_indices = ((ereco - xlow) / bin_width).long()
            bin_indices = torch.clamp(bin_indices, 0, nbins)  # under/overflow allowed

            hist_tensor = torch.zeros(nbins + 1, device=self.device).scatter_add_(0, bin_indices, weights)
            hist_tensor *= vec_ratioPOT[isize]

            # embed into full oldworld row
            bin_index_base = 0
            for ich in range(1, pred_channel_index):
                bin_index_base += self.map_default_h1d_pred[ich].GetNbinsX() + 1

            matrix_temp = torch.zeros((1, self.default_oldworld_rows), device=self.device)
            matrix_temp[0, bin_index_base : bin_index_base + nbins + 1] = hist_tensor

            self.matrix_oscillation_base_oldworld_pred -= matrix_temp

    def set_oscillation_base_added(self, vec_ratioPOT, vec_vec_eventinfo, pred_channel_index, str_osc_mode):
        """
        Adds predicted events (with oscillation) into matrix_oscillation_oldworld_pred.

        Args:
            vec_ratioPOT (list of float)
            vec_vec_eventinfo (list of list of EventInfo)
            pred_channel_index (int)
            str_osc_mode (str): e.g., "nue2nue", "numu2nue", etc.
        """
        total_pred_chs = len(self.map_default_h1d_pred)
        if pred_channel_index > total_pred_chs:
            raise IndexError(f"pred_channel_index({pred_channel_index}) > total_pred_chs({total_pred_chs})")

        # Map oscillation mode to flag
        OSC_MODE = {
            "nue2nue": 1,
            "numu2numu": 2,
            "numu2nue": 3,
            "nue2numu": 4,
            "nueNC": 5,
            "numuNC": 6,
        }
        if str_osc_mode not in OSC_MODE:
            raise ValueError(f"Unknown osc mode '{str_osc_mode}'. Must be one of {list(OSC_MODE)}")
        flag_osc = OSC_MODE[str_osc_mode]

        for line_user, event_group in enumerate(vec_vec_eventinfo):
            nbins = self.map_default_h1d_pred_bins[pred_channel_index]
            xlow = self.map_default_h1d_pred_xlow[pred_channel_index]
            xhigh = self.map_default_h1d_pred_xhgh[pred_channel_index]
            bin_width = (xhigh - xlow) / nbins

            # Extract event features
            e_true = torch.tensor([ev.e2e_Etrue for ev in event_group], device=self.device)
            baseline = torch.tensor([ev.e2e_baseline for ev in event_group], device=self.device)
            e_reco = torch.tensor([ev.e2e_Ereco for ev in event_group], device=self.device)
            weights = torch.tensor([ev.e2e_weight_xs for ev in event_group], device=self.device)

            # Oscillation probability
            p_osc = prob_oscillation(e_true, baseline, self.dm2_41, self.sin2_2theta_14, self.sin2_theta_24, flag_osc)
            total_weight = p_osc * weights

            # Histogram fill (with probability-weighted reco energy)
            bin_indices = ((e_reco - xlow) / bin_width).long()
            bin_indices = torch.clamp(bin_indices, 0, nbins)
            hist_tensor = torch.zeros(nbins + 1, device=self.device).scatter_add_(0, bin_indices, total_weight)

            # POT scaling
            hist_tensor *= vec_ratioPOT[line_user]

            # Offset into oldworld matrix
            bin_index_base = 0
            for ich in range(1, pred_channel_index):
                bin_index_base += self.map_default_h1d_pred[ich].GetNbinsX() + 1

            matrix_temp = torch.zeros((1, self.default_oldworld_rows), device=self.device)
            matrix_temp[0, bin_index_base : bin_index_base + nbins + 1] = hist_tensor

            self.matrix_oscillation_oldworld_pred += matrix_temp

    def apply_oscillation(self):
        """
        Resets oscillation prediction to base, then adds contributions
        from flagged channels using self.set_oscillation_base_added()
        """
        self.matrix_oscillation_oldworld_pred = self.matrix_oscillation_base_oldworld_pred.clone()

        def maybe_add(flag, scale, eventinfo, index, mode):
            if flag:
                self.set_oscillation_base_added(scale, eventinfo, index, mode)

        # NuMI
        maybe_add(self.flag_NuMI_nueCC_from_intnue, self.vector_NuMI_nueCC_from_intnue_scaleFPOT, self.vector_vector_NuMI_nueCC_from_intnue_FC_eventinfo, 22, "nue2nue")
        maybe_add(self.flag_NuMI_nueCC_from_intnue, self.vector_NuMI_nueCC_from_intnue_scaleFPOT, self.vector_vector_NuMI_nueCC_from_intnue_PC_eventinfo, 23, "nue2nue")

        maybe_add(self.flag_NuMI_nueCC_from_overlaynumu, self.vector_NuMI_nueCC_from_overlaynumu_scaleFPOT, self.vector_vector_NuMI_nueCC_from_overlaynumu_FC_eventinfo, 22, "numu2numu")
        maybe_add(self.flag_NuMI_nueCC_from_overlaynumu, self.vector_NuMI_nueCC_from_overlaynumu_scaleFPOT, self.vector_vector_NuMI_nueCC_from_overlaynumu_PC_eventinfo, 23, "numu2numu")

        maybe_add(self.flag_NuMI_numuCC_from_overlaynumu, self.vector_NuMI_numuCC_from_overlaynumu_scaleFPOT, self.vector_vector_NuMI_numuCC_from_overlaynumu_FC_eventinfo, 24, "numu2numu")
        maybe_add(self.flag_NuMI_numuCC_from_overlaynumu, self.vector_NuMI_numuCC_from_overlaynumu_scaleFPOT, self.vector_vector_NuMI_numuCC_from_overlaynumu_PC_eventinfo, 25, "numu2numu")

        maybe_add(self.flag_NuMI_CCpi0_from_overlaynumu, self.vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT, self.vector_vector_NuMI_CCpi0_from_overlaynumu_FC_eventinfo, 26, "numu2numu")
        maybe_add(self.flag_NuMI_CCpi0_from_overlaynumu, self.vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT, self.vector_vector_NuMI_CCpi0_from_overlaynumu_PC_eventinfo, 27, "numu2numu")

        maybe_add(self.flag_NuMI_NCpi0_from_overlaynumu, self.vector_NuMI_NCpi0_from_overlaynumu_scaleFPOT, self.vector_vector_NuMI_NCpi0_from_overlaynumu_eventinfo, 28, "numu2numu")

        maybe_add(self.flag_NuMI_nueCC_from_overlaynueNC, self.vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT, self.vector_vector_NuMI_nueCC_from_overlaynueNC_FC_eventinfo, 22, "nueNC")
        maybe_add(self.flag_NuMI_nueCC_from_overlaynueNC, self.vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT, self.vector_vector_NuMI_nueCC_from_overlaynueNC_PC_eventinfo, 23, "nueNC")

        maybe_add(self.flag_NuMI_nueCC_from_overlaynumuNC, self.vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT, self.vector_vector_NuMI_nueCC_from_overlaynumuNC_FC_eventinfo, 22, "numuNC")
        maybe_add(self.flag_NuMI_nueCC_from_overlaynumuNC, self.vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT, self.vector_vector_NuMI_nueCC_from_overlaynumuNC_PC_eventinfo, 23, "numuNC")

        maybe_add(self.flag_NuMI_numuCC_from_overlaynueNC, self.vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT, self.vector_vector_NuMI_numuCC_from_overlaynueNC_FC_eventinfo, 24, "nueNC")
        maybe_add(self.flag_NuMI_numuCC_from_overlaynueNC, self.vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT, self.vector_vector_NuMI_numuCC_from_overlaynueNC_PC_eventinfo, 25, "nueNC")

        maybe_add(self.flag_NuMI_numuCC_from_overlaynumuNC, self.vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT, self.vector_vector_NuMI_numuCC_from_overlaynumuNC_FC_eventinfo, 24, "numuNC")
        maybe_add(self.flag_NuMI_numuCC_from_overlaynumuNC, self.vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT, self.vector_vector_NuMI_numuCC_from_overlaynumuNC_PC_eventinfo, 25, "numuNC")

        maybe_add(self.flag_NuMI_CCpi0_from_overlaynueNC, self.vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT, self.vector_vector_NuMI_CCpi0_from_overlaynueNC_FC_eventinfo, 26, "nueNC")
        maybe_add(self.flag_NuMI_CCpi0_from_overlaynueNC, self.vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT, self.vector_vector_NuMI_CCpi0_from_overlaynueNC_PC_eventinfo, 27, "nueNC")

        maybe_add(self.flag_NuMI_CCpi0_from_overlaynumuNC, self.vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT, self.vector_vector_NuMI_CCpi0_from_overlaynumuNC_FC_eventinfo, 26, "numuNC")
        maybe_add(self.flag_NuMI_CCpi0_from_overlaynumuNC, self.vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT, self.vector_vector_NuMI_CCpi0_from_overlaynumuNC_PC_eventinfo, 27, "numuNC")

        maybe_add(self.flag_NuMI_NCpi0_from_overlaynueNC, self.vector_NuMI_NCpi0_from_overlaynueNC_scaleFPOT, self.vector_vector_NuMI_NCpi0_from_overlaynueNC_eventinfo, 28, "nueNC")
        maybe_add(self.flag_NuMI_NCpi0_from_overlaynumuNC, self.vector_NuMI_NCpi0_from_overlaynumuNC_scaleFPOT, self.vector_vector_NuMI_NCpi0_from_overlaynumuNC_eventinfo, 28, "numuNC")

        maybe_add(self.flag_NuMI_nueCC_from_appnue, self.vector_NuMI_nueCC_from_appnue_scaleFPOT, self.vector_vector_NuMI_nueCC_from_appnue_FC_eventinfo, 36, "numu2nue")
        maybe_add(self.flag_NuMI_nueCC_from_appnue, self.vector_NuMI_nueCC_from_appnue_scaleFPOT, self.vector_vector_NuMI_nueCC_from_appnue_PC_eventinfo, 37, "numu2nue")

        maybe_add(self.flag_NuMI_numuCC_from_appnue, self.vector_NuMI_numuCC_from_appnue_scaleFPOT, self.vector_vector_NuMI_numuCC_from_appnue_FC_eventinfo, 38, "numu2nue")
        maybe_add(self.flag_NuMI_numuCC_from_appnue, self.vector_NuMI_numuCC_from_appnue_scaleFPOT, self.vector_vector_NuMI_numuCC_from_appnue_PC_eventinfo, 39, "numu2nue")

        maybe_add(self.flag_NuMI_CCpi0_from_appnue, self.vector_NuMI_CCpi0_from_appnue_scaleFPOT, self.vector_vector_NuMI_CCpi0_from_appnue_FC_eventinfo, 40, "numu2nue")
        maybe_add(self.flag_NuMI_CCpi0_from_appnue, self.vector_NuMI_CCpi0_from_appnue_scaleFPOT, self.vector_vector_NuMI_CCpi0_from_appnue_PC_eventinfo, 41, "numu2nue")

        maybe_add(self.flag_NuMI_NCpi0_from_appnue, self.vector_NuMI_NCpi0_from_appnue_scaleFPOT, self.vector_vector_NuMI_NCpi0_from_appnue_eventinfo, 42, "numu2nue")

        # BNB
        maybe_add(self.flag_BNB_nueCC_from_intnue, self.vector_BNB_nueCC_from_intnue_scaleFPOT, self.vector_vector_BNB_nueCC_from_intnue_FC_eventinfo, 1, "nue2nue")
        maybe_add(self.flag_BNB_nueCC_from_intnue, self.vector_BNB_nueCC_from_intnue_scaleFPOT, self.vector_vector_BNB_nueCC_from_intnue_PC_eventinfo, 2, "nue2nue")

        maybe_add(self.flag_BNB_nueCC_from_overlaynumu, self.vector_BNB_nueCC_from_overlaynumu_scaleFPOT, self.vector_vector_BNB_nueCC_from_overlaynumu_FC_eventinfo, 1, "numu2numu")
        maybe_add(self.flag_BNB_nueCC_from_overlaynumu, self.vector_BNB_nueCC_from_overlaynumu_scaleFPOT, self.vector_vector_BNB_nueCC_from_overlaynumu_PC_eventinfo, 2, "numu2numu")

        maybe_add(self.flag_BNB_numuCC_from_overlaynumu, self.vector_BNB_numuCC_from_overlaynumu_scaleFPOT, self.vector_vector_BNB_numuCC_from_overlaynumu_FC_eventinfo, 3, "numu2numu")
        maybe_add(self.flag_BNB_numuCC_from_overlaynumu, self.vector_BNB_numuCC_from_overlaynumu_scaleFPOT, self.vector_vector_BNB_numuCC_from_overlaynumu_PC_eventinfo, 4, "numu2numu")

        maybe_add(self.flag_BNB_CCpi0_from_overlaynumu, self.vector_BNB_CCpi0_from_overlaynumu_scaleFPOT, self.vector_vector_BNB_CCpi0_from_overlaynumu_FC_eventinfo, 5, "numu2numu")
        maybe_add(self.flag_BNB_CCpi0_from_overlaynumu, self.vector_BNB_CCpi0_from_overlaynumu_scaleFPOT, self.vector_vector_BNB_CCpi0_from_overlaynumu_PC_eventinfo, 6, "numu2numu")

        maybe_add(self.flag_BNB_NCpi0_from_overlaynumu, self.vector_BNB_NCpi0_from_overlaynumu_scaleFPOT, self.vector_vector_BNB_NCpi0_from_overlaynumu_eventinfo, 7, "numu2numu")

        maybe_add(self.flag_BNB_nueCC_from_overlaynueNC, self.vector_BNB_nueCC_from_overlaynueNC_scaleFPOT, self.vector_vector_BNB_nueCC_from_overlaynueNC_FC_eventinfo, 1, "nueNC")
        maybe_add(self.flag_BNB_nueCC_from_overlaynueNC, self.vector_BNB_nueCC_from_overlaynueNC_scaleFPOT, self.vector_vector_BNB_nueCC_from_overlaynueNC_PC_eventinfo, 2, "nueNC")

        maybe_add(self.flag_BNB_nueCC_from_overlaynumuNC, self.vector_BNB_nueCC_from_overlaynumuNC_scaleFPOT, self.vector_vector_BNB_nueCC_from_overlaynumuNC_FC_eventinfo, 1, "numuNC")
        maybe_add(self.flag_BNB_nueCC_from_overlaynumuNC, self.vector_BNB_nueCC_from_overlaynumuNC_scaleFPOT, self.vector_vector_BNB_nueCC_from_overlaynumuNC_PC_eventinfo, 2, "numuNC")

        maybe_add(self.flag_BNB_numuCC_from_overlaynueNC, self.vector_BNB_numuCC_from_overlaynueNC_scaleFPOT, self.vector_vector_BNB_numuCC_from_overlaynueNC_FC_eventinfo, 3, "nueNC")
        maybe_add(self.flag_BNB_numuCC_from_overlaynueNC, self.vector_BNB_numuCC_from_overlaynueNC_scaleFPOT, self.vector_vector_BNB_numuCC_from_overlaynueNC_PC_eventinfo, 4, "nueNC")

        maybe_add(self.flag_BNB_numuCC_from_overlaynumuNC, self.vector_BNB_numuCC_from_overlaynumuNC_scaleFPOT, self.vector_vector_BNB_numuCC_from_overlaynumuNC_FC_eventinfo, 3, "numuNC")
        maybe_add(self.flag_BNB_numuCC_from_overlaynumuNC, self.vector_BNB_numuCC_from_overlaynumuNC_scaleFPOT, self.vector_vector_BNB_numuCC_from_overlaynumuNC_PC_eventinfo, 4, "numuNC")

        maybe_add(self.flag_BNB_CCpi0_from_overlaynueNC, self.vector_BNB_CCpi0_from_overlaynueNC_scaleFPOT, self.vector_vector_BNB_CCpi0_from_overlaynueNC_FC_eventinfo, 5, "nueNC")
        maybe_add(self.flag_BNB_CCpi0_from_overlaynueNC, self.vector_BNB_CCpi0_from_overlaynueNC_scaleFPOT, self.vector_vector_BNB_CCpi0_from_overlaynueNC_PC_eventinfo, 6, "nueNC")

        maybe_add(self.flag_BNB_CCpi0_from_overlaynumuNC, self.vector_BNB_CCpi0_from_overlaynumuNC_scaleFPOT, self.vector_vector_BNB_CCpi0_from_overlaynumuNC_FC_eventinfo, 5, "numuNC")
        maybe_add(self.flag_BNB_CCpi0_from_overlaynumuNC, self.vector_BNB_CCpi0_from_overlaynumuNC_scaleFPOT, self.vector_vector_BNB_CCpi0_from_overlaynumuNC_PC_eventinfo, 6, "numuNC")

        maybe_add(self.flag_BNB_NCpi0_from_overlaynueNC, self.vector_BNB_NCpi0_from_overlaynueNC_scaleFPOT, self.vector_vector_BNB_NCpi0_from_overlaynueNC_eventinfo, 7, "nueNC")
        maybe_add(self.flag_BNB_NCpi0_from_overlaynumuNC, self.vector_BNB_NCpi0_from_overlaynumuNC_scaleFPOT, self.vector_vector_BNB_NCpi0_from_overlaynumuNC_eventinfo, 7, "numuNC")

        maybe_add(self.flag_BNB_nueCC_from_appnue, self.vector_BNB_nueCC_from_appnue_scaleFPOT, self.vector_vector_BNB_nueCC_from_appnue_FC_eventinfo, 15, "numu2nue")
        maybe_add(self.flag_BNB_nueCC_from_appnue, self.vector_BNB_nueCC_from_appnue_scaleFPOT, self.vector_vector_BNB_nueCC_from_appnue_PC_eventinfo, 16, "numu2nue")

        maybe_add(self.flag_BNB_numuCC_from_appnue, self.vector_BNB_numuCC_from_appnue_scaleFPOT, self.vector_vector_BNB_numuCC_from_appnue_FC_eventinfo, 17, "numu2nue")
        maybe_add(self.flag_BNB_numuCC_from_appnue, self.vector_BNB_numuCC_from_appnue_scaleFPOT, self.vector_vector_BNB_numuCC_from_appnue_PC_eventinfo, 18, "numu2nue")

        maybe_add(self.flag_BNB_CCpi0_from_appnue, self.vector_BNB_CCpi0_from_appnue_scaleFPOT, self.vector_vector_BNB_CCpi0_from_appnue_FC_eventinfo, 19, "numu2nue")
        maybe_add(self.flag_BNB_CCpi0_from_appnue, self.vector_BNB_CCpi0_from_appnue_scaleFPOT, self.vector_vector_BNB_CCpi0_from_appnue_PC_eventinfo, 20, "numu2nue")

        maybe_add(self.flag_BNB_NCpi0_from_appnue, self.vector_BNB_NCpi0_from_appnue_scaleFPOT, self.vector_vector_BNB_NCpi0_from_appnue_eventinfo, 21, "numu2nue")

    def set_oscillation_base(self, default_eventlist_dir):
        print("\n ---> set_oscillation_base")
        self.matrix_oscillation_base_oldworld_pred = self.matrix_default_oldworld_pred

        str_dirbase = default_eventlist_dir

        def process_runs(runs, scale_fpot, fc_tree, fc_vec, pc_tree=None, pc_vec=None):
            for strfile_mcPOT, strfile_dataPOT, strfile_mc_e2e in runs:
                strfile_mcPOT = str_dirbase + strfile_mcPOT
                strfile_dataPOT = str_dirbase + strfile_dataPOT
                strfile_mc_e2e = str_dirbase + strfile_mc_e2e

                self.set_oscillation_base_subfunc(
                    strfile_mcPOT,
                    strfile_dataPOT,
                    scale_fpot,
                    strfile_mc_e2e,
                    fc_tree,
                    fc_vec
                )

                if pc_tree and pc_vec:
                    self.set_oscillation_base_subfunc(
                        strfile_mcPOT,
                        strfile_dataPOT,
                        None,
                        strfile_mc_e2e,
                        pc_tree,
                        pc_vec
                    )

        # === flag_NuMI_nueCC_from_intnue ===
        if self.flag_NuMI_nueCC_from_intnue:
            print("\n      ---> flag_NuMI_nueCC_from_intnue")
            runs = [
                ("checkout_prodgenie_run1_fhc_intrinsic_nue_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_intrinsic.root"),
                ("checkout_prodgenie_run1_rhc_intrinsic_nue_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_intrinsic.root"),
                ("checkout_prodgenie_run2_fhc_intrinsic_nue_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_intrinsic.root"),
                ("checkout_prodgenie_run2_rhc_intrinsic_nue_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_intrinsic.root"),
                ("checkout_prodgenie_run3_rhc_intrinsic_nue_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_intrinsic.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_nueCC_from_intnue_scaleFPOT,
                         "tree_nueCC_from_intnue_FC",
                         self.vector_vector_NuMI_nueCC_from_intnue_FC_eventinfo,
                         "tree_nueCC_from_intnue_PC",
                         self.vector_vector_NuMI_nueCC_from_intnue_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_nueCC_from_intnue_scaleFPOT,
                                            self.vector_vector_NuMI_nueCC_from_intnue_FC_eventinfo,
                                            22, "nue2nue")
            self.set_oscillation_base_minus(self.vector_NuMI_nueCC_from_intnue_scaleFPOT,
                                            self.vector_vector_NuMI_nueCC_from_intnue_PC_eventinfo,
                                            23, "nue2nue")

        # === flag_NuMI_nueCC_from_overlaynumu ===
        if self.flag_NuMI_nueCC_from_overlaynumu:
            print("\n      ---> flag_NuMI_nueCC_from_overlaynumu")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_nueCC_from_overlaynumu_scaleFPOT,
                         "tree_nueCC_from_overlaynumu_FC",
                         self.vector_vector_NuMI_nueCC_from_overlaynumu_FC_eventinfo,
                         "tree_nueCC_from_overlaynumu_PC",
                         self.vector_vector_NuMI_nueCC_from_overlaynumu_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_nueCC_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_NuMI_nueCC_from_overlaynumu_FC_eventinfo,
                                            22, "numu2numu")
            self.set_oscillation_base_minus(self.vector_NuMI_nueCC_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_NuMI_nueCC_from_overlaynumu_PC_eventinfo,
                                            23, "numu2numu")
        # === flag_NuMI_nueCC_from_overlaynueNC ===
        if self.flag_NuMI_nueCC_from_overlaynueNC:
            print("\n      ---> flag_NuMI_nueCC_from_overlaynueNC")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT,
                         "tree_nueCC_from_overlaynueNC_FC",
                         self.vector_vector_NuMI_nueCC_from_overlaynueNC_FC_eventinfo,
                         "tree_nueCC_from_overlaynueNC_PC",
                         self.vector_vector_NuMI_nueCC_from_overlaynueNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_NuMI_nueCC_from_overlaynueNC_FC_eventinfo,
                                            22, "nueNC")
            self.set_oscillation_base_minus(self.vector_NuMI_nueCC_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_NuMI_nueCC_from_overlaynueNC_PC_eventinfo,
                                            23, "nueNC")
        # === flag_NuMI_nueCC_from_overlaynumuNC ===
        if self.flag_NuMI_nueCC_from_overlaynumuNC:
            print("\n      ---> flag_NuMI_nueCC_from_overlaynumuNC")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT,
                         "tree_nueCC_from_overlaynumuNC_FC",
                         self.vector_vector_NuMI_nueCC_from_overlaynumuNC_FC_eventinfo,
                         "tree_nueCC_from_overlaynumuNC_PC",
                         self.vector_vector_NuMI_nueCC_from_overlaynumuNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_NuMI_nueCC_from_overlaynumuNC_FC_eventinfo,
                                            22, "numuNC")
            self.set_oscillation_base_minus(self.vector_NuMI_nueCC_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_NuMI_nueCC_from_overlaynumuNC_PC_eventinfo,
                                            23, "numuNC")

        # === flag_NuMI_numuCC_from_overlaynumu ===
        if self.flag_NuMI_numuCC_from_overlaynumu:
            print("\n      ---> flag_NuMI_numuCC_from_overlaynumu")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_numuCC_from_overlaynumu_scaleFPOT,
                         "tree_numuCC_from_overlaynumu_FC",
                         self.vector_vector_NuMI_numuCC_from_overlaynumu_FC_eventinfo,
                         "tree_numuCC_from_overlaynumu_PC",
                         self.vector_vector_NuMI_numuCC_from_overlaynumu_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_numuCC_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_NuMI_numuCC_from_overlaynumu_FC_eventinfo,
                                            24, "numu2numu")
            self.set_oscillation_base_minus(self.vector_NuMI_numuCC_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_NuMI_numuCC_from_overlaynumu_PC_eventinfo,
                                            25, "numu2numu")


        # === flag_NuMI_numuCC_from_overlaynueNC ===
        if self.flag_NuMI_numuCC_from_overlaynueNC:
            print("\n      ---> flag_NuMI_numuCC_from_overlaynueNC")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT,
                         "tree_numuCC_from_overlaynueNC_FC",
                         self.vector_vector_NuMI_numuCC_from_overlaynueNC_FC_eventinfo,
                         "tree_numuCC_from_overlaynueNC_PC",
                         self.vector_vector_NuMI_numuCC_from_overlaynueNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_NuMI_numuCC_from_overlaynueNC_FC_eventinfo,
                                            24, "nueNC")
            self.set_oscillation_base_minus(self.vector_NuMI_numuCC_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_NuMI_numuCC_from_overlaynueNC_PC_eventinfo,
                                            25, "nueNC")

        # === flag_NuMI_numuCC_from_overlaynumuNC ===
        if self.flag_NuMI_numuCC_from_overlaynumuNC:
            print("\n      ---> flag_NuMI_numuCC_from_overlaynumuNC")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT,
                         "tree_numuCC_from_overlaynumuNC_FC",
                         self.vector_vector_NuMI_numuCC_from_overlaynumuNC_FC_eventinfo,
                         "tree_numuCC_from_overlaynumuNC_PC",
                         self.vector_vector_NuMI_numuCC_from_overlaynumuNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_NuMI_numuCC_from_overlaynumuNC_FC_eventinfo,
                                            24, "numuNC")
            self.set_oscillation_base_minus(self.vector_NuMI_numuCC_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_NuMI_numuCC_from_overlaynumuNC_PC_eventinfo,
                                            25, "numuNC")

        # === flag_NuMI_CCpi0_from_overlaynumu ===
        if self.flag_NuMI_CCpi0_from_overlaynumu:
            print("\n      ---> flag_NuMI_CCpi0_from_overlaynumu")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT,
                         "tree_CCpi0_from_overlaynumu_FC",
                         self.vector_vector_NuMI_CCpi0_from_overlaynumu_FC_eventinfo,
                         "tree_CCpi0_from_overlaynumu_PC",
                         self.vector_vector_NuMI_CCpi0_from_overlaynumu_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_NuMI_CCpi0_from_overlaynumu_FC_eventinfo,
                                            26, "numu2numu")
            self.set_oscillation_base_minus(self.vector_NuMI_CCpi0_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_NuMI_CCpi0_from_overlaynumu_PC_eventinfo,
                                            27, "numu2numu")

        # === flag_NuMI_CCpi0_from_overlaynueNC ===
        if self.flag_NuMI_CCpi0_from_overlaynueNC:
            print("\n      ---> flag_NuMI_CCpi0_from_overlaynueNC")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT,
                         "tree_CCpi0_from_overlaynueNC_FC",
                         self.vector_vector_NuMI_CCpi0_from_overlaynueNC_FC_eventinfo,
                         "tree_CCpi0_from_overlaynueNC_PC",
                         self.vector_vector_NuMI_CCpi0_from_overlaynueNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_NuMI_CCpi0_from_overlaynueNC_FC_eventinfo,
                                            26, "nueNC")
            self.set_oscillation_base_minus(self.vector_NuMI_CCpi0_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_NuMI_CCpi0_from_overlaynueNC_PC_eventinfo,
                                            27, "nueNC")

        # === flag_NuMI_CCpi0_from_overlaynumuNC ===
        if self.flag_NuMI_CCpi0_from_overlaynumuNC:
            print("\n      ---> flag_NuMI_CCpi0_from_overlaynumuNC")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT,
                         "tree_CCpi0_from_overlaynumuNC_FC",
                         self.vector_vector_NuMI_CCpi0_from_overlaynumuNC_FC_eventinfo,
                         "tree_CCpi0_from_overlaynumuNC_PC",
                         self.vector_vector_NuMI_CCpi0_from_overlaynumuNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_NuMI_CCpi0_from_overlaynumuNC_FC_eventinfo,
                                            26, "numuNC")
            self.set_oscillation_base_minus(self.vector_NuMI_CCpi0_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_NuMI_CCpi0_from_overlaynumuNC_PC_eventinfo,
                                            27, "numuNC")

        # === flag_NuMI_NCpi0_from_overlaynumu ===
        if self.flag_NuMI_NCpi0_from_overlaynumu:
            print("\n      ---> flag_NuMI_NCpi0_from_overlaynumu")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_NCpi0_from_overlaynumu_scaleFPOT,
                         "tree_NCpi0_from_overlaynumu",
                         self.vector_vector_NuMI_NCpi0_from_overlaynumu_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_NCpi0_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_NuMI_NCpi0_from_overlaynumu_eventinfo,
                                            28, "numu2numu")

        # === flag_NuMI_NCpi0_from_overlaynueNC ===
        if self.flag_NuMI_NCpi0_from_overlaynueNC:
            print("\n      ---> flag_NuMI_NCpi0_from_overlaynueNC")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_NCpi0_from_overlaynueNC_scaleFPOT,
                         "tree_NCpi0_from_overlaynueNC",
                         self.vector_vector_NuMI_NCpi0_from_overlaynueNC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_NCpi0_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_NuMI_NCpi0_from_overlaynueNC_eventinfo,
                                            28, "nueNC")
        # === flag_NuMI_NCpi0_from_overlaynumuNC ===
        if self.flag_NuMI_NCpi0_from_overlaynumuNC:
            print("\n      ---> flag_NuMI_NCpi0_from_overlaynumuNC")
            runs = [
                ("checkout_prodgenie_run1_fhc_nu_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run1_rhc_nu_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_fhc_nu_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_nu_overlay.root"),
                ("checkout_prodgenie_run2_rhc_nu_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_nu_overlay.root"),
                ("checkout_prodgenie_run3_rhc_nu_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_NCpi0_from_overlaynumuNC_scaleFPOT,
                         "tree_NCpi0_from_overlaynumuNC",
                         self.vector_vector_NuMI_NCpi0_from_overlaynumuNC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_NCpi0_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_NuMI_NCpi0_from_overlaynumuNC_eventinfo,
                                            28, "numuNC")
        # === flag_NuMI_nueCC_from_appnue ===
        if self.flag_NuMI_nueCC_from_appnue or True:
            print("\n      ---> flag_NuMI_nueCC_from_appnue")
            runs = [
                ("checkout_prodgenie_run1_fhc_fullosc_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_appnue.root"),
                ("checkout_prodgenie_run1_rhc_fullosc_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_appnue.root"),
                ("checkout_prodgenie_run2_fhc_fullosc_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_appnue.root"),
                ("checkout_prodgenie_run2_rhc_fullosc_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_appnue.root"),
                ("checkout_prodgenie_run3_rhc_fullosc_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_appnue.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_nueCC_from_appnue_scaleFPOT,
                         "tree_nueCC_from_appnue_FC",
                         self.vector_vector_NuMI_nueCC_from_appnue_FC_eventinfo,
                         "tree_nueCC_from_appnue_PC",
                         self.vector_vector_NuMI_nueCC_from_appnue_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_nueCC_from_appnue_scaleFPOT,
                                            self.vector_vector_NuMI_nueCC_from_appnue_FC_eventinfo,
                                            36, "numu2nue")
            self.set_oscillation_base_minus(self.vector_NuMI_nueCC_from_appnue_scaleFPOT,
                                            self.vector_vector_NuMI_nueCC_from_appnue_PC_eventinfo,
                                            37, "numu2nue")
        # === flag_NuMI_numuCC_from_appnue ===
        if self.flag_NuMI_numuCC_from_appnue or True:
            print("\n      ---> flag_NuMI_numuCC_from_appnue")
            runs = [
                ("checkout_prodgenie_run1_fhc_fullosc_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_appnue.root"),
                ("checkout_prodgenie_run1_rhc_fullosc_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_appnue.root"),
                ("checkout_prodgenie_run2_fhc_fullosc_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_appnue.root"),
                ("checkout_prodgenie_run2_rhc_fullosc_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_appnue.root"),
                ("checkout_prodgenie_run3_rhc_fullosc_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_appnue.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_numuCC_from_appnue_scaleFPOT,
                         "tree_numuCC_from_appnue_FC",
                         self.vector_vector_NuMI_numuCC_from_appnue_FC_eventinfo,
                         "tree_numuCC_from_appnue_PC",
                         self.vector_vector_NuMI_numuCC_from_appnue_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_numuCC_from_appnue_scaleFPOT,
                                            self.vector_vector_NuMI_numuCC_from_appnue_FC_eventinfo,
                                            38, "numu2nue")
            self.set_oscillation_base_minus(self.vector_NuMI_numuCC_from_appnue_scaleFPOT,
                                            self.vector_vector_NuMI_numuCC_from_appnue_PC_eventinfo,
                                            39, "numu2nue")
        # === flag_NuMI_CCpi0_from_appnue ===
        if self.flag_NuMI_CCpi0_from_appnue or True:
            print("\n      ---> flag_NuMI_CCpi0_from_appnue")
            runs = [
                ("checkout_prodgenie_run1_fhc_fullosc_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_appnue.root"),
                ("checkout_prodgenie_run1_rhc_fullosc_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_appnue.root"),
                ("checkout_prodgenie_run2_fhc_fullosc_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_appnue.root"),
                ("checkout_prodgenie_run2_rhc_fullosc_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_appnue.root"),
                ("checkout_prodgenie_run3_rhc_fullosc_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_appnue.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_CCpi0_from_appnue_scaleFPOT,
                         "tree_CCpi0_from_appnue_FC",
                         self.vector_vector_NuMI_CCpi0_from_appnue_FC_eventinfo,
                         "tree_CCpi0_from_appnue_PC",
                         self.vector_vector_NuMI_CCpi0_from_appnue_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_CCpi0_from_appnue_scaleFPOT,
                                            self.vector_vector_NuMI_CCpi0_from_appnue_FC_eventinfo,
                                            40, "numu2nue")
            self.set_oscillation_base_minus(self.vector_NuMI_CCpi0_from_appnue_scaleFPOT,
                                            self.vector_vector_NuMI_CCpi0_from_appnue_PC_eventinfo,
                                            41, "numu2nue")
        # === flag_NuMI_NCpi0_from_appnue ===
        if self.flag_NuMI_NCpi0_from_appnue:
            print("\n      ---> flag_NuMI_NCpi0_from_appnue")
            runs = [
                ("checkout_prodgenie_run1_fhc_fullosc_overlay.root", "run1_fhc_data_numi.root", "roofile_obj_NuMI_run1_FHC_appnue.root"),
                ("checkout_prodgenie_run1_rhc_fullosc_overlay.root", "run1_rhc_data_numi.root", "roofile_obj_NuMI_run1_RHC_appnue.root"),
                ("checkout_prodgenie_run2_fhc_fullosc_overlay.root", "run2_fhc_data_numi.root", "roofile_obj_NuMI_run2_FHC_appnue.root"),
                ("checkout_prodgenie_run2_rhc_fullosc_overlay.root", "run2_rhc_data_numi.root", "roofile_obj_NuMI_run2_RHC_appnue.root"),
                ("checkout_prodgenie_run3_rhc_fullosc_overlay.root", "run3_rhc_data_numi.root", "roofile_obj_NuMI_run3_RHC_appnue.root")
            ]
            process_runs(runs,
                         self.vector_NuMI_NCpi0_from_appnue_scaleFPOT,
                         "tree_NCpi0_from_appnue",
                         self.vector_vector_NuMI_NCpi0_from_appnue_eventinfo)

            self.set_oscillation_base_minus(self.vector_NuMI_NCpi0_from_appnue_scaleFPOT,
                                            self.vector_vector_NuMI_NCpi0_from_appnue_eventinfo,
                                            42, "numu2nue")
        
        

        # === flag_BNB_nueCC_from_intnue ===
        if self.flag_BNB_nueCC_from_intnue:
            print("\n      ---> flag_BNB_nueCC_from_intnue")
            runs = [
                ("checkout_prodgenie_bnb_intrinsic_nue_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_intrinsic.root"),
                ("checkout_prodgenie_bnb_intrinsic_nue_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_intrinsic.root"),
                ("checkout_prodgenie_bnb_intrinsic_nue_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_intrinsic.root")
            ]
            process_runs(runs,
                         self.vector_BNB_nueCC_from_intnue_scaleFPOT,
                         "tree_nueCC_from_intnue_FC",
                         self.vector_vector_BNB_nueCC_from_intnue_FC_eventinfo,
                         "tree_nueCC_from_intnue_PC",
                         self.vector_vector_BNB_nueCC_from_intnue_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_nueCC_from_intnue_scaleFPOT,
                                            self.vector_vector_BNB_nueCC_from_intnue_FC_eventinfo,
                                            1, "nue2nue")
            self.set_oscillation_base_minus(self.vector_BNB_nueCC_from_intnue_scaleFPOT,
                                            self.vector_vector_BNB_nueCC_from_intnue_FC_eventinfo,
                                            2, "nue2nue")

        # === flag_BNB_nueCC_from_overlaynumu ===
        if self.flag_BNB_nueCC_from_overlaynumu:
            print("\n      ---> flag_BNB_nueCC_from_overlaynumu")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_nueCC_from_overlaynumu_scaleFPOT,
                         "tree_nueCC_from_overlaynumu_FC",
                         self.vector_vector_BNB_nueCC_from_overlaynumu_FC_eventinfo,
                         "tree_nueCC_from_overlaynumu_PC",
                         self.vector_vector_BNB_nueCC_from_overlaynumu_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_nueCC_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_BNB_nueCC_from_overlaynumu_FC_eventinfo,
                                            1, "numu2numu")
            self.set_oscillation_base_minus(self.vector_BNB_nueCC_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_BNB_nueCC_from_overlaynumu_PC_eventinfo,
                                            2, "numu2numu")
      
      
        # === flag_BNB_nueCC_from_overlaynueNC ===
        if self.flag_BNB_nueCC_from_overlaynueNC:
            print("\n      ---> flag_BNB_nueCC_from_overlaynueNC")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_nueCC_from_overlaynueNC_scaleFPOT,
                         "tree_nueCC_from_overlaynueNC_FC",
                         self.vector_vector_BNB_nueCC_from_overlaynueNC_FC_eventinfo,
                         "tree_nueCC_from_overlaynueNC_PC",
                         self.vector_vector_BNB_nueCC_from_overlaynueNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_nueCC_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_BNB_nueCC_from_overlaynueNC_FC_eventinfo,
                                            1, "nueNC")
            self.set_oscillation_base_minus(self.vector_BNB_nueCC_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_BNB_nueCC_from_overlaynueNC_PC_eventinfo,
                                            2, "nueNC")
                                            
        # === flag_BNB_nueCC_from_overlaynumuNC ===
        if self.flag_BNB_nueCC_from_overlaynumuNC:
            print("\n      ---> flag_BNB_nueCC_from_overlaynumuNC")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_nueCC_from_overlaynumuNC_scaleFPOT,
                         "tree_nueCC_from_overlaynumuNC_FC",
                         self.vector_vector_BNB_nueCC_from_overlaynumuNC_FC_eventinfo,
                         "tree_nueCC_from_overlaynumuNC_PC",
                         self.vector_vector_BNB_nueCC_from_overlaynumuNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_nueCC_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_BNB_nueCC_from_overlaynumuNC_FC_eventinfo,
                                            1, "numuNC")
            self.set_oscillation_base_minus(self.vector_BNB_nueCC_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_BNB_nueCC_from_overlaynumuNC_PC_eventinfo,
                                            2, "numuNC")

        # === flag_BNB_numuCC_from_overlaynumu ===
        if self.flag_BNB_numuCC_from_overlaynumu:
            print("\n      ---> flag_BNB_numuCC_from_overlaynumu")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_numuCC_from_overlaynumu_scaleFPOT,
                         "tree_numuCC_from_overlaynumu_FC",
                         self.vector_vector_BNB_numuCC_from_overlaynumu_FC_eventinfo,
                         "tree_numuCC_from_overlaynumu_PC",
                         self.vector_vector_BNB_numuCC_from_overlaynumu_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_numuCC_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_BNB_numuCC_from_overlaynumu_FC_eventinfo,
                                            3, "numu2numu")
            self.set_oscillation_base_minus(self.vector_BNB_numuCC_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_BNB_numuCC_from_overlaynumu_PC_eventinfo,
                                            4, "numu2numu")


        # === flag_BNB_numuCC_from_overlaynueNC ===
        if self.flag_BNB_numuCC_from_overlaynueNC:
            print("\n      ---> flag_BNB_numuCC_from_overlaynueNC")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_numuCC_from_overlaynueNC_scaleFPOT,
                         "tree_numuCC_from_overlaynueNC_FC",
                         self.vector_vector_BNB_numuCC_from_overlaynueNC_FC_eventinfo,
                         "tree_numuCC_from_overlaynueNC_PC",
                         self.vector_vector_BNB_numuCC_from_overlaynueNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_numuCC_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_BNB_numuCC_from_overlaynueNC_FC_eventinfo,
                                            3, "nueNC")
            self.set_oscillation_base_minus(self.vector_BNB_numuCC_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_BNB_numuCC_from_overlaynueNC_PC_eventinfo,
                                            4, "nueNC")

        # === flag_BNB_numuCC_from_overlaynumuNC ===
        if self.flag_BNB_numuCC_from_overlaynumuNC:
            print("\n      ---> flag_BNB_numuCC_from_overlaynumuNC")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_numuCC_from_overlaynumuNC_scaleFPOT,
                         "tree_numuCC_from_overlaynumuNC_FC",
                         self.vector_vector_BNB_numuCC_from_overlaynumuNC_FC_eventinfo,
                         "tree_numuCC_from_overlaynumuNC_PC",
                         self.vector_vector_BNB_numuCC_from_overlaynumuNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_numuCC_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_BNB_numuCC_from_overlaynumuNC_FC_eventinfo,
                                            3, "numuNC")
            self.set_oscillation_base_minus(self.vector_BNB_numuCC_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_BNB_numuCC_from_overlaynumuNC_PC_eventinfo,
                                            4, "numuNC")

        # === flag_BNB_CCpi0_from_overlaynumu ===
        if self.flag_BNB_CCpi0_from_overlaynumu:
            print("\n      ---> flag_BNB_CCpi0_from_overlaynumu")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_CCpi0_from_overlaynumu_scaleFPOT,
                         "tree_CCpi0_from_overlaynumu_FC",
                         self.vector_vector_BNB_CCpi0_from_overlaynumu_FC_eventinfo,
                         "tree_CCpi0_from_overlaynumu_PC",
                         self.vector_vector_BNB_CCpi0_from_overlaynumu_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_CCpi0_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_BNB_CCpi0_from_overlaynumu_FC_eventinfo,
                                            5, "numu2numu")
            self.set_oscillation_base_minus(self.vector_BNB_CCpi0_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_BNB_CCpi0_from_overlaynumu_PC_eventinfo,
                                            6, "numu2numu")

        # === flag_BNB_CCpi0_from_overlaynueNC ===
        if self.flag_BNB_CCpi0_from_overlaynueNC:
            print("\n      ---> flag_BNB_CCpi0_from_overlaynueNC")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_CCpi0_from_overlaynueNC_scaleFPOT,
                         "tree_CCpi0_from_overlaynueNC_FC",
                         self.vector_vector_BNB_CCpi0_from_overlaynueNC_FC_eventinfo,
                         "tree_CCpi0_from_overlaynueNC_PC",
                         self.vector_vector_BNB_CCpi0_from_overlaynueNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_CCpi0_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_BNB_CCpi0_from_overlaynueNC_FC_eventinfo,
                                            5, "nueNC")
            self.set_oscillation_base_minus(self.vector_BNB_CCpi0_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_BNB_CCpi0_from_overlaynueNC_PC_eventinfo,
                                            6, "nueNC")

        # === flag_BNB_CCpi0_from_overlaynumuNC ===
        if self.flag_BNB_CCpi0_from_overlaynumuNC:
            print("\n      ---> flag_BNB_CCpi0_from_overlaynumuNC")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_CCpi0_from_overlaynumuNC_scaleFPOT,
                         "tree_CCpi0_from_overlaynumuNC_FC",
                         self.vector_vector_BNB_CCpi0_from_overlaynumuNC_FC_eventinfo,
                         "tree_CCpi0_from_overlaynumuNC_PC",
                         self.vector_vector_BNB_CCpi0_from_overlaynumuNC_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_CCpi0_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_BNB_CCpi0_from_overlaynumuNC_FC_eventinfo,
                                            5, "numuNC")
            self.set_oscillation_base_minus(self.vector_BNB_CCpi0_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_BNB_CCpi0_from_overlaynumuNC_PC_eventinfo,
                                            6, "numuNC")

        # === flag_BNB_NCpi0_from_overlaynumu ===
        if self.flag_BNB_NCpi0_from_overlaynumu:
            print("\n      ---> flag_BNB_NCpi0_from_overlaynumu")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_NCpi0_from_overlaynumu_scaleFPOT,
                         "tree_NCpi0_from_overlaynumu",
                         self.vector_vector_BNB_NCpi0_from_overlaynumu_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_NCpi0_from_overlaynumu_scaleFPOT,
                                            self.vector_vector_BNB_NCpi0_from_overlaynumu_eventinfo,
                                            7, "numu2numu")

        # === flag_BNB_NCpi0_from_overlaynueNC ===
        if self.flag_BNB_NCpi0_from_overlaynueNC:
            print("\n      ---> flag_BNB_NCpi0_from_overlaynueNC")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_NCpi0_from_overlaynueNC_scaleFPOT,
                         "tree_NCpi0_from_overlaynueNC",
                         self.vector_vector_BNB_NCpi0_from_overlaynueNC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_NCpi0_from_overlaynueNC_scaleFPOT,
                                            self.vector_vector_BNB_NCpi0_from_overlaynueNC_eventinfo,
                                            7, "nueNC")
                                            
        # === flag_BNB_NCpi0_from_overlaynumuNC ===
        if self.flag_BNB_NCpi0_from_overlaynumuNC:
            print("\n      ---> flag_BNB_NCpi0_from_overlaynumuNC")
            runs = [
                ("checkout_prodgenie_bnb_nu_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_nu_overlay.root"),
                ("checkout_prodgenie_bnb_nu_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_nu_overlay.root")
            ]
            process_runs(runs,
                         self.vector_BNB_NCpi0_from_overlaynumuNC_scaleFPOT,
                         "tree_NCpi0_from_overlaynumuNC",
                         self.vector_vector_BNB_NCpi0_from_overlaynumuNC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_NCpi0_from_overlaynumuNC_scaleFPOT,
                                            self.vector_vector_BNB_NCpi0_from_overlaynumuNC_eventinfo,
                                            7, "numuNC")
                                            
        # === flag_BNB_nueCC_from_appnue ===
        if self.flag_BNB_nueCC_from_appnue or True:
            print("\n      ---> flag_BNB_nueCC_from_appnue")
            runs = [
                ("checkout_prodgenie_bnb_numu2nue_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_appnue.root"),
                ("checkout_prodgenie_bnb_numu2nue_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_appnue.root"),
                ("checkout_prodgenie_bnb_numu2nue_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_appnue.root")
            ]
            process_runs(runs,
                         self.vector_BNB_nueCC_from_appnue_scaleFPOT,
                         "tree_nueCC_from_appnue_FC",
                         self.vector_vector_BNB_nueCC_from_appnue_FC_eventinfo,
                         "tree_nueCC_from_appnue_PC",
                         self.vector_vector_BNB_nueCC_from_appnue_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_nueCC_from_appnue_scaleFPOT,
                                            self.vector_vector_BNB_nueCC_from_appnue_FC_eventinfo,
                                            15, "numu2nue")
            self.set_oscillation_base_minus(self.vector_BNB_nueCC_from_appnue_scaleFPOT,
                                            self.vector_vector_BNB_nueCC_from_appnue_PC_eventinfo,
                                            16, "numu2nue")
                                            
        # === flag_BNB_numuCC_from_appnue ===
        if self.flag_BNB_numuCC_from_appnue or True:
            print("\n      ---> flag_BNB_numuCC_from_appnue")
            runs = [
                ("checkout_prodgenie_bnb_numu2nue_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_appnue.root"),
                ("checkout_prodgenie_bnb_numu2nue_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_appnue.root"),
                ("checkout_prodgenie_bnb_numu2nue_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_appnue.root")
            ]
            process_runs(runs,
                         self.vector_BNB_numuCC_from_appnue_scaleFPOT,
                         "tree_numuCC_from_appnue_FC",
                         self.vector_vector_BNB_numuCC_from_appnue_FC_eventinfo,
                         "tree_numuCC_from_appnue_PC",
                         self.vector_vector_BNB_numuCC_from_appnue_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_numuCC_from_appnue_scaleFPOT,
                                            self.vector_vector_BNB_numuCC_from_appnue_FC_eventinfo,
                                            17, "numu2nue")
            self.set_oscillation_base_minus(self.vector_BNB_numuCC_from_appnue_scaleFPOT,
                                            self.vector_vector_BNB_numuCC_from_appnue_PC_eventinfo,
                                            18, "numu2nue")
                                            
        # === flag_BNB_CCpi0_from_appnue ===
        if self.flag_BNB_CCpi0_from_appnue or True:
            print("\n      ---> flag_BNB_CCpi0_from_appnue")
            runs = [
                ("checkout_prodgenie_bnb_numu2nue_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_appnue.root"),
                ("checkout_prodgenie_bnb_numu2nue_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_appnue.root"),
                ("checkout_prodgenie_bnb_numu2nue_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_appnue.root")
            ]
            process_runs(runs,
                         self.vector_BNB_CCpi0_from_appnue_scaleFPOT,
                         "tree_CCpi0_from_appnue_FC",
                         self.vector_vector_BNB_CCpi0_from_appnue_FC_eventinfo,
                         "tree_CCpi0_from_appnue_PC",
                         self.vector_vector_BNB_CCpi0_from_appnue_PC_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_CCpi0_from_appnue_scaleFPOT,
                                            self.vector_vector_BNB_CCpi0_from_appnue_FC_eventinfo,
                                            19, "numu2nue")
            self.set_oscillation_base_minus(self.vector_BNB_CCpi0_from_appnue_scaleFPOT,
                                            self.vector_vector_BNB_CCpi0_from_appnue_PC_eventinfo,
                                            20, "numu2nue")
                                            
        # === flag_BNB_NCpi0_from_appnue ===
        if self.flag_BNB_NCpi0_from_appnue:
            print("\n      ---> flag_BNB_NCpi0_from_appnue")
            runs = [
                ("checkout_prodgenie_bnb_numu2nue_overlay_run1.root", "run1_data_bnb.root", "roofile_obj_BNB_run1_appnue.root"),
                ("checkout_prodgenie_bnb_numu2nue_overlay_run2.root", "run2_data_bnb.root", "roofile_obj_BNB_run2_appnue.root"),
                ("checkout_prodgenie_bnb_numu2nue_overlay_run3.root", "run3_data_bnb.root", "roofile_obj_BNB_run3_appnue.root")
            ]
            process_runs(runs,
                         self.vector_BNB_NCpi0_from_appnue_scaleFPOT,
                         "tree_NCpi0_from_appnue",
                         self.vector_vector_BNB_NCpi0_from_appnue_eventinfo)

            self.set_oscillation_base_minus(self.vector_BNB_NCpi0_from_appnue_scaleFPOT,
                                            self.vector_vector_BNB_NCpi0_from_appnue_eventinfo,
                                            21, "numu2nue")

        
    def Set_default_cv_cov(self, default_cv_file, default_dirtadd_file, default_mcstat_file,
                           default_fluxXs_dir, default_detector_dir, device='cpu'):

        print("\n ---> Set_default_cv_cov\n")
        print(f"      ---> default_cv_file       {default_cv_file}")
        print(f"      ---> default_dirtadd_file  {default_dirtadd_file}")
        print(f"      ---> default_mcstat_file   {default_mcstat_file}")
        print(f"      ---> default_fluxXs_dir    {default_fluxXs_dir}")
        print(f"      ---> default_detector_dir  {default_detector_dir}")

        # --- Load transformation matrix
        with uproot.open(default_cv_file) as file:
            print("\n      ---> matrix_transform")
            mat = file['mat_collapse'].to_numpy()
            self.default_oldworld_rows, self.default_newworld_rows = mat.shape
            self.matrix_transform = torch.tensor(mat, dtype=torch.float32, device=device)

        print(f"      ---> default_oldworld_rows  {self.default_oldworld_rows}")
        print(f"      ---> default_newworld_rows  {self.default_newworld_rows}")

        # --- Initialize matrices
        zeros = lambda r, c=1: torch.zeros((r, c), dtype=torch.float32, device=device)
        self.matrix_default_newworld_meas = zeros(self.default_newworld_rows)
        self.matrix_default_oldworld_pred = zeros(self.default_oldworld_rows)
        self.matrix_oscillation_base_oldworld_pred = zeros(self.default_oldworld_rows)
        self.matrix_oscillation_oldworld_pred = zeros(self.default_oldworld_rows)

        self.matrix_default_oldworld_abs_syst_addi = torch.zeros(
            (self.default_oldworld_rows, self.default_oldworld_rows), dtype=torch.float32, device=device)

        self.matrix_default_oldworld_rel_syst_flux  = self.matrix_default_oldworld_abs_syst_addi.clone()
        self.matrix_default_oldworld_rel_syst_geant = self.matrix_default_oldworld_abs_syst_addi.clone()
        self.matrix_default_oldworld_rel_syst_Xs    = self.matrix_default_oldworld_abs_syst_addi.clone()
        self.matrix_default_oldworld_rel_syst_det   = self.matrix_default_oldworld_abs_syst_addi.clone()

        self.matrix_default_newworld_abs_syst_mcstat = torch.zeros(
            (self.default_newworld_rows, self.default_newworld_rows), dtype=torch.float32, device=device)

        self.matrix_eff_newworld_abs_syst_addi  = torch.zeros_like(self.matrix_default_newworld_abs_syst_mcstat)
        self.matrix_eff_newworld_abs_syst_mcstat = torch.zeros_like(self.matrix_default_newworld_abs_syst_mcstat)
        self.matrix_eff_newworld_abs_syst_flux   = torch.zeros_like(self.matrix_default_newworld_abs_syst_mcstat)
        self.matrix_eff_newworld_abs_syst_geant  = torch.zeros_like(self.matrix_default_newworld_abs_syst_mcstat)
        self.matrix_eff_newworld_abs_syst_Xs     = torch.zeros_like(self.matrix_default_newworld_abs_syst_mcstat)
        self.matrix_eff_newworld_abs_syst_det    = torch.zeros_like(self.matrix_default_newworld_abs_syst_mcstat)
        self.matrix_eff_newworld_abs_syst_total  = torch.zeros_like(self.matrix_default_newworld_abs_syst_mcstat)

        self.matrix_eff_newworld_meas  = zeros(self.default_newworld_rows)
        self.matrix_eff_newworld_pred  = zeros(self.default_newworld_rows)
        self.matrix_eff_newworld_noosc = zeros(self.default_newworld_rows)
        self.matrix_fitdata_newworld   = zeros(self.default_newworld_rows)

        # --- Load measurements
        print("\n      ---> measurement")
        self.vector_default_newworld_meas = []
        self.map_default_h1d_meas_bins = {}
        self.map_default_h1d_meas_xlow = {}
        self.map_default_h1d_meas_xhgh = {}

        with uproot.open(default_cv_file) as file:
            for idx in range(1, 10001):
                name = f"hdata_obsch_{idx}"
                if name not in file: break
                hist = file[name]
                bins = hist.numbins
                xlow, xhgh = hist.axis().edges[0], hist.axis().edges[-1]
                self.map_default_h1d_meas_bins[idx] = bins
                self.map_default_h1d_meas_xlow[idx] = xlow
                self.map_default_h1d_meas_xhgh[idx] = xhgh
                values = hist.values(flow=True)
                self.vector_default_newworld_meas.extend(values)
                print(f"      {idx:3d}, bins {bins + 1}, {hist.title}")

        if self.default_newworld_rows != len(self.vector_default_newworld_meas):
            raise ValueError("default_newworld_rows != vector_default_newworld_meas")

        self.matrix_default_newworld_meas[:, 0] = torch.tensor(
            self.vector_default_newworld_meas, dtype=torch.float32, device=device)

        # --- Load predictions
        print("\n      ---> prediction")
        self.vector_default_oldworld_pred = []
        self.map_default_h1d_pred_bins = {}
        self.map_default_h1d_pred_xlow = {}
        self.map_default_h1d_pred_xhgh = {}

        with uproot.open(default_cv_file) as file:
            for idx in range(1, 10001):
                name = f"histo_{idx}"
                if name not in file: break
                hist = file[name]
                bins = hist.numbins
                xlow, xhgh = hist.axis().edges[0], hist.axis().edges[-1]
                self.map_default_h1d_pred_bins[idx] = bins
                self.map_default_h1d_pred_xlow[idx] = xlow
                self.map_default_h1d_pred_xhgh[idx] = xhgh
                values = hist.values(flow=True)
                self.vector_default_oldworld_pred.extend(values)
                print(f"      {idx:3d}, bins {bins + 1}, {hist.title}")

        if self.default_oldworld_rows != len(self.vector_default_oldworld_pred):
            raise ValueError("default_oldworld_rows != vector_default_oldworld_pred")

        self.matrix_default_oldworld_pred[:, 0] = torch.tensor(
            self.vector_default_oldworld_pred, dtype=torch.float32, device=device)

        # --- Additional uncertainty: dirt
        if getattr(self, 'flag_syst_dirt', False):
            print("\n      ---> Dirt: additional uncertainty, Yes")
            with uproot.open(default_dirtadd_file) as file:
                mat = file['cov_mat_add'].to_numpy()
                self.matrix_default_oldworld_abs_syst_addi = torch.tensor(mat, dtype=torch.float32, device=device)
        else:
            print("\n      ---> Dirt: additional uncertainty, No")

        # --- MC stat
        if getattr(self, 'flag_syst_mcstat', False):
            print("\n      ---> MCstat, Yes")
            with open(default_mcstat_file) as f:
                lines = f.readlines()
                if len(lines) - 1 != self.default_newworld_rows:
                    raise ValueError("mcstat != default_newworld_rows")
                _, _ = map(float, lines[0].split())  # Lee, run
                for idx, line in enumerate(lines[1:], start=0):
                    _, _, _, _, mc_stat, _ = map(float, line.split())
                    self.matrix_default_newworld_abs_syst_mcstat[idx, idx] = mc_stat
        else:
            print("\n      ---> MCstat, No")

        # --- Flux
        if getattr(self, 'flag_syst_flux', False):
            print("\n      ---> flux, Yes")
            for idx in range(1, 14):
                with uproot.open(os.path.join(default_fluxXs_dir, f"cov_{idx}.root")) as file:
                    mat = file[f"frac_cov_xf_mat_{idx}"].to_numpy()
                    self.matrix_default_oldworld_rel_syst_flux += torch.tensor(mat, dtype=torch.float32, device=device)
        else:
            print("\n      ---> flux, No")

        # --- Geant
        if getattr(self, 'flag_syst_geant', False):
            print("\n      ---> geant, Yes")
            for idx in range(14, 17):
                with uproot.open(os.path.join(default_fluxXs_dir, f"cov_{idx}.root")) as file:
                    mat = file[f"frac_cov_xf_mat_{idx}"].to_numpy()
                    self.matrix_default_oldworld_rel_syst_geant += torch.tensor(mat, dtype=torch.float32, device=device)
        else:
            print("\n      ---> geant, No")

        # --- Xs
        if getattr(self, 'flag_syst_Xs', False):
            print("\n      ---> Xs, Yes")
            with uproot.open(os.path.join(default_fluxXs_dir, "cov_17.root")) as file:
                mat = file["frac_cov_xf_mat_17"].to_numpy()
                self.matrix_default_oldworld_rel_syst_Xs = torch.tensor(mat, dtype=torch.float32, device=device)
        else:
            print("\n      ---> Xs, No")

        # --- Detector
        if getattr(self, 'flag_syst_det', False):
            print("\n      ---> detector, Yes")
            detector_files = {
                1: "cov_LYDown.root", 2: "cov_LYRayleigh.root", 3: "cov_Recomb2.root",
                4: "cov_SCE.root", 6: "cov_WMThetaXZ.root", 7: "cov_WMThetaYZ.root",
                8: "cov_WMX.root", 9: "cov_WMYZ.root", 10: "cov_LYatt.root"
            }
            for idx, filename in detector_files.items():
                with uproot.open(os.path.join(default_detector_dir, filename)) as file:
                    mat = file[f"frac_cov_det_mat_{idx}"].to_numpy()
                    self.matrix_default_oldworld_rel_syst_det += torch.tensor(mat, dtype=torch.float32, device=device)
        else:
            print("\n      ---> detector, No")
