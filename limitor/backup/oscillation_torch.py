import torch

OSC_MODE = {
    "nue2nue": 1,
    "numu2numu": 2,
    "numu2nue": 3,
    "nue2numu": 4,  # unused in original
    "nueNC": 5,
    "numuNC": 6
}

def prob_oscillation(Etrue, baseline, dm2_41, sin2_2theta_14, sin2_theta_24, mode):
    """
    GPU-ready sterile neutrino oscillation probability calculator.
    Based on TOsc::Prob_oscillaion from C++.

    Args:
        Etrue (float or Tensor): neutrino true energy [GeV]
        baseline (float or Tensor): baseline [km]
        dm2_41 (float or Tensor): delta m^2_41 [eV^2]
        sin2_2theta_14 (float or Tensor): sin^2(2θ14)
        sin2_theta_24 (float or Tensor): sin^2(θ24)
        mode (str or int): one of {nue2nue, numu2numu, numu2nue, ...}

    Returns:
        Tensor: oscillation probability
    """
    # Convert all inputs to tensors if needed
    def to_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

    Etrue = to_tensor(Etrue)
    baseline = to_tensor(baseline)
    dm2_41 = to_tensor(dm2_41)
    sin2_2theta_14 = to_tensor(sin2_2theta_14)
    sin2_theta_24 = to_tensor(sin2_theta_24)

    # Move all tensors to same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Etrue = Etrue.to(device)
    baseline = baseline.to(device)
    dm2_41 = dm2_41.to(device)
    sin2_2theta_14 = sin2_2theta_14.to(device)
    sin2_theta_24 = sin2_theta_24.to(device)

    if isinstance(mode, str):
        mode = OSC_MODE[mode]

    # --- Oscillation logic ---
    y = sin2_2theta_14 / sin2_theta_24
    y = torch.where(torch.abs(y - 1.0) < 1e-6, y.new_tensor(1.0), y)
    x = (1 + torch.sqrt(1 - y)) / 2

    eff_sin2_theta_14 = x
    eff_cos2_theta_14 = 1 - x
    eff_sin2_2theta_14 = 4 * eff_sin2_theta_14 * eff_cos2_theta_14

    sin2_Delta = torch.sin(1.267 * dm2_41 * baseline / Etrue) ** 2

    if mode == 1:  # nue → nue
        prob = 1 - eff_sin2_2theta_14 * sin2_Delta
    elif mode == 2:  # numu → numu
        prob = 1 - 4 * eff_cos2_theta_14 * sin2_theta_24 * (1 - eff_cos2_theta_14 * sin2_theta_24) * sin2_Delta
    elif mode == 3:  # numu → nue
        prob = eff_sin2_2theta_14 * sin2_theta_24 * sin2_Delta
    elif mode == 5:  # nue NC
        prob = 1 - eff_sin2_2theta_14 * (1 - sin2_theta_24) * sin2_Delta
    elif mode == 6:  # numu NC
        prob = 1 - (eff_cos2_theta_14 ** 2) * (4 * sin2_theta_24 * (1 - sin2_theta_24)) * sin2_Delta
    else:
        raise ValueError(f"Unsupported mode {mode}")

    return prob
