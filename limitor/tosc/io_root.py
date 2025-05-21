# model/io_root.py

import uproot
import awkward as ak
import numpy as np
from ..data_structures import EventInfo

def load_hist_vector_from_root(file_path, hist_names, device="cuda"):
    """
    Loads a list of 1D histograms into a vectorized tensor.

    Args:
        file_path (str): ROOT file
        hist_names (list of str): Names of TH1D histograms to extract
        device (str): Target device ('cuda' or 'cpu')

    Returns:
        Tensor: shape (len(hist_names), nbins)
    """
    with uproot.open(file_path) as f:
        hist_tensors = []
        for name in hist_names:
            h = f[name]
            counts = torch.tensor(h.values(flow=False), dtype=torch.float32, device=device)
            hist_tensors.append(counts)
        return torch.stack(hist_tensors)


def load_matrix_from_root(file_path, key, device="cuda"):
    """
    Loads a 2D matrix from a ROOT file (assumes flat array encoding or TMatrixD serialization).

    Args:
        file_path (str): path to ROOT file
        key (str): object name
        device (str): 'cuda' or 'cpu'

    Returns:
        Tensor: 2D tensor
    """
    with uproot.open(file_path) as f:
        obj = f[key]
        if hasattr(obj, "to_numpy"):
            array = obj.to_numpy()[0]
        else:
            array = obj.values
        return torch.tensor(array, dtype=torch.float32, device=device)


def load_eventinfo_tree(mc_pot_file, data_pot_file, vec_ratioPOT,
                        mc_e2e_file, tree_name, vec_vec_eventinfo):
    """
    Replaces TOsc::Set_oscillation_base_subfunc.
    Fills vec_vec_eventinfo with a list of EventInfo dataclass instances,
    and optionally pushes POT scale into vec_ratioPOT.
    """
    # Open ROOT file with event tree
    with uproot.open(mc_e2e_file) as f:
        tree = f[tree_name]
        arrs = tree.arrays(["e2e_pdg", "e2e_Etrue", "e2e_Ereco", "e2e_weight_xs", "e2e_baseline"])

    # Build EventInfo objects
    event_list = []
    for pdg, Etrue, Ereco, w, L in zip(arrs["e2e_pdg"], arrs["e2e_Etrue"], arrs["e2e_Ereco"],
                                       arrs["e2e_weight_xs"], arrs["e2e_baseline"]):
        event_list.append(EventInfo(pdg=pdg, Etrue=Etrue, Ereco=Ereco, weight=w, baseline=L))

    vec_vec_eventinfo.append(event_list)

    if vec_ratioPOT is not None:
        mc_pot = read_first_pot(mc_pot_file)
        data_pot = read_first_pot(data_pot_file)
        vec_ratioPOT.append(data_pot / mc_pot)
        print(f"            ---> MC POT {mc_pot:.3e}\t{mc_pot_file}")
        print(f"            ---> DD POT {data_pot:.3e}\t{data_pot_file}")

    print(f"            ---> entries {len(event_list):10d}     {mc_e2e_file:50s}   --> {tree_name}")

def read_first_pot(filename):
    with uproot.open(filename) as f:
        tree = f["T"]
        return tree["pot"].array(entry_start=0, entry_stop=1)[0]
