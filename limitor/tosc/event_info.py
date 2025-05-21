from dataclasses import dataclass

@dataclass
class EventInfo:
    e2e_pdg: int
    e2e_flag_FC: int
    e2e_Etrue: float
    e2e_Ereco: float
    e2e_weight_xs: float
    e2e_baseline: float
