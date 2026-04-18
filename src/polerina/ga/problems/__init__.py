from polerina.ga.problems.base import Problem
from polerina.ga.problems.mis import MIS
from polerina.ga.problems.maxcut import MaxCut

def get_problem(name: str) -> Problem:
    if name.lower() == "mis":
        return MIS()
    elif name.lower() == "maxcut":
        return MaxCut()
    else:
        raise ValueError(f"Unknown problem: {name}")
