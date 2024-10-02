from scipy.stats import entropy
from kl_evolution.core.data_objects.serie import Serie


class KLDivergence:

    @staticmethod
    def compute(p: Serie, q: Serie) -> float:
        return KLDivergence.__compute__(p=p, q=q)

    @staticmethod
    def __compute__(p: Serie, q: Serie) -> float:
        if not p or not q:
            raise Exception("Both p and q must be specified")

        is_p_empty = p.__all_eq__(other_value=0)
        is_q_empty = q.__all_eq__(other_value=0)

        if is_p_empty and is_q_empty:
            return 0.0
        elif is_p_empty:
            return 0.0
        elif is_q_empty:
            return float("inf")

        return entropy(pk=p.values, qk=q.values, nan_policy="omit")
