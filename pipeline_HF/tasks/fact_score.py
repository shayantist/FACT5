
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_fact_score_label(verdicts):
    """
    Generates a fact score label based on the verdicts provided. The fact score label can be one of the following:
    - True: All atomic claims are true.
    - Mostly True: More than half of the atomic claims are true.
    - Half True: Half of the atomic claims are true.
    - Mostly False: More than half of the atomic claims are false.
    - Pants on Fire: All atomic claims are false.
    - Unverifiable: The number of unverifiable atomic claims is greater than or equal to the number of true/false atomic claims.

    Args:
        verdicts (list): A list of verdicts (True/False/Unverifiable) for each atomic claim within a statement.

    Returns:
        str: The fact score label.
    """

    label = 'Unknown'
    perc_unverified = 0
    v_cleaned = verdicts
    if 'Unveriable' in verdicts:
        v_cleaned = verdicts.remove('Unverifiable')
        perc_unverified = Counter(verdicts)['Unverifiable'] / len(verdicts)
    perc_true = Counter(verdicts)['True'] / len(verdicts)
    perc_false = Counter(verdicts)['False'] / len(verdicts)
    perc = [perc_true, perc_false, perc_unverified]
    winner = np.argwhere(perc == np.amax(perc))

    if len(winner) == 3:  # three-way tie
        label = "Unverifiable"

    elif len(winner) == 2:  # two-way tie
        if 0 in winner and 1 in winner:  # half true
            label = 'Half True'
        elif 0 in winner and 2 in winner:  # true & unverifable
            label = "Unverifiable"
        elif 1 in winner and 2 in winner:  # false & unverifable
            label = "Unverifiable"

    elif winner == 0:
        if perc_true == 1:  # all true
            label = "True"
        elif Counter(v_cleaned)['True'] / len(v_cleaned) > 0.5:  # mostly true
            label = "Mostly True"

    elif winner == 1:
        if perc_false == 1:  # all false
            label = "Pants on Fire"
        elif Counter(v_cleaned)['False'] / len(v_cleaned) > 0.5:  # mostly false
            label = "Mostly False"

    elif winner == 2:
        label = 'Unverifiable'
    return label