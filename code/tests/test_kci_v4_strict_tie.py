"""Exact-tie behavior must not be treated as correct RAW by ID order."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_v4_pairwise_strict_gate import exact_action,strict_correct
def test_ambiguous_exact_tie_is_unresolved_not_correct():
    scores={"ABORT":-18.0,"SEND_RAW_R":-3.5,"VERIFY_PRIVATE_R":-3.5}
    assert exact_action(scores)=="UNRESOLVED_TIE"
    assert not strict_correct(scores,"SEND_RAW_R")
    assert not strict_correct(scores,"VERIFY_PRIVATE_R")
def test_unequal_semantic_scores_are_strictly_classified():
    scores={"ABORT":-18.0,"SEND_RAW_R":-4.25,"VERIFY_PRIVATE_R":-3.5}
    assert exact_action(scores)=="VERIFY_PRIVATE_R"
    assert strict_correct(scores,"VERIFY_PRIVATE_R")
    assert not strict_correct(scores,"SEND_RAW_R")
