"""Baseline forwarding before the contrast-coverage regression is repaired."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'20261117_query_coverage'))
from training_data import schedule,matched_groups
