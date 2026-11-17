"""Paths and integrity for the preserved parent experiment."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];PARENT=R.parent/'20261117_query_coverage'
sys.path.insert(0,str(PARENT))
from probe_data import OLD,read,rows,sha,verify_prior as verify_older


def verify_prior():
    verify_older();plan=read(PARENT/'PLAN.json');seal=read(PARENT/'SEALED.json')
    assert seal['complete'] and all(sha(PARENT/n)==h for n,h in seal['files'].items())
    assert all(sha(PARENT/n)==h for n,h in {**plan['source_hashes'],**plan['data_hashes']}.items())
    assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items())
