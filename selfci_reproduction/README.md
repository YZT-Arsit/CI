# SelfCI reproduction

The run protocol and measured results are in [PROTOCOL.md](PROTOCOL.md), [REPRODUCE.md](REPRODUCE.md) and [RESULTS.md](RESULTS.md). This snapshot includes the evaluation wrappers, public model and dataset revisions, dependency versions and aggregate measurements.

From this directory, prepare the upstream implementation with:

```sh
git clone https://github.com/sw-programmer/SelfCI.git upstream
git -C upstream checkout --detach 9cf9b2a53242c69e5f46150bbc34aea0859b98e4
python fetch_adapters.py
```

Download the two public parquet files from the pinned URLs in [data/sources.json](data/sources.json) to their listed paths under `data/`, then verify their SHA-256 digests before running `prepare_data.py`. The run scripts use Linux GPU environments and the pinned base models described in the reproduction guide. The KCI transfer evaluation additionally requires the separately held role-visible development assets. Row-level generated replies, local checkpoints and development-source records referenced in the run notes are stored with the local experiment.
