# Contextual Privacy and Security

This repository hosts projects and datasets from the Contextual Privacy and Security group at Google.

## Camber Dataset

The dataset from the paper
[Privacy Reasoning in Ambiguous Contexts](https://www.arxiv.org/abs/2506.12241) is found in the [camber](camber/) folder.

### Description

The dataset has two components: PrivacyLens+ and ConfAIde+. CAMBER data points
are derived from [PrivacyLens](https://github.com/SALT-NLP/PrivacyLens) and
[ConfAIde](https://github.com/skywalker023/confaide). Each datapoint describes a
hypothetical scenario where a personal detail of a person (subject) is requested
from one agent (sender) by another (recipient). Each datapoint is annotated with
a binary label specifying whether it is appropriate for that personal detail
about the subject be shared by the sender, based on various factors (e.g., prior
consent, sensitivity of the personal detail). The original data points in
PrivacyLens and ConfAIde were synthetically generated. Each CAMBER datapoint
modifies an original data point by synthetically rephrasing or expanding the
original scenarios.

### Structure

CAMBER examples contain 6 data fields:

-   `example_id`: a unique identifier of the data point
-   `example_name`: metadata linking the data point to an instance in the source
    dataset
-   `expansion_type`: the kind of synthetic modification used to generate the
    data point
-   `expansion_dimension`: metadata about the modification used to generate the
    data point
-   `appropriateness_label`: binary label indicating whether the sender should
    share the personal detail of the subject with the recipient
-   `scenario`: Description of the hypothetical scenario. Each scenario contains
    5 fields for PrivacyLens+, and 8 for ConfAIde+.

### Dataset format

The examples are stored in tab-separated CSV files. To read them in pandas
dataframes, use the following code:

```
import pandas as pd
with open("privacylensplus.csv, "r") as f:
  privacylens_dataset = pd.read_csv(f, sep="\t")
with open("confaideplus.csv, "r") as f:
  confaide_dataset = pd.read_csv(f, sep="\t")
```