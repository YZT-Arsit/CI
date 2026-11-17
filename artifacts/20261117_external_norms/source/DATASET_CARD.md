---
license: cc-by-4.0
language:
- en
tags:
- privacy norm
- language model agent
size_categories:
- n<1K
---

# Dataset for "PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action"
<p align="center">
| <a href="https://arxiv.org/abs/2409.00138"><b>Paper</b></a> | <a href="https://github.com/SALT-NLP/PrivacyLens"><b>Code</b></a> | <a href="https://salt-nlp.github.io/PrivacyLens/"><b>Website</b></a> |
</p>

## Overview

<p align="center">
  <img src="https://raw.githubusercontent.com/SALT-NLP/PrivacyLens/main/assets/overview.png" style="width: 90%; height: auto;">
</p>

PrivacyLens is a data construction and multi-level evaluation framework for **evaluating privacy norm awareness of language models in action**.


### What you can do with PrivacyLens?
#### 1. Constructing contextualized data points.
PrivacyLens proposes to uncover privacy-sensitive scenarios with three levels of data points:
- **Privacy-Sensitive Seed:** A negative privacy norm expressed as a 5-tuple, `(data type, data subject, data sender, data recipient, transmission principle)`.
- **Vignette:** An expressive piece of story that captures the essence of a privacy-sensitive seed.
- **Trajectory:** A sequence of LM agent actions and the environment observations that simulate the behavior of an agent in a privacy-sensitive scenario, excluding the final action.


The effort required to obtain data points increases from seed to vignette to trajectory. To help create more contextualized data points, PrivacyLens offers a data construction pipeline that programmatically converts seeds into vignettes and trajectories.

Check out our [codebase](https://github.com/SALT-NLP/PrivacyLens) to use the data construction pipeline to create more data points yourself.


#### 2. Conducting both probing-based and action-based evaluations.
- **Probing-based Evaluation:** PrivacyLens provides multi-choice question template to probe the privacy norm awareness of LMs at different levels.
- **Action-based Evaluation:** PrivacyLens evaluates the final action of an LM agent using the trajectory data points and computes the leakage rate and helpfulness rating of the final action.


## Dataset

In the paper, we collect 493 privacy-sensitive seeds from U.S. privacy regulations, privacy literature on vulnerable groups, and crowdsourcing. We further extend them into vignettes and trajectories.

Each data point is formatted as follows:
```json
{
   "name": "unique id",
   "seed": {
      "data_type": "data type",
      "data_subject": "who the data is about",
      "data_sender": "who sends the data",
      "data_sender_name": "Jane/John",
      "data_recipient": "who the data is sent to",
      "transmission_principle": "how the data is transmitted",
      "source": "source of the seed",
      "source_details": {"...": "backtrace information for seeds collected from regulations or literature"}
   },
   "vignette": {
      "story": "expressive vignette",
      "data_type_concrete": "concrete data type in the vignette",
      "data_subject_concrete": "concrete data subject in the vignette",
      "data_sender_concrete": "concrete data sender in the vignette",
      "data_recipient_concrete": "concrete data recipient in the vignette"
   },
   "trajectory": {
      "user_name": "Jane Doe/John Doe",
      "user_email": "jane.doe@gmail.com/john.doe@gmail.com",
      "user_instruction": "user instruction",
      "toolkits": ["tool 1", "tool 2", "..."],
      "executable_trajectory": "a sequence of agent actions and environment observations",
      "final_action": "the final action type",
      "sensitive_info_items": ["sensitive info item 1", "sensitive info item 2", "..."]
   }
}
```

You can view the data points through the Dataset Viewer provided by Hugging Face Dataset. Since the agent trajectory can be long, you can also use our [data inspection tool](https://github.com/SALT-NLP/PrivacyLens/blob/main/helper/inspect_data.py) with `streamlit run inspect_data.py`.


## Usage
Check out information [here](https://github.com/SALT-NLP/PrivacyLens/tree/main?tab=readme-ov-file#evaluate-lms-privacy-norm-awareness) to see how we use the dataset to probe the privacy norm awareness of LMs and evaluate them in action.

You are encouraged to repurpose the dataset, but please do not use it directly for training.

## Citation
Please cite our paper if you find the dataset useful.
```bibtex
@misc{shao2024privacylensevaluatingprivacynorm,
      title={PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action}, 
      author={Yijia Shao and Tianshi Li and Weiyan Shi and Yanchen Liu and Diyi Yang},
      year={2024},
      eprint={2409.00138},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.00138}, 
}
```
