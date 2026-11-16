
# PiSAs: Benchmarking Contextual Integrity in Multi-User Agentic Systems

[Paper](https://arxiv.org/abs/2607.05318) | [Dataset](https://huggingface.co/datasets/ServiceNow/PiSAs)

**PiSAs (Privacy in Shared Agentic Systems)** is a benchmark for evaluating privacy violations in multi-user organizational settings, where agents collaborate across users to complete tasks.

Such systems often need to gather and exchange information distributed across multiple users. This creates privacy risks: agents may disclose information that is inappropriate for the task or expose information to users who should not have access to it.

PiSAs evaluates these two dimensions of **Contextual Integrity (CI)** using separate annotations for each attribute:

- **Appropriateness:** Whether the information is appropriate to use for the given task.
- **Visibility:** Which users are permitted to access the information.

These annotations allow us to measure privacy violations across agent communication, memory, and task execution, while also evaluating task completeness and utility.

PiSAs is system-agnostic and can be used to evaluate different agent architectures, memory configurations, and privacy-preserving methods.

![PiSAs problem setting](docs/assets/problem-setting.png)

## Benchmark

PiSAs contains **265 scenarios across nine tasks** spanning six domains: information technology, healthcare, academia, aviation, cultural heritage, and public administration.

Each scenario consists of an organizational task, a set of users with different roles, and information distributed across them. The benchmark provides ground-truth annotations for task appropriateness, user-level visibility, and the correct task outcome.

Our paper evaluates three agent configurations:

- **Single:** A single agent has access to information from all users.
- **Centralized:** Each user has an agent, with communication routed through a central coordinator.
- **Decentralized:** Each user has an agent, and agents communicate directly through token passing.

We also evaluate the effects of memory configurations, privacy instructions, and specialized privacy mitigations.

The benchmark measures:

- **Appropriateness violations:** Inappropriate information disclosed during task execution or inter-agent communication.
- **Visibility violations:** Information exposed to agents representing users who should not have access to it.
- **Completeness:** The fraction of required task information successfully gathered.
- **Utility:** Whether the system produces the correct task outcome.

The dataset is available on [Hugging Face](https://huggingface.co/datasets/ServiceNow/PiSAs).

## Getting started

### Installation

```bash
git clone https://github.com/ServiceNow/PiSAs.git
cd PiSAs/evaluation

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENROUTER_API_KEY=your_api_key
```

### Run an evaluation

List the available benchmark tasks:

```bash
python run_pipeline.py --list-tasks
```

Evaluate a single-agent system on one task:

```bash
python run_benchmark.py \
    --task uas_flight_readiness \
    --agent-llm anthropic/claude-sonnet-4-6 \
    -s single \
    --run-index 0
```

Evaluate all three agent configurations with three runs per scenario:

```bash
python run_benchmark.py \
    --task uas_flight_readiness \
    --agent-llm anthropic/claude-sonnet-4-6 \
    --run-index 0,1,2 \
    --hide-task-from-peers
```

The evaluation pipeline executes the scenarios, scores privacy violations and task performance, and aggregates the results.

For additional options, including memory configurations, judge models, and evaluation of custom scenarios, see the [evaluation documentation](evaluation/README.md).

## Creating new scenarios

PiSAs supports the construction of new benchmark tasks using task-family seeds.

A seed defines the organizational task, participants, decision rules, required information, and privacy-sensitive attributes. The scenario-generation pipeline uses this specification to construct scenarios with ground-truth task outcomes and privacy annotations.

The `scenario_generation/` directory contains the seeds and generation scripts used for the benchmark.

See the [scenario generation documentation](scenario_generation/README.md) for instructions on creating and validating new scenarios.

## Repository structure

- `evaluation/` — Agent configurations, evaluation pipeline, judges, and metrics.
- `scenario_generation/` — Task seeds, scenario generation, and validation.
- `docs/assets/` — Figures from the paper.

## Citation

If you use PiSAs in your research, please cite our paper:

```bibtex
@article{gupta2026pisas,
  title   = {PiSAs: Benchmarking Contextual Integrity in Multi-User Agentic Systems},
  author  = {Gupta, Shubham and
             Mohammadi Sepahvand, Nazanin and
             Kumar, Abhinav and
             Subakan, Cem and
             Gella, Spandana and
             No{\"e}l, Pierre-Andr{\'e} and
             Taslakian, Perouz and
             Bagdasarian, Eugene and
             Zantedeschi, Valentina},
  journal = {arXiv preprint arXiv:2607.05318},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.05318}
}
```
