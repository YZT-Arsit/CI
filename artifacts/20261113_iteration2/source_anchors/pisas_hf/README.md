---
license: apache-2.0
task_categories:
- text-generation
tags:
- privacy
- contextual-integrity
- multi-agent
- benchmark
---

# PiSAs: Benchmarking Contextual Integrity in Multi-User Agentic Systems

Paper: [arXiv:2607.05318](https://arxiv.org/abs/2607.05318)

PiSAs (Privacy in Shared Agentic systems) is a benchmark for **contextual integrity in multi-user, multi-agent LLM systems**. Each scenario places an executor agent in an organisation where it serves several users at once. The agent is given a decision to make, and the information it needs is spread across colleagues and their agents. That information is mixed in with details whose flow into this decision would be inappropriate under the norms of the context in which they were originally shared. A system is scored on two things: whether it reaches the right decision, and which information flows it creates, meaning what it gathers, what it passes between agents and users, and what it discloses in its final answer.

## Tasks

| Task | Decision | Scenarios |
|---|---|---|
| `meeting_allocation` | schedule meetings into rooms and time slots | 35 |
| `JIRA_allocation` | assign tickets to engineers | 25 |
| `severity_classification` | set an incident's severity level | 25 |
| `inpatient_discharge` | choose a discharge disposition | 30 |
| `thesis_readiness` | decide whether a student proceeds to defense | 30 |
| `manuscript_submission` | judge a manuscript's submission readiness | 30 |
| `uas_flight_readiness` | clear or ground a Part 107 drone flight | 30 |
| `outgoing_museum_loan` | release or hold a museum loan shipment | 30 |
| `special_event_permit_readiness` | approve or refuse a special-event permit | 30 |

265 scenarios, 5751 attributes in total.

## Layout

```
<task>/scenario_NN/
  scenario.json        cast, organisation, task text, timeline of artifacts each person holds
  utility.json         attribute values, the oracle answer, the decision rule, required evidence
  appropriateness.json per attribute: appropriate (a legitimate input) or inappropriate (private)
  visibility.json      per attribute: holder, visible_to, hidden_from
```

`appropriateness.json` defines what a system should and should not use. `visibility.json` defines
who may legitimately know each fact, and is what agent-level privacy violations are scored against.

## Licence

Apache 2.0. All scenarios are synthetic: the people, organisations and records in them are invented.
