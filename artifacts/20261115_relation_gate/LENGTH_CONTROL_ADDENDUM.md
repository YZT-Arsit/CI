# Additional tokenizer-matched control

The initial 32 fresh maps preserved character length, event count and workflow
length, but the audit found that different random four-letter stems could split
into different Qwen token counts. The failed audit and v1 scripts/results remain.
This is a check of a confound, not a change to a learned model or acceptance gate.

Before running the additional cases: retain the same seed, all original draws,
all thresholds, weights and labels. For meeting case IDs only, use the SAME random
four-letter stem followed by 1 versus 2. No rejection sampling or choice based on
model outcomes. Query mapping and all other strings remain as originally drawn.
Use new cases_length_matched/ and models_length_matched/ directories. Audit all
paired Qwen prompt lengths. Do not merge the two passes as independent evidence.
Identity and exact symbol exchange are unchanged controls. No checkpoint training.

Both passes must be reported; the strictly matched pass is the stronger control
for the user's requested exclusion of prompt-length shortcuts. Token length may
differ across different maps; it is matched BETWEEN the two opposite-label
histories within each map, task, horizon and route binding.
