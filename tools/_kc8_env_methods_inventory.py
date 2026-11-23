from pathlib import Path
p=Path(r"D:\research\kci_ci_causal_coverage_v2_20261122\code\kci\kci_policy_horizon_env.py")
s=p.read_text(encoding="utf8")
for line in s.splitlines():
 if line.startswith('class ') or line.startswith('    def ') or 'closure' in line or 'known_' in line:
  print(line[:210])
