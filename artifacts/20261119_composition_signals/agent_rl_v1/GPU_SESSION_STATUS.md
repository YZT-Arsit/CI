# 5090 session deployment handoff

User-supplied SeeTaCloud rental is reachable on its published TCP SSH port.
Direct key-authenticated SSH remains unapproved by the server (Permission denied).
**No server commands, model downloads or GPU training have happened.**

On the authorized Windows computer, generated dedicated SSH key:
- Public key: C:\Users\hasee\.ssh\kci_gpu_5090.pub
- Private key: C:\Users\hasee\.ssh\kci_gpu_5090 (do not disclose)
- SeeTaCloud console > container instance > set SSH key login: user must register the above PUBLIC key.

Temporary password staging file was deleted, and its nonexistence verified.

Deployment payload:
- artifacts\20261119_composition_signals\agent_rl_v1\gpu_payload_v2.zip
- SHA256 1556515ac0ce36b7ead3d77030e3fb143184f5cfe6eef3da2719c765d96e640d
- About 243KB, 80 files; KCI source code, immutable source anchors, GPU instructions.
- Does not include old model weights or any credentials.
- Windows key deployment script: artifacts\20261119_composition_signals\agent_rl_v1\GPU_DEPLOY.ps1
- Script currently stops at initial SSH authentication, before SCP.

When public key is registered: use the existing Windows Remote Desktop Commander to run the GPU_DEPLOY.ps1 script, inspect nvidia-smi/Python/model cache and then execute inference-only 4B rollout. Train only if actual 4B rollout sample has useful safe-success/violation outcome contrasts, then cap GRPO optimizer updates.

IMPORTANT: Blackwell RTX5090 needs CUDA 12.8+ compatible PyTorch and current bitsandbytes. Qwen/Qwen3-4B-Instruct-2507 requires transformers >=4.51; use local cached model or a new clean download directory, not historical CPU model caches. The 4B path has not been smoke tested on this rented server.

No background work has been scheduled. This file records the exact remaining authorization blocker.
