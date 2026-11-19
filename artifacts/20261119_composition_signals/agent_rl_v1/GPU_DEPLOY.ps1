# KCI-Align 5090 keyed deployment: source transfer + hardware/environment inspection.
# No pip install, no model downloads, no GPU training.
$ErrorActionPreference = "Stop"
$repo = "D:\research\longhistory"
$bundle = Join-Path $repo "artifacts\20261119_composition_signals\agent_rl_v1\gpu_payload_v2.zip"
$key = "C:\Users\hasee\.ssh\kci_gpu_5090"
$hostName = "connect.westd.seetacloud.com"
$port = 47681
$target = "root@$hostName"
$remote = "/root/kci_agent_rl_20261119"
if (!(Test-Path $key) -or !(Test-Path $bundle)) { throw "Missing key or verified source bundle" }
$hash = (Get-FileHash -Algorithm SHA256 $bundle).Hash.ToLower()
if ($hash -ne "1556515ac0ce36b7ead3d77030e3fb143184f5cfe6eef3da2719c765d96e640d") { throw "Source bundle hash mismatch" }
$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$scp = "$env:WINDIR\System32\OpenSSH\scp.exe"
$opts = @("-i",$key,"-o","BatchMode=yes","-o","StrictHostKeyChecking=accept-new","-o","ConnectTimeout=12","-p","$port")
& $ssh @opts $target "mkdir -p '$remote'"
if ($LASTEXITCODE -ne 0) { throw "Key authorization or initial server connection failed" }
$dest = $target + ":" + $remote + "/gpu_payload_v2.zip"
& $scp -i $key -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $port $bundle $dest
if ($LASTEXITCODE -ne 0) { throw "Source upload failed" }
$remoteScript = "set -e; cd '$remote'; echo GPU_PACKAGE_REMOTE_SHA; sha256sum gpu_payload_v2.zip; unzip -q -n gpu_payload_v2.zip; echo GPU_HARDWARE; nvidia-smi --query-gpu=name,memory.total,memory.used,compute_cap --format=csv; echo PYTHON; python --version; echo TORCH_ENV; python -m pip list --format=freeze | grep -Ei 'torch|transformers|peft|bitsandbytes|accelerate|huggingface|safetensors' || true; echo DISK; df -h /root; echo MODEL_CACHES; du -sh /root/.cache/huggingface /root/autodl-tmp /root/models 2>/dev/null || true"
& $ssh @opts $target $remoteScript
if ($LASTEXITCODE -ne 0) { throw "Remote deployment/inspection failed" }
Write-Output "GPU_DEPLOYMENT_AND_HARDWARE_INSPECTION_COMPLETE"
