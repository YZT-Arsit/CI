"""Private CPU-only dependency setup; not an experiment or global install."""
import sys,subprocess,json,os,time
from pathlib import Path
R=Path(__file__).resolve().parent
old=json.loads((R.parent/'20261115_semantic_repair/wsl_environment.json').read_text())
target=Path.home()/'.cache/kci_lm_deps_20261116/site-packages';target.mkdir(parents=True,exist_ok=True)
packages=['transformers==4.51.3','peft==0.15.2','accelerate==1.6.0','safetensors==0.5.3',
 'huggingface_hub==0.30.2','tokenizers==0.21.4','packaging==25.0','pyyaml==6.0.2',
 'regex==2024.11.6','tqdm==4.67.1','requests==2.32.3','urllib3==2.3.0',
 'certifi==2025.1.31','charset_normalizer==3.4.1','idna==3.10','psutil==6.1.1']
t=time.perf_counter()
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(target),
 '--report',str(R/'lm_install_report.json'),*packages],check=True)
(R/'environment.json').write_text(json.dumps({'torch_path':old['private_package_path'],
 'lm_dependency_path':str(target),'python':sys.version,'cpu_only':True,
 'setup_seconds':time.perf_counter()-t,'packages':packages},indent=2))
