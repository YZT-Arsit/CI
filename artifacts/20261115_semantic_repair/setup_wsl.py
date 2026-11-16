"""Install CPU-only dependencies in a persistent private Linux cache."""
from pathlib import Path
import subprocess,sys,json

R=Path(__file__).resolve().parent
target=Path.home()/'.cache/kci_semantic_deps_20261115'
subprocess.run([sys.executable,'-m','pip','install','torch==2.15.0+cpu','numpy==2.5.4',
    '--index-url','https://download.pytorch.org/whl/cpu','--target',str(target),
    '--report',str(R/'wsl_install_report.json')],check=True)
(R/'wsl_environment.json').write_text(json.dumps({'private_package_path':str(target),'python':sys.version,
    'cpu_only':True,'global_environment_modified':False},indent=2))
