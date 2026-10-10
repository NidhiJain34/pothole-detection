#!/usr/bin/env python3
"""Read-only pre-training check. Does NOT start training or modify dataset files."""
from pathlib import Path
import subprocess, sys, importlib.util

ROOT = Path('/home/CL502-13/Downloads/latestmergedproject2.v1i.yolo26')
DATA_YAML = ROOT / 'data_final_cleaned.yaml'
DATASET = ROOT / 'final_cleaned_dataset_no_conflicts'
REPORT = ROOT / 'pretraining_hardware_check.txt'

def run_command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return p.returncode, (p.stdout + '\n' + p.stderr).strip()
    except FileNotFoundError:
        return 127, f'Command not found: {args[0]}'
    except subprocess.TimeoutExpired:
        return 124, f'Timed out: {" ".join(args)}'
    except Exception as exc:
        return 1, f'{type(exc).__name__}: {exc}'

def main():
    lines, issues = [], []
    lines += ['YOLO PRE-TRAINING HARDWARE AND CONFIG CHECK', '=' * 64,
              'This script does NOT start training or change dataset files.',
              f'Project root: {ROOT}', f'Python executable: {sys.executable}',
              f'Python version: {sys.version.split()[0]}', '']

    lines += ['PYTORCH / CUDA', '-' * 64]
    if importlib.util.find_spec('torch') is None:
        lines.append('PyTorch: NOT INSTALLED in this Python environment')
        issues.append('PyTorch is missing from the project virtual environment.')
    else:
        try:
            import torch
            lines += [f'PyTorch version: {torch.__version__}',
                      f'CUDA available to PyTorch: {torch.cuda.is_available()}']
            if torch.cuda.is_available():
                lines += [f'CUDA version used by PyTorch: {torch.version.cuda}',
                          f'GPU count: {torch.cuda.device_count()}']
                for i in range(torch.cuda.device_count()):
                    prop = torch.cuda.get_device_properties(i)
                    lines += [f'GPU {i}: {prop.name}', f'  Total VRAM: {prop.total_memory / (1024**3):.2f} GiB']
                lines.append('GPU result: CUDA GPU is available.')
            else:
                lines.append('GPU result: CUDA GPU is not available to PyTorch.')
                issues.append('CUDA is unavailable to PyTorch; confirm GPU/driver before choosing device=0.')
        except Exception as exc:
            lines.append(f'PyTorch check failed: {type(exc).__name__}: {exc}')
            issues.append('Could not check PyTorch/CUDA successfully.')

    lines += ['', 'NVIDIA DRIVER', '-' * 64]
    code, output = run_command(['nvidia-smi'])
    lines.append(output or f'nvidia-smi exit code: {code}')
    if code != 0:
        lines.append('Note: nvidia-smi did not run successfully; there may be no NVIDIA GPU or a driver/path issue.')
    lines += ['', 'ULTRALYTICS', '-' * 64]
    if importlib.util.find_spec('ultralytics') is None:
        lines.append('Ultralytics: NOT INSTALLED in this Python environment')
        issues.append('Ultralytics is missing from the project virtual environment.')
    else:
        try:
            import ultralytics
            lines.append(f'Ultralytics version: {getattr(ultralytics, "__version__", "unknown")}')
        except Exception as exc:
            lines.append(f'Ultralytics import failed: {type(exc).__name__}: {exc}')
            issues.append('Ultralytics could not be imported.')

    lines += ['', 'DATA YAML AND DATASET PATHS', '-' * 64,
              f'YAML exists: {DATA_YAML.is_file()} — {DATA_YAML}']
    if not DATA_YAML.is_file():
        issues.append('data_final_cleaned.yaml is missing.')
    else:
        lines.append('YAML contents:')
        lines.extend('  ' + x for x in DATA_YAML.read_text(encoding='utf-8').splitlines())
    lines.append(f'Dataset folder exists: {DATASET.is_dir()} — {DATASET}')
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    for split in ('train', 'valid', 'test'):
        image_dir, label_dir = DATASET / split / 'images', DATASET / split / 'labels'
        image_count = sum(1 for p in image_dir.rglob('*') if p.is_file() and p.suffix.lower() in image_exts) if image_dir.is_dir() else 0
        label_count = sum(1 for p in label_dir.rglob('*.txt') if p.is_file()) if label_dir.is_dir() else 0
        lines.append(f'{split}: images={image_count:,}, label files={label_count:,}')
        if not image_dir.is_dir() or not label_dir.is_dir():
            issues.append(f'{split} image/label directory is missing.')
        elif image_count != label_count:
            issues.append(f'{split} image and label file counts differ.')

    lines += ['', 'MODEL WEIGHTS', '-' * 64]
    candidates = [ROOT / 'yolo26n.pt', Path.cwd() / 'yolo26n.pt',
                  Path.home() / '.cache' / 'ultralytics' / 'yolo26n.pt',
                  Path.home() / '.config' / 'Ultralytics' / 'yolo26n.pt']
    found = []
    for p in candidates:
        if p.is_file() and p not in found:
            found.append(p)
    if found:
        lines.extend(f'Found: {p} ({p.stat().st_size / (1024*1024):.1f} MiB)' for p in found)
    else:
        lines += ['yolo26n.pt was not found in common local locations.',
                  'Ultralytics may download weights automatically if the installed version supports this and internet is available.',
                  'Do not download weights from an untrusted source.']

    lines += ['', 'RECOMMENDED NEXT ACTION', '-' * 64]
    if issues:
        lines.append('STATUS: REVIEW ITEMS BEFORE TRAINING')
        lines.extend('- ' + x for x in issues)
    else:
        lines += ['STATUS: BASIC CHECKS PASSED',
                  'This does not prove every annotation is correct or guarantee training fits in GPU memory.',
                  'Next, run a short 1–3 epoch smoke test before the full run.']
    lines += ['', 'No training was started.']
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'\nReport saved to: {REPORT}')

if __name__ == '__main__':
    main()
