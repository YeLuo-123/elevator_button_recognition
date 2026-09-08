"""Run shipped button-state weights, then render the temporal comparison video."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check_weight(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing weight: {path}. Run git lfs pull or extract the full bundle.')
    with path.open('rb') as stream:
        if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise RuntimeError(f'{path} is a Git LFS pointer. Run git lfs pull before inference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/predict'/datetime.now().strftime('button_state_%Y%m%d_%H%M%S'))
    parser.add_argument('--adapter', type=Path, default=ROOT/'models/button_state_lora')
    parser.add_argument('--base', help='Local model directory or Hugging Face model ID')
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f'Video not found: {args.source}')
    if args.batch < 1:
        parser.error('--batch must be positive')
    if args.output.exists():
        parser.error(f'Output directory already exists: {args.output}')
    check_weight(ROOT/'models/best.pt')
    check_weight(args.adapter/'adapter_model.safetensors')
    local_base = ROOT/'models/Qwen3-VL-2B-Instruct'
    base = args.base or (str(local_base) if (local_base/'config.json').is_file() else 'Qwen/Qwen3-VL-2B-Instruct')
    if Path(base).is_dir():
        check_weight(Path(base)/'model.safetensors')
    import torch
    if not torch.cuda.is_available():
        parser.error('This inference pipeline requires CUDA. Install a compatible NVIDIA driver and CUDA-enabled PyTorch.')
    source = str(args.source.resolve())
    output = args.output.resolve()
    print(f'Base model: {base}\nAdapter: {args.adapter}\nOutput: {output}', flush=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/test_button_state_video.py'),
        '--source', source, '--output', str(output/'raw'), '--adapter', str(args.adapter.resolve()),
        '--base', base, '--batch', str(args.batch)], check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/test_temporal_video.py'),
        '--source', source, '--predictions', str(output/'raw/predictions.jsonl'),
        '--output', str(output/'temporal')], check=True)
    print(f'Annotated video: {output / "temporal/temporal.mp4"}\nComparison: {output / "temporal/comparison.mp4"}')


if __name__ == '__main__':
    main()
