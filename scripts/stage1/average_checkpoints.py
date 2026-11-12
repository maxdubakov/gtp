"""Average the weights of Stage 1 checkpoints of one run into a new checkpoint.

Usage:
  uv run scripts/stage1/average_checkpoints.py out.pth run/checkpoints/step_0030000.pth run/checkpoints/step_0040000.pth
"""

import sys

import torch


def main(out, paths):
    states = [torch.load(p, map_location='cpu', weights_only=True)['model'] for p in paths]
    avg = {k: (sum(s[k].float() for s in states) / len(states)).to(states[0][k].dtype) for k in states[0]}
    torch.save({'iteration': 0, 'model': avg, 'config': {'averaged': list(paths)}}, out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
