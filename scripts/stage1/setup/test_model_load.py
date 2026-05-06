"""Verify that the Kong CRNN model loads from the pretrained checkpoint and
produces a valid forward pass on a synthetic 10-second audio clip.
"""

import torch

from gtp import REPO_ROOT
from gtp.stage1.model.kong import Note_pedal

CHECKPOINT_PATH = str(REPO_ROOT / 'models' / 'pretrained' / 'CRNN_note_F1=0.9677_pedal_F1=0.9186.pth')

FRAMES_PER_SECOND = 100
CLASSES_NUM = 88
SAMPLE_RATE = 16000
CLIP_SECONDS = 10


def main():
    # Select device: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # 1. Instantiate model
    print('Building model...')
    model = Note_pedal(frames_per_second=FRAMES_PER_SECOND, classes_num=CLASSES_NUM)

    # 2. Load pretrained checkpoint
    print(f'Loading checkpoint from: {CHECKPOINT_PATH}')
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print('Checkpoint loaded successfully.')

    # 3. Forward pass on a random 10-second audio clip
    num_samples = SAMPLE_RATE * CLIP_SECONDS
    x = torch.randn(1, num_samples, dtype=torch.float32).to(device)
    print(f'Input shape: {tuple(x.shape)}')

    with torch.no_grad():
        output = model(x)

    # 4. Print output shapes
    print('Output shapes:')
    for key, tensor in output.items():
        print(f'  {key}: {tuple(tensor.shape)}')

    print('Forward pass succeeded.')


if __name__ == '__main__':
    main()
