from pathlib import Path

import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image


def create_fig(fn, img_data, num_img_row, save_figure=False):
    if save_figure:
        save_image(img_data.data.cpu(), fn, nrow=num_img_row)
    grid = make_grid(img_data, nrow=num_img_row)
    return (
        grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to('cpu', torch.uint8)
            .numpy()
    )


def text_sample_to_file(log_tag: str, text_sample: str, epoch: int, exp_path: Path):
    """Write the generated text sample to a file with name log_tag."""
    base_path = exp_path / 'text_gen'
    if not base_path.exists():
        base_path.mkdir()
    file_path = base_path / log_tag
    with open(file_path, 'a') as textfile:
        textfile.write(f'\n{"*" * 20}\n Epoch: {epoch}\n{text_sample}\n{"*" * 20}')
