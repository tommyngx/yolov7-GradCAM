import argparse
import logging
import os
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s", level=logging.INFO)

from models.experimental import attempt_load
from utils.torch_utils import select_device


@torch.no_grad()
def run(weights=None,  # model.pt path(s)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        model=None,
        ):
    # Initialize/load model and set device
    device = select_device(device)
    # Load model
    model = attempt_load(weights, map_location=device)
    model = model.model
    print(model)

    model_list = {}
    for i, layer in model.named_modules():
        model_list[i] = layer

    model_list = {k: v for k, v in model_list.items()}
    # print("modules :",model_list.keys())

    for name in model_list.keys():
        print(f"{name}")

    return None


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov7_100e_64b_pre.pt',
                        help='model.pt path(s)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    return opt


def main(opt):
    logger.info(f'show layers ... ')
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
