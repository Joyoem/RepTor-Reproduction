import argparse
import copy
import torch
from reptor import reptor9, reptor_a
from reptor.utils import count_parameters


def build(name: str):
    return reptor9() if name == "reptor9" else reptor_a()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["reptor9", "reptor_a"], default="reptor9")
    parser.add_argument("--frames", type=int, default=101)
    args = parser.parse_args()

    model = build(args.model).eval()
    x = torch.randn(1, 40, args.frames)
    with torch.no_grad():
        y = model(x)
    deployed = copy.deepcopy(model).switch_to_deploy().eval()

    print(f"Input shape       : {tuple(x.shape)}")
    print(f"Output shape      : {tuple(y.shape)}")
    print(f"Train-graph params: {count_parameters(model):,}")
    print(f"Deploy params     : {count_parameters(deployed):,}")

    try:
        from thop import profile
        macs, params = profile(deployed, inputs=(x,), verbose=False)
        print(f"Deploy MACs       : {macs / 1e6:.3f} M")
        print(f"THOP params       : {params / 1e3:.3f} K")
    except ImportError:
        print("THOP not installed; skipping MAC count.")


if __name__ == "__main__":
    main()
