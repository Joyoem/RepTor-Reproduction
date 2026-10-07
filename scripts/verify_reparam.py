import copy
import torch
from reptor import reptor9


def main() -> None:
    torch.manual_seed(0)
    model = reptor9(num_classes=12).train()
    with torch.no_grad():
        for _ in range(8):
            model(torch.randn(16, 40, 101))

    model.eval()
    x = torch.randn(4, 40, 101)
    with torch.no_grad():
        y_before = model(x)

    deployed = copy.deepcopy(model).switch_to_deploy().eval()
    with torch.no_grad():
        y_after = deployed(x)

    max_abs_diff = (y_before - y_after).abs().max().item()
    mean_abs_diff = (y_before - y_after).abs().mean().item()
    print(f"max abs diff : {max_abs_diff:.8e}")
    print(f"mean abs diff: {mean_abs_diff:.8e}")
    if max_abs_diff > 1e-4:
        raise SystemExit("FAILED: train-graph and deploy-graph outputs are not equivalent.")
    print("PASS: structural re-parameterization is numerically equivalent.")


if __name__ == "__main__":
    main()
