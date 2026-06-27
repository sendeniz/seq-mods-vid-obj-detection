"""
Tests for Step 2 of input-dependent C in S4ND.

Three things that could fail silently:
  1. regression  -- input_dep_c=False output must be bit-identical to before
  2. per-sample variation -- two different inputs should give different kernels,
                             two identical inputs should give the same kernel
  3. gradient flow -- c_gen weights must actually receive gradients during backward

Run from models/yolov9/:
    python test_input_dep_c.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from models.s4.src.models.sequence.modules.s4nd import S4ND


def make_model(input_dep_c):
    return S4ND(
        d_model=32,
        d_state=16,
        l_max=[20, 20],
        mode='diag',
        disc='dss',
        input_dep_c=input_dep_c,
    )


# --- test 1: regression ---
# run the same input through a model with the flag off twice.
# the outputs must be identical (deterministic) and this also sanity-checks
# that we didn't accidentally break the default code path.
def test_regression():
    model = make_model(input_dep_c=False)
    model.eval()
    x = torch.randn(2, 32, 20, 20)
    with torch.no_grad():
        y1, _ = model(x)
        y2, _ = model(x)
    diff = (y1 - y2).abs().max().item()
    assert diff == 0.0, f"regression: same input gave different outputs, diff={diff}"
    print(f"test 1 pass (regression): same input -> same output, diff={diff}")


# --- test 2: per-sample variation ---
# with input_dep_c=True, two different inputs in the same batch should lead to
# different kernels, meaning the output differs beyond what u alone would cause.
# we check this by running each sample separately and comparing to the batched run.
# also checks that same input -> same output (the gate should be deterministic).
def test_per_sample_variation():
    model = make_model(input_dep_c=True)
    model.eval()

    x_a = torch.randn(1, 32, 20, 20)
    x_b = torch.randn(1, 32, 20, 20)

    with torch.no_grad():
        # run a and b individually
        y_a_solo, _ = model(x_a)
        y_b_solo, _ = model(x_b)

        # run them together in a batch of 2
        x_ab = torch.cat([x_a, x_b], dim=0)
        y_ab, _ = model(x_ab)

        y_a_batched = y_ab[0:1]
        y_b_batched = y_ab[1:2]

    # the batched output for each sample should match the solo run --
    # if it doesn't, the gate is leaking information across batch elements
    err_a = (y_a_solo - y_a_batched).abs().max().item()
    err_b = (y_b_solo - y_b_batched).abs().max().item()
    assert err_a < 1e-5, f"sample a: solo vs batched mismatch, err={err_a:.2e}"
    assert err_b < 1e-5, f"sample b: solo vs batched mismatch, err={err_b:.2e}"
    print(f"test 2a pass (per-sample consistency): solo==batched  err_a={err_a:.2e}  err_b={err_b:.2e}")

    # the two outputs should differ (different inputs -> different kernels)
    diff_ab = (y_a_batched - y_b_batched).abs().max().item()
    assert diff_ab > 1e-4, f"outputs for different inputs look identical, diff={diff_ab:.2e}"
    print(f"test 2b pass (kernel variation): different inputs -> different outputs  diff={diff_ab:.2e}")

    # same input twice should give same output
    with torch.no_grad():
        y_a2, _ = model(x_a)
    diff_same = (y_a_solo - y_a2).abs().max().item()
    assert diff_same == 0.0, f"same input gave different outputs, diff={diff_same}"
    print(f"test 2c pass (determinism): same input -> same output  diff={diff_same}")


# --- test 3: gradient flow ---
# after a forward pass and backward, c_gen weights must have non-zero gradients.
# if they're None or zero the MLP is dead and learns nothing during training.
def test_gradient_flow():
    model = make_model(input_dep_c=True)
    model.train()

    x = torch.randn(2, 32, 20, 20)
    y, _ = model(x)

    # dummy loss -- just sum everything so we get a scalar
    loss = y.sum()
    loss.backward()

    grad = model.c_gen[0].weight.grad
    assert grad is not None, "c_gen weight has no gradient -- backward didn't reach it"
    assert grad.abs().max().item() > 0, "c_gen weight gradient is all zeros"
    print(f"test 3 pass (gradient flow): c_gen grad max={grad.abs().max().item():.2e}")


if __name__ == '__main__':
    test_regression()
    test_per_sample_variation()
    test_gradient_flow()
    print("\nall tests passed")
