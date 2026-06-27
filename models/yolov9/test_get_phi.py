"""
Test SSMKernelDiag.get_phi(L).

The idea: the kernel forward() computes K[c,h,l] = Re(sum_n C[c,h,n] * phi[h,n,l])
but doesn't expose phi separately. get_phi() returns phi (shape H,N,L) so we can
swap in our own C later. This test checks the factorization holds by recontracting
C_raw (from _get_params) with phi and comparing to the original K.

Run from models/yolov9/:
    python test_get_phi.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from models.s4.src.models.sequence.kernels.ssm import SSMKernelDiag


def run_test(disc, H=8, N=16, C_ch=4, L=20, bandlimit=None):
    print(f"\ndisc={disc}  H={H}  N={N}  C_ch={C_ch}  L={L}  bandlimit={bandlimit}")

    kernel = SSMKernelDiag(
        d_model=H,
        d_state=N,
        channels=C_ch,
        l_max=L,
        disc=disc,
        bandlimit=bandlimit,
    )
    kernel.eval()

    with torch.no_grad():
        # normal forward -- what we want to reproduce
        K_orig, _ = kernel(L)           # (C_ch, H, L) real
        print(f"  K_orig  shape={tuple(K_orig.shape)}  range=[{K_orig.min():.4f}, {K_orig.max():.4f}]")

        # phi from get_phi -- (H, N, L) complex
        phi = kernel.get_phi(L)
        print(f"  phi     shape={tuple(phi.shape)}  |phi| range=[{phi.abs().min():.4f}, {phi.abs().max():.4f}]")

        # raw C before B is multiplied in -- (C_ch, H, N) complex
        _, _, _, C_raw = kernel._get_params()
        print(f"  C_raw   shape={tuple(C_raw.shape)}")

        # recontract: should recover K_orig
        K_recomp = torch.einsum('chn,hnl->chl', C_raw, phi).real.float()
        print(f"  K_recomp shape={tuple(K_recomp.shape)}  range=[{K_recomp.min():.4f}, {K_recomp.max():.4f}]")

        # first 5 values for visual check
        print(f"  K_orig  [c=0,h=0,:5] = {K_orig[0,0,:5].tolist()}")
        print(f"  K_recomp[c=0,h=0,:5] = {K_recomp[0,0,:5].tolist()}")

        max_err = (K_orig - K_recomp).abs().max().item()
        rel_err = (K_orig - K_recomp).abs().mean().item() / (K_orig.abs().mean().item() + 1e-8)
        print(f"  max_err={max_err:.2e}  rel_err={rel_err:.2e}  ", end="")

        ok = max_err < 1e-4
        print("PASS" if ok else "FAIL")
        return ok


if __name__ == '__main__':
    results = [
        run_test('zoh'),
        run_test('dss'),
        run_test('dss', bandlimit=0.1),  # config used by the best model
    ]

    print()
    if all(results):
        print("all tests passed -- safe to use einsum('bchn,hnl->bchl', C_gen(u), phi).real")
    else:
        print("some tests failed")
