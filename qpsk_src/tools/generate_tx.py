"""Backward-compatible shim delegating to common.generate_tx."""

from common.generate_tx import POLYNOMIALS, generate_prbs_lfsr, main

__all__ = ["POLYNOMIALS", "generate_prbs_lfsr", "main"]

if __name__ == "__main__":
    main()
