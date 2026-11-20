"""Backward-compatible re-export shim for shared EVM calculation utilities."""

from common.evm import compute_evm, compute_theoretical_evm_from_snr

__all__ = ["compute_evm", "compute_theoretical_evm_from_snr"]
