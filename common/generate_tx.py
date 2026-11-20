"""Shared Maximal Length PRBS (LFSR) sequence generator."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

POLYNOMIALS = {
    7: [7, 6],    # x^7 + x^6 + 1
    9: [9, 5],    # x^9 + x^5 + 1
    11: [11, 9],  # x^11 + x^9 + 1
    15: [15, 14], # x^15 + x^14 + 1
    17: [17, 14], # x^17 + x^14 + 1
    23: [23, 18], # x^23 + x^18 + 1
    31: [31, 28], # x^31 + x^28 + 1
}


def generate_prbs_lfsr(
    order: int,
    seed: int,
    file_path: str,
    taps: Sequence[int],
    mask: int,
) -> None:
    """
    Generates a Maximal Length Sequence (m-sequence) based on the LFSR order.
    The length is automatically set to (2^order - 1).
    """
    if seed == 0:
        raise ValueError("Seed cannot be 0 for an LFSR.")

    max_length = (1 << order) - 1
    abs_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    state = seed & mask
    if state == 0:
        raise ValueError("Masked seed cannot be 0 for an LFSR.")

    chunk_size = 65536
    buffer = bytearray(chunk_size)
    buf_idx = 0

    try:
        with open(file_path, "wb") as f:
            for _ in range(max_length):
                bit = state & 1
                buffer[buf_idx] = 48 + bit
                buf_idx += 1

                if buf_idx == chunk_size:
                    f.write(buffer)
                    buf_idx = 0

                feedback = 0
                for tap in taps:
                    feedback ^= state >> (tap - 1)
                feedback &= 1

                state = ((state << 1) | feedback) & mask

            if buf_idx > 0:
                f.write(buffer[:buf_idx])

        print(f"Successfully saved {max_length} bits to: {file_path}")
    except Exception as exc:
        print(f"Error while saving the file: {exc}", file=sys.stderr)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Maximal Length PRBS Generator")
    parser.add_argument(
        "--order",
        type=int,
        choices=POLYNOMIALS.keys(),
        required=True,
        help="Polynomial order (defines sequence length as 2^n - 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination file path (e.g., ../data/tx.txt)",
    )
    parser.add_argument(
        "--seed", type=int, default=756, help="Initial non-zero seed (default: 756)"
    )

    args = parser.parse_args(argv)
    mask = (1 << args.order) - 1
    taps = POLYNOMIALS[args.order]
    generate_prbs_lfsr(args.order, args.seed, args.output, taps, mask)


if __name__ == "__main__":
    main()
