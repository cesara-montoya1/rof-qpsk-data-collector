import os
import argparse


def generate_prbs_lfsr(order, seed, file_path, taps, mask):
    """
    Generates a Maximal Length Sequence (m-sequence) based on the LFSR order.
    The length is automatically set to (2^order - 1).
    """
    if seed == 0:
        raise ValueError("Seed cannot be 0 for an LFSR.")

    # Calculate maximal length: 2^n - 1
    max_length = (1 << order) - 1

    # Ensure absolute path for directory creation
    abs_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    sequence = []
    state = seed & mask

    print(f"Generating PRBS{order} sequence...")
    print(f"Maximal Length: {max_length} bits")
    print(f"Polynomial Taps: {taps}")
    print(f"Initial Seed: {seed}")

    for _ in range(max_length):
        # Extract output bit (LSB)
        bit = state & 1
        sequence.append(str(bit))

        # Calculate feedback bit (XOR of specified taps)
        feedback = 0
        for tap in taps:
            feedback ^= state >> (tap - 1)

        feedback &= 1

        # Shift left and apply feedback at the input, mask to maintain order size
        state = ((state << 1) | feedback) & mask

    try:
        with open(file_path, "w") as f:
            f.write("".join(sequence))
        print(f"✅ Successfully saved {max_length} bits to: {file_path}")
    except Exception as e:
        print(f"❌ Error while saving the file: {e}")


if __name__ == "__main__":
    # Standard ITU-T/O.150 Polynomial Taps (1-indexed)
    POLYNOMIALS = {
        7: [7, 6],  # x^7 + x^6 + 1
        9: [9, 5],  # x^9 + x^5 + 1
        11: [11, 9],  # x^11 + x^9 + 1
        15: [15, 14],  # x^15 + x^14 + 1
        17: [17, 14],  # x^17 + x^14 + 1
        23: [23, 18],  # x^23 + x^18 + 1
        31: [31, 28],  # x^31 + x^28 + 1
    }

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

    args = parser.parse_args()

    # Mask to keep the state within the 'order' number of bits
    mask = (1 << args.order) - 1
    taps = POLYNOMIALS[args.order]

    generate_prbs_lfsr(args.order, args.seed, args.output, taps, mask)
