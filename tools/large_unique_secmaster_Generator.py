#!/usr/bin/env python3
import argparse
import os
import string
import itertools

def symbol_generator():
    """Generate unique symbol strings:
    First, all letter-only combinations from 1 to 4 characters.
    Then, if exhausted, generate 4-char base36 strings.
    """
    letters = string.ascii_uppercase
    # Generate letter-only combinations (1-4 letters)
    for length in range(1, 5):
        for tup in itertools.product(letters, repeat=length):
            yield ''.join(tup)
    # If we run out, generate 4-character base36 strings.
    i = 0
    while True:
        s = base36(i)
        if len(s) > 4:
            break
        yield s.zfill(4)
        i += 1

def base36(num):
    """Convert an integer to a base36 string."""
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if num == 0:
        return "0"
    result = ""
    while num:
        num, rem = divmod(num, 36)
        result = chars[rem] + result
    return result

def cusip_generator():
    """Generate sequential 9-digit CUSIP strings, from '000000000' to '999999999'."""
    i = 0
    while i <= 999999999:
        yield str(i).zfill(9)
        i += 1

def parse_size(size_str):
    """Convert a size string (e.g., '60GB', '1024MB', '512KB') to bytes."""
    size_str = size_str.strip().upper()
    if size_str.endswith("GB"):
        return float(size_str[:-2]) * 1024**3
    elif size_str.endswith("MB"):
        return float(size_str[:-2]) * 1024**2
    elif size_str.endswith("KB"):
        return float(size_str[:-2]) * 1024
    else:
        return float(size_str)

def main():
    parser = argparse.ArgumentParser(description="Generate a large security master CSV file")
    parser.add_argument("--size", required=True,
                        help="Target file size (e.g., 60GB, 1024MB, 512KB)")
    parser.add_argument("--output", required=True,
                        help="Output file path")
    args = parser.parse_args()

    target_size = parse_size(args.size)
    output_path = args.output

    # Write header and then rows until target file size is reached or unique values run out.
    header = "Symbol,CUSIP,Description,Region,Exchange,AssetClass,Currency,Country\n"
    # Constant columns for all rows.
    description = "Security Template 1"
    region = "US"
    exchange = "NYSE"
    asset_class = "Equity"
    currency = "USD"
    country = "USA"
    row_template = "{symbol},{cusip},{description},{region},{exchange},{asset_class},{currency},{country}\n"

    sym_gen = symbol_generator()
    cusip_gen = cusip_generator()

    bytes_written = 0
    row_count = 0

    with open(output_path, "w", newline="") as f:
        f.write(header)
        bytes_written += len(header.encode("utf-8"))
        print(f"Header written ({len(header.encode('utf-8'))} bytes)")

        # Generate rows until we reach the target file size.
        while bytes_written < target_size:
            try:
                symbol = next(sym_gen)
                cusip = next(cusip_gen)
            except StopIteration:
                print("Ran out of unique values!")
                break

            row = row_template.format(
                symbol=symbol,
                cusip=cusip,
                description=description,
                region=region,
                exchange=exchange,
                asset_class=asset_class,
                currency=currency,
                country=country
            )
            f.write(row)
            bytes_written += len(row.encode("utf-8"))
            row_count += 1

            if row_count % 1000000 == 0:
                print(f"Generated {row_count} rows, file size: {bytes_written / (1024**3):.2f} GB")

    print(f"Finished: Generated {row_count} rows, total file size: {bytes_written / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
