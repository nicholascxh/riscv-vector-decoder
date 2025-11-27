# RISC-V Vector (RVV) Instruction Decoder

A Python-based decoder for RISC-V Vector (RVV) instructions based on the RVV specification 1.0.

## Overview

This repository contains two Python scripts for decoding RISC-V Vector (RVV) machine code instructions into human-readable assembly mnemonics:

- **`rvv_decoder.py`** - Interactive decoder for single instructions
- **`rvv_decoder_file.py`** - Batch processor for files containing hex instructions

Both scripts share the same core decoding logic but serve different use cases.

## Features

- Decodes RVV 1.0 vector instructions including:
  - Vector arithmetic operations (integer, floating-point, mask)
  - Vector load/store instructions
  - Vector configuration instructions (`vsetvli`, `vsetivli`, `vsetvl`)
  - Special vector operations (moves, conversions, reductions)
- Supports both interactive and batch processing
- Debug mode for detailed decoding information
- File processing with in-place hex-to-assembly conversion

## Usage

### Interactive Decoder (`rvv_decoder.py`)

```bash
# Single instruction decoding
python rvv_decoder.py 020585d7

# Interactive mode
python rvv_decoder.py
> 01077d57
Decoded: vsetvli x26, x14, e32, m1, tu, mu
```

### File Processor (`rvv_decoder_file.py`)

```bash
# Process a file and output to console
python rvv_decoder_file.py -i input.txt

# Process a file and save to output file
python rvv_decoder_file.py -i input.txt -o output.txt

# Enable debug output
python rvv_decoder_file.py -i input.txt -d
```

### Command Line Options

- `-i, --input`: Input file to process (required for file mode)
- `-o, --output`: Output file (optional, defaults to stdout)
- `-d, --debug`: Enable debug output
- `hex_string`: Single hex string to decode (8 digits, no 0x prefix)

## Input Format

The decoder accepts 32-bit instruction encodings as 8-character hexadecimal strings:

- **Without 0x prefix**: `020585d7`
- **With 0x prefix**: `0x020585d7`

For file processing, the script will automatically detect and replace all 8-character hex strings with their decoded assembly equivalents.

## Examples

```bash
# Vector configuration
01077d57 → vsetvli x26, x14, e32, m1, tu, mu

# Vector load
0605b507 → vle64.v v10, (x11)

# Vector arithmetic  
02564557 → vadd.vx v10, v5, x12
```

## Requirements

- Python 3+
- No external dependencies

## References

- [RISC-V Vector Extension Specification 1.0](https://storage.googleapis.com/shodan-public-artifacts/RVV-Specification-Docs/riscv-v-spec-1.0-frozen-for-public-review.pdf)

## Notes

- This decoder is based on the RVV 1.0 specification
- Non-vector instructions are not decoded

## Contributing

This is a personal project, and may be prone to errors. Feel free to submit issues and pull requests to help improve the decoder's accuracy and functionality.
