# Native input validation

The Python bindings validate every shape, numeric value, dimension order,
cumulative query norm, offset array, grouped-corpus boundary, panel alignment,
checkpoint list, native integer conversion, and result count before calling a
C++ kernel. Reusable `PackedCorpusDimMajor`, `PackedCorpusWide`, and
`PackedCorpusPanels` objects own immutable arrays, so validated metadata cannot
be changed between queries.

The C ABI repeats the safety-critical checks and returns `UINT64_MAX` for an
invalid call. This protects callers that bypass Python from inconsistent
metadata, unsafe size arithmetic, and invalid result buffers. Python converts
that sentinel into an exception.

Run the focused validation gate after building portable kernels:

```sh
UV_CACHE_DIR=/tmp/bondmaxsim-uv-cache uv run pytest -q tests/test_native_validation.py
```

Run the native ASan/UBSan boundary harnesses with:

```sh
make test-sanitize
```

The fused harness covers query lengths 1, 24, 25, 192, 193, and 257, all
production fused entry points, the maximum eight checkpoints, malformed panel
metadata, and one- and two-thread execution. The per-document and wide-block
harnesses cover all of their public entry points plus malformed offsets and
group metadata.
