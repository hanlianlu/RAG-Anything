# [Feature Request]: Replace Docling Parser CLI subprocess with Python API

> **Status:** Draft — proposed for [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything)
>
> A reference implementation is available at
> [hanlianlu/RAG-Anything](https://github.com/hanlianlu/RAG-Anything)
> (commits [`dca256c`](https://github.com/hanlianlu/RAG-Anything/commit/dca256c79803195ee5b2a400b18e0ce3e61f4768)
> and [`6bb76fb`](https://github.com/hanlianlu/RAG-Anything/commit/6bb76fb030bd8e3526e7292a108fac7995f2cc9b)).

---

## Summary

Replace the current `DoclingParser` implementation that shells out to the
`docling` CLI via `subprocess.run` with a direct integration through the
[Docling Python API](https://github.com/DS4SD/docling)
(`docling.document_converter.DocumentConverter`).

This eliminates process-spawning overhead, avoids disk I/O round-trips for
intermediate JSON/Markdown files, and enables in-memory model reuse across
consecutive parse calls — yielding significant performance gains for
multi-document workloads while preserving full backward compatibility.

---

## Motivation / Problem

The current `DoclingParser` in HKUDS/RAG-Anything invokes Docling through
its command-line interface:

```python
# Current approach (raganything/parser.py – _run_docling_command)
cmd = [
    "docling",
    "--output", str(file_output_dir),
    "--to", "json",
    "--to", "md",
    str(input_path),
]
result = subprocess.run(cmd, **docling_subprocess_kwargs)
```

After the subprocess completes, output files are read back from disk
(`_read_output_files`). This pattern has several drawbacks:

| Issue | Impact |
|-------|--------|
| **Process-spawn overhead** | Each `parse_*` call forks a new process, loads the Python interpreter, and re-initializes all Docling models from scratch. |
| **Disk I/O round-trip** | Docling writes JSON + Markdown to disk; the parser then reads them back. This is unnecessary when the data is immediately consumed in-memory. |
| **No model reuse** | Docling's deep-learning models (table structure, OCR, layout) are loaded fresh on every invocation — the most expensive part of the pipeline. |
| **Fragile error handling** | Errors surface as `subprocess.CalledProcessError` with stderr strings rather than typed Python exceptions with full stack traces. |
| **Platform quirks** | Windows requires `CREATE_NO_WINDOW` flags; PATH must include the `docling` entry-point. These are unnecessary when calling Python directly. |
| **Limited pipeline control** | CLI flags expose only a subset of Docling's configuration surface. The Python API offers fine-grained control over pipeline options, format options, and OCR settings. |

---

## Proposed Solution

### Core Idea

Use `docling.document_converter.DocumentConverter` directly in Python:

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# Build pipeline options from user kwargs
pipeline_options = PdfPipelineOptions()
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
pipeline_options.do_ocr = True

# Create converter (reused across calls)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    }
)

# Parse document — no subprocess, no disk I/O
result = converter.convert(str(input_path))
doc_dict = result.document.export_to_dict()

# Convert to MinerU-compatible content-list format
content_list = self.read_from_block_recursive(
    doc_dict["body"], "body", img_output_dir, 0, "0", doc_dict
)
```

### Architecture Changes

The refactored `DoclingParser` introduces the following internal methods:

| Method | Purpose |
|--------|---------|
| `_ensure_docling_imports()` | Lazy-import `docling` modules on first use; raises `ImportError` with actionable message if the package is missing. |
| `_build_pipeline_options(**kwargs)` | Map user-facing kwargs (`table_mode`, `tables`, `allow_ocr`, `artifacts_path`) to `PdfPipelineOptions` attributes. |
| `_validate_kwargs(**kwargs)` | Fail-fast on unknown keyword arguments to catch typos early. MinerU-specific kwargs are silently ignored for cross-parser compatibility. |
| `_make_converter_key(**kwargs)` | Produce a hashable cache key from converter-relevant kwargs only. |
| `_get_converter(**kwargs)` | Return a cached `DocumentConverter` instance, creating a new one only when the effective configuration changes. |
| `_parse_with_python_api(input_path, output_dir, **kwargs)` | Core parsing entry-point: validates, converts, and transforms the result into the MinerU-compatible content-list format. |

### Methods Removed

| Method | Reason |
|--------|--------|
| `_run_docling_command()` | Replaced by `_parse_with_python_api()` — no subprocess needed. |
| `_read_output_files()` | Replaced by in-memory `export_to_dict()` — no disk read needed. |

### Updated Methods

| Method | Change |
|--------|--------|
| `parse_pdf()` | Now delegates to `_parse_with_python_api()` instead of `_run_docling_command()` + `_read_output_files()`. |
| `parse_office_doc()` | Same delegation to `_parse_with_python_api()`. |
| `parse_html()` | Same delegation to `_parse_with_python_api()`. |
| `check_installation()` | Now checks `_ensure_docling_imports()` (Python package availability) instead of running `docling --version` subprocess. |

---

## Backward Compatibility

### Kwargs Classification

To ensure a smooth migration for existing users, kwargs are classified into
three tiers:

```python
# MinerU-specific — silently ignored (shared call-paths may forward these)
_IGNORED_KWARGS = frozenset({
    "backend", "device", "source", "formula", "table",
    "vlm_url", "start_page", "end_page",
})

# Docling-specific — actively mapped to PdfPipelineOptions
_KNOWN_KWARGS = frozenset({
    "table_mode", "tables", "allow_ocr", "ocr_engine",
    "ocr_lang", "pdf_backend", "artifacts_path",
    "abort_on_error", "env",
})

# Accepted for backward-compat but NOT wired into the Python API
_COMPAT_ONLY_KWARGS = frozenset({
    "ocr_engine", "ocr_lang", "pdf_backend",
    "abort_on_error", "env",
})
```

- **No breaking changes** to the public `parse_pdf()`, `parse_document()`,
  `parse_office_doc()`, `parse_html()` signatures.
- Callers passing `env={"KEY": "VAL"}` (previously used for subprocess
  environment) will have the type validated but the value silently ignored.
- Unknown kwargs raise `TypeError` immediately for fail-fast debugging.

### Output Format

The content-list output format is **identical** to the current
implementation. The same `read_from_block_recursive()` /
`read_from_block()` methods are used to transform the Docling document
dict into the MinerU-compatible structure.

---

## Performance Benefits

| Metric | CLI subprocess (current) | Python API (proposed) |
|--------|--------------------------|-----------------------|
| Model loading | Every call | Once (cached converter) |
| Process creation | `subprocess.run` per call | None |
| Disk I/O | Write JSON+MD, read back | In-memory only (except images) |
| Error diagnostics | stderr strings | Full Python tracebacks |
| Pipeline configurability | CLI flag subset | Full `PdfPipelineOptions` surface |

For batch workloads parsing N documents sequentially, the Python API path
avoids N−1 redundant model loads — potentially saving **minutes** of
wall-clock time depending on model size and hardware.

---

## Test Coverage

The reference implementation includes a comprehensive test suite
(`tests/testparser_python_api.py`) with **45 test methods** covering:

- **Construction** — lazy initialization, no premature imports
- **Lazy imports** — module caching, `ImportError` when docling is missing
- **Pipeline options** — `table_mode`, `tables`, `allow_ocr`, `artifacts_path` mapping
- **Kwarg validation** — unknown kwargs rejected, MinerU kwargs silently ignored, `env` type-checking
- **Converter caching** — reuse on identical config, invalidation on config change
- **`_parse_with_python_api()`** — end-to-end flow with mocked `DocumentConverter`
- **`parse_pdf()` / `parse_office_doc()` / `parse_html()`** — delegation to core method
- **`parse_document()`** — format dispatch, unsupported format error
- **`check_installation()`** — success and failure paths

All docling imports are fully mocked, enabling the test suite to run in
CI environments without installing the `docling` package.

---

## Implementation Scope

### Files Changed

| File | Lines changed | Description |
|------|---------------|-------------|
| `raganything/parser.py` | ~620 modified | Replace CLI subprocess methods with Python API methods in `DoclingParser` |
| `tests/testparser_python_api.py` | ~525 added | New comprehensive test suite for Python API mode |
| `tests/testparser_kwargs.py` | ~120 modified | Update existing kwargs tests for new validation behavior |

### Dependencies

- **No new required dependencies.** `docling` remains an optional package.
- The `check_installation()` method gracefully reports whether the package
  is available.
- `pip install docling` is the only setup step for users who want to use
  the Docling parser backend.

---

## Migration Guide

### For End Users

No action required. The public API is unchanged:

```python
from raganything.parser import DoclingParser

parser = DoclingParser()
content = parser.parse_document("report.pdf", output_dir="./output")
```

### For Callers Passing `env`

The `env` kwarg is still accepted but no longer has any effect:

```python
# Before (CLI subprocess):
parser.parse_pdf("doc.pdf", env={"DOCLING_CACHE": "/tmp"})
# After (Python API): accepted without error, but env is ignored.
```

### For Advanced Configuration

The Python API exposes more options than the CLI:

```python
parser.parse_pdf(
    "doc.pdf",
    table_mode="accurate",      # TableFormerMode.ACCURATE
    tables=True,                # do_table_structure = True
    allow_ocr=True,             # do_ocr = True
    artifacts_path="/models",   # custom model artifacts directory
)
```

---

## Checklist

- [x] Confirmed key difference between fork and upstream is CLI→Python API
- [x] Reference implementation tested with mocked docling (CI-safe)
- [x] Backward compatibility preserved for all existing kwargs
- [x] Output format identical to current CLI-based implementation
- [x] No new required dependencies introduced
- [ ] Upstream review and approval (HKUDS/RAG-Anything)

---

## References

- Docling Python API: https://github.com/DS4SD/docling
- `DocumentConverter` usage: https://ds4sd.github.io/docling/
- Reference implementation: https://github.com/hanlianlu/RAG-Anything
  - Commit [`dca256c`](https://github.com/hanlianlu/RAG-Anything/commit/dca256c79803195ee5b2a400b18e0ce3e61f4768): Initial refactoring (docstrings, config, processor wiring)
  - Commit [`6bb76fb`](https://github.com/hanlianlu/RAG-Anything/commit/6bb76fb030bd8e3526e7292a108fac7995f2cc9b): Full CLI→Python API replacement with tests
