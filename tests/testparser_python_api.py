#!/usr/bin/env python3
"""
Tests for DoclingParser Python API mode.

Validates the Python API path using the DocumentConverter directly.
All docling imports are mocked since docling may not be installed in CI.

Usage:
    pytest tests/testparser_python_api.py -v
"""

import sys
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers – mock the docling package hierarchy so that lazy imports succeed
# ---------------------------------------------------------------------------

def _make_docling_mocks():
    """Return a dict of mock docling modules and key classes."""
    # Pipeline options
    MockTableFormerMode = MagicMock()
    MockTableFormerMode.ACCURATE = "accurate"
    MockTableFormerMode.FAST = "fast"

    MockPdfPipelineOptions = MagicMock()

    # OCR option classes
    MockEasyOcrOptions = MagicMock()
    MockTesseractOcrOptions = MagicMock()
    MockTesseractCliOcrOptions = MagicMock()

    MockPipelineModule = MagicMock()
    MockPipelineModule.PdfPipelineOptions = MockPdfPipelineOptions
    MockPipelineModule.TableFormerMode = MockTableFormerMode
    MockPipelineModule.EasyOcrOptions = MockEasyOcrOptions
    MockPipelineModule.TesseractOcrOptions = MockTesseractOcrOptions
    MockPipelineModule.TesseractCliOcrOptions = MockTesseractCliOcrOptions

    # InputFormat
    MockInputFormat = MagicMock()
    MockInputFormat.PDF = "PDF"

    MockBaseModels = MagicMock()
    MockBaseModels.InputFormat = MockInputFormat

    # DocumentConverter / PdfFormatOption
    MockDocumentConverter = MagicMock()
    MockPdfFormatOption = MagicMock()

    MockConverterModule = MagicMock()
    MockConverterModule.DocumentConverter = MockDocumentConverter
    MockConverterModule.PdfFormatOption = MockPdfFormatOption

    # PDF backend classes
    MockDlParseV2Backend = MagicMock()
    MockDlParseV1Backend = MagicMock()
    MockPyPdfiumBackend = MagicMock()

    return {
        "pipeline_module": MockPipelineModule,
        "base_models_module": MockBaseModels,
        "converter_module": MockConverterModule,
        "DocumentConverter": MockDocumentConverter,
        "PdfFormatOption": MockPdfFormatOption,
        "PdfPipelineOptions": MockPdfPipelineOptions,
        "TableFormerMode": MockTableFormerMode,
        "InputFormat": MockInputFormat,
        "EasyOcrOptions": MockEasyOcrOptions,
        "TesseractOcrOptions": MockTesseractOcrOptions,
        "TesseractCliOcrOptions": MockTesseractCliOcrOptions,
        "DlParseV2Backend": MockDlParseV2Backend,
        "DlParseV1Backend": MockDlParseV1Backend,
        "PyPdfiumBackend": MockPyPdfiumBackend,
    }


def _block_docling_imports(monkeypatch):
    """Use *monkeypatch* to make all ``docling.*`` imports raise ``ImportError``.

    This works regardless of whether docling is actually installed.
    Any cached ``docling`` modules are removed from ``sys.modules`` so
    the parser performs a fresh import attempt.
    """
    _real_import = (
        __builtins__.__import__
        if hasattr(__builtins__, "__import__")
        else __import__
    )

    def _block(name, *args, **kwargs):
        if name.startswith("docling"):
            raise ImportError("mocked: docling not installed")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _block)
    for key in list(sys.modules):
        if key == "docling" or key.startswith("docling."):
            monkeypatch.delitem(sys.modules, key)


def _install_docling_mocks(monkeypatch, mocks):
    """Inject mock docling modules into sys.modules via *monkeypatch*.

    Using ``monkeypatch.setitem`` ensures that the original
    ``sys.modules`` state is restored automatically at the end of each
    test — even when ``docling`` is genuinely installed.
    """
    if "docling" not in sys.modules:
        monkeypatch.setitem(sys.modules, "docling", MagicMock())
    monkeypatch.setitem(
        sys.modules, "docling.document_converter", mocks["converter_module"]
    )
    if "docling.datamodel" not in sys.modules:
        monkeypatch.setitem(sys.modules, "docling.datamodel", MagicMock())
    monkeypatch.setitem(
        sys.modules, "docling.datamodel.base_models", mocks["base_models_module"]
    )
    monkeypatch.setitem(
        sys.modules, "docling.datamodel.pipeline_options", mocks["pipeline_module"]
    )

    # PDF backend modules – each exposes a single backend class
    _mock_v2_backend_module = MagicMock()
    _mock_v2_backend_module.DoclingParseV2DocumentBackend = mocks["DlParseV2Backend"]
    _mock_v1_backend_module = MagicMock()
    _mock_v1_backend_module.DoclingParseDocumentBackend = mocks["DlParseV1Backend"]
    _mock_pypdfium_backend_module = MagicMock()
    _mock_pypdfium_backend_module.PyPdfiumDocumentBackend = mocks["PyPdfiumBackend"]

    if "docling.backend" not in sys.modules:
        monkeypatch.setitem(sys.modules, "docling.backend", MagicMock())
    monkeypatch.setitem(
        sys.modules,
        "docling.backend.docling_parse_v2_backend",
        _mock_v2_backend_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "docling.backend.docling_parse_backend",
        _mock_v1_backend_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "docling.backend.pypdfium2_backend",
        _mock_pypdfium_backend_module,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def docling_mocks(monkeypatch):
    """Provide docling mock modules; cleanup is automatic via monkeypatch."""
    mocks = _make_docling_mocks()
    _install_docling_mocks(monkeypatch, mocks)
    yield mocks


@pytest.fixture
def docling_parser(docling_mocks):
    """Return a DoclingParser instance."""
    from raganything.parser import DoclingParser
    return DoclingParser()


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_converter_initially_none(self, docling_mocks):
        from raganything.parser import DoclingParser
        parser = DoclingParser()
        assert parser._converter is None


# ---------------------------------------------------------------------------
# 2. Lazy docling import
# ---------------------------------------------------------------------------

class TestLazyImport:

    def test_ensure_imports_returns_modules(self, docling_parser, docling_mocks):
        imports = docling_parser._ensure_docling_imports()
        assert "DocumentConverter" in imports
        assert "PdfFormatOption" in imports
        assert "InputFormat" in imports
        assert "PdfPipelineOptions" in imports
        assert "TableFormerMode" in imports
        assert "EasyOcrOptions" in imports
        assert "TesseractOcrOptions" in imports
        assert "TesseractCliOcrOptions" in imports

    def test_ensure_imports_cached(self, docling_parser, docling_mocks):
        first = docling_parser._ensure_docling_imports()
        second = docling_parser._ensure_docling_imports()
        assert first is second

    def test_import_error_when_docling_missing(self, monkeypatch):
        """When docling is not installed, _ensure_docling_imports raises ImportError."""
        _block_docling_imports(monkeypatch)

        from raganything.parser import DoclingParser
        parser = DoclingParser()
        with pytest.raises(ImportError, match="docling"):
            parser._ensure_docling_imports()


# ---------------------------------------------------------------------------
# 3. Pipeline options building
# ---------------------------------------------------------------------------

class TestPipelineOptions:

    def test_default_pipeline_options(self, docling_parser, docling_mocks):
        docling_parser._build_pipeline_options()
        docling_mocks["PdfPipelineOptions"].assert_called()

    def test_table_mode_accurate(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(table_mode="accurate")
        assert opts.table_structure_options.mode == docling_mocks["TableFormerMode"].ACCURATE

    def test_table_mode_fast(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(table_mode="fast")
        assert opts.table_structure_options.mode == docling_mocks["TableFormerMode"].FAST

    def test_tables_disabled(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(tables=False)
        assert opts.do_table_structure is False

    def test_tables_enabled(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(tables=True)
        assert opts.do_table_structure is True

    def test_ocr_disabled(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(allow_ocr=False)
        assert opts.do_ocr is False

    def test_ocr_enabled(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(allow_ocr=True)
        assert opts.do_ocr is True

    def test_artifacts_path(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(artifacts_path="/models/docling")
        assert opts.artifacts_path == "/models/docling"

    def test_ignores_mineru_kwargs(self, docling_parser, docling_mocks):
        """MinerU-specific kwargs should be silently ignored."""
        # Should not raise
        docling_parser._build_pipeline_options(
            backend="pipeline",
            device="cpu",
            source="huggingface",
            formula=False,
            table=False,
            vlm_url="http://localhost",
            start_page=1,
            end_page=2,
        )

    def test_ocr_engine_easyocr(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(ocr_engine="easyocr")
        docling_mocks["EasyOcrOptions"].assert_called_once_with()
        assert opts.ocr_options == docling_mocks["EasyOcrOptions"].return_value

    def test_ocr_engine_tesseract(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(ocr_engine="tesseract")
        docling_mocks["TesseractOcrOptions"].assert_called_once_with()
        assert opts.ocr_options == docling_mocks["TesseractOcrOptions"].return_value

    def test_ocr_engine_tesseract_cli(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(ocr_engine="tesseract_cli")
        docling_mocks["TesseractCliOcrOptions"].assert_called_once_with()
        assert opts.ocr_options == docling_mocks["TesseractCliOcrOptions"].return_value

    def test_ocr_engine_invalid(self, docling_parser, docling_mocks):
        with pytest.raises(ValueError, match="Unsupported OCR engine"):
            docling_parser._build_pipeline_options(ocr_engine="invalid_engine")

    def test_ocr_lang_sets_language(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(ocr_lang="en,de")
        # When only ocr_lang is provided, defaults to easyocr engine
        docling_mocks["EasyOcrOptions"].assert_called_once_with(lang=["en", "de"])
        assert opts.ocr_options == docling_mocks["EasyOcrOptions"].return_value

    def test_ocr_engine_and_lang_combined(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(
            ocr_engine="tesseract", ocr_lang="eng,deu"
        )
        docling_mocks["TesseractOcrOptions"].assert_called_once_with(
            lang=["eng", "deu"]
        )
        assert opts.ocr_options == docling_mocks["TesseractOcrOptions"].return_value

    def test_ocr_lang_empty_string_defaults(self, docling_parser, docling_mocks):
        """An empty ocr_lang string should produce default OCR options (no lang list)."""
        opts = docling_parser._build_pipeline_options(ocr_lang="")
        docling_mocks["EasyOcrOptions"].assert_called_once_with()
        assert opts.ocr_options == docling_mocks["EasyOcrOptions"].return_value


# ---------------------------------------------------------------------------
# 4. Kwarg validation (fail-fast on typos)
# ---------------------------------------------------------------------------

class TestKwargValidation:

    def test_unknown_kwarg_raises(self, docling_parser, docling_mocks):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            docling_parser._validate_kwargs(unknown_arg="fail")

    def test_known_kwargs_accepted(self, docling_parser, docling_mocks):
        # Should not raise
        docling_parser._validate_kwargs(
            table_mode="accurate",
            tables=True,
            allow_ocr=True,
            ocr_engine="tesseract",
            ocr_lang="en",
            pdf_backend="dlparse_v1",
            artifacts_path="/models",
            abort_on_error=True,
            env={"K": "V"},
        )

    def test_ignored_mineru_kwargs_accepted(self, docling_parser, docling_mocks):
        # MinerU kwargs should not raise
        docling_parser._validate_kwargs(
            backend="pipeline",
            device="cpu",
        )

    def test_env_must_be_dict(self, docling_parser, docling_mocks):
        with pytest.raises(TypeError, match="'env' must be a dict"):
            docling_parser._validate_kwargs(env="not_a_dict")

    def test_env_values_must_be_str(self, docling_parser, docling_mocks):
        with pytest.raises(TypeError, match="str to str"):
            docling_parser._validate_kwargs(env={"K": 123})


# ---------------------------------------------------------------------------
# 5. Converter management (caching & invalidation)
# ---------------------------------------------------------------------------

class TestConverterManagement:

    def test_get_converter_creates_instance(self, docling_parser, docling_mocks):
        converter = docling_parser._get_converter()
        assert converter is not None
        assert docling_parser._converter is not None

    def test_converter_reused_same_kwargs(self, docling_parser, docling_mocks):
        c1 = docling_parser._get_converter(table_mode="accurate")
        c2 = docling_parser._get_converter(table_mode="accurate")
        assert c1 is c2

    def test_converter_recreated_different_kwargs(self, docling_parser, docling_mocks):
        docling_parser._get_converter(table_mode="accurate")
        call_count_after_first = docling_mocks["DocumentConverter"].call_count
        docling_parser._get_converter(table_mode="fast")
        assert docling_mocks["DocumentConverter"].call_count == call_count_after_first + 1

    def test_compat_only_kwargs_do_not_invalidate_cache(self, docling_parser, docling_mocks):
        """Backward-compat kwargs (abort_on_error, env) must NOT cause a cache miss."""
        docling_parser._get_converter(table_mode="accurate")
        call_count_after_first = docling_mocks["DocumentConverter"].call_count
        # Changing only compat-only kwargs should reuse the same converter
        docling_parser._get_converter(
            table_mode="accurate", abort_on_error=True, env={"K": "V"}
        )
        assert docling_mocks["DocumentConverter"].call_count == call_count_after_first

    def test_ocr_engine_invalidates_cache(self, docling_parser, docling_mocks):
        """Changing ocr_engine must create a new converter (it's now wired)."""
        docling_parser._get_converter(table_mode="accurate")
        call_count_after_first = docling_mocks["DocumentConverter"].call_count
        docling_parser._get_converter(
            table_mode="accurate", ocr_engine="tesseract"
        )
        assert docling_mocks["DocumentConverter"].call_count == call_count_after_first + 1

    def test_ocr_lang_invalidates_cache(self, docling_parser, docling_mocks):
        """Changing ocr_lang must create a new converter (it's now wired)."""
        docling_parser._get_converter(table_mode="accurate")
        call_count_after_first = docling_mocks["DocumentConverter"].call_count
        docling_parser._get_converter(
            table_mode="accurate", ocr_lang="en,de"
        )
        assert docling_mocks["DocumentConverter"].call_count == call_count_after_first + 1

    def test_pdf_backend_invalidates_cache(self, docling_parser, docling_mocks):
        """Changing pdf_backend must create a new converter (it's now wired)."""
        docling_parser._get_converter(table_mode="accurate")
        call_count_after_first = docling_mocks["DocumentConverter"].call_count
        docling_parser._get_converter(
            table_mode="accurate", pdf_backend="dlparse_v2"
        )
        assert docling_mocks["DocumentConverter"].call_count == call_count_after_first + 1


# ---------------------------------------------------------------------------
# 5b. PDF backend resolution
# ---------------------------------------------------------------------------

class TestPdfBackendResolution:

    def test_resolve_dlparse_v2(self, docling_parser, docling_mocks):
        cls = docling_parser._resolve_pdf_backend("dlparse_v2")
        assert cls is docling_mocks["DlParseV2Backend"]

    def test_resolve_dlparse_v1(self, docling_parser, docling_mocks):
        cls = docling_parser._resolve_pdf_backend("dlparse_v1")
        assert cls is docling_mocks["DlParseV1Backend"]

    def test_resolve_pypdfium2(self, docling_parser, docling_mocks):
        cls = docling_parser._resolve_pdf_backend("pypdfium2")
        assert cls is docling_mocks["PyPdfiumBackend"]

    def test_resolve_invalid_backend(self, docling_parser, docling_mocks):
        with pytest.raises(ValueError, match="Unsupported PDF backend"):
            docling_parser._resolve_pdf_backend("nonexistent")

    def test_pdf_backend_passed_to_format_option(self, docling_parser, docling_mocks):
        """pdf_backend should be passed as 'backend' to PdfFormatOption."""
        docling_parser._get_converter(pdf_backend="dlparse_v2")
        # Verify PdfFormatOption was called with the backend kwarg
        call_kwargs = docling_mocks["PdfFormatOption"].call_args
        assert call_kwargs is not None
        _, kw = call_kwargs
        assert "backend" in kw
        assert kw["backend"] is docling_mocks["DlParseV2Backend"]


# ---------------------------------------------------------------------------
# 6. Routing: parse_pdf
# ---------------------------------------------------------------------------

class TestRoutingParsePdf:

    def test_uses_python_api(self, docling_parser, docling_mocks, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        with patch.object(docling_parser, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "hello", "page_idx": 0}]
            result = docling_parser.parse_pdf(fake_pdf, output_dir=str(tmp_path))

        mock_api.assert_called_once()
        assert result == [{"type": "text", "text": "hello", "page_idx": 0}]

    def test_forwards_kwargs(self, docling_parser, docling_mocks, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        with patch.object(docling_parser, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "ok", "page_idx": 0}]
            docling_parser.parse_pdf(
                fake_pdf,
                output_dir=str(tmp_path),
                table_mode="accurate",
                allow_ocr=False,
            )

        _, kw = mock_api.call_args
        assert kw.get("table_mode") == "accurate"
        assert kw.get("allow_ocr") is False


# ---------------------------------------------------------------------------
# 7. Routing: parse_document / parse_office_doc / parse_html
# ---------------------------------------------------------------------------

class TestRoutingParseDocument:

    def test_pdf(self, docling_parser, docling_mocks, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        with patch.object(docling_parser, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "api", "page_idx": 0}]
            docling_parser.parse_document(fake_pdf, output_dir=str(tmp_path))

        mock_api.assert_called_once()

    def test_office(self, docling_parser, docling_mocks, tmp_path):
        fake_docx = tmp_path / "test.docx"
        fake_docx.write_bytes(b"PK\x03\x04")

        with patch.object(docling_parser, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "office", "page_idx": 0}]
            docling_parser.parse_office_doc(fake_docx, output_dir=str(tmp_path))

        mock_api.assert_called_once()

    def test_html(self, docling_parser, docling_mocks, tmp_path):
        fake_html = tmp_path / "test.html"
        fake_html.write_text("<html><body>hello</body></html>")

        with patch.object(docling_parser, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "html", "page_idx": 0}]
            docling_parser.parse_html(fake_html, output_dir=str(tmp_path))

        mock_api.assert_called_once()


# ---------------------------------------------------------------------------
# 8. Python API content conversion
# ---------------------------------------------------------------------------

class TestPythonAPIConversion:
    """Test that _parse_with_python_api produces correct content format."""

    def _make_mock_result(self, doc_dict):
        """Create a mock ConversionResult with export_to_dict()."""
        mock_result = MagicMock()
        mock_result.document.export_to_dict.return_value = doc_dict
        return mock_result

    def test_simple_text_document(self, docling_parser, docling_mocks, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        doc_dict = {
            "body": {
                "children": [
                    {"$ref": "#/texts/0"},
                    {"$ref": "#/texts/1"},
                ]
            },
            "texts": [
                {"label": "paragraph", "orig": "First paragraph", "children": []},
                {"label": "paragraph", "orig": "Second paragraph", "children": []},
            ],
            "pictures": [],
            "tables": [],
            "groups": [],
        }

        mock_result = self._make_mock_result(doc_dict)
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        with patch.object(docling_parser, '_get_converter', return_value=mock_converter):
            result = docling_parser._parse_with_python_api(
                fake_pdf, output_dir=str(tmp_path)
            )

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "First paragraph"
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "Second paragraph"

    def test_table_content(self, docling_parser, docling_mocks, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        doc_dict = {
            "body": {
                "children": [
                    {"$ref": "#/tables/0"},
                ]
            },
            "texts": [],
            "pictures": [],
            "tables": [
                {
                    "data": [["A", "B"], ["1", "2"]],
                    "caption": "Test table",
                    "footnote": "",
                    "children": [],
                }
            ],
            "groups": [],
        }

        mock_result = self._make_mock_result(doc_dict)
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        with patch.object(docling_parser, '_get_converter', return_value=mock_converter):
            result = docling_parser._parse_with_python_api(
                fake_pdf, output_dir=str(tmp_path)
            )

        assert len(result) == 1
        assert result[0]["type"] == "table"
        assert result[0]["table_body"] == [["A", "B"], ["1", "2"]]
        assert result[0]["table_caption"] == "Test table"

    def test_formula_content(self, docling_parser, docling_mocks, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        doc_dict = {
            "body": {
                "children": [
                    {"$ref": "#/texts/0"},
                ]
            },
            "texts": [
                {"label": "formula", "orig": "E = mc^2", "children": []},
            ],
            "pictures": [],
            "tables": [],
            "groups": [],
        }

        mock_result = self._make_mock_result(doc_dict)
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        with patch.object(docling_parser, '_get_converter', return_value=mock_converter):
            result = docling_parser._parse_with_python_api(
                fake_pdf, output_dir=str(tmp_path)
            )

        assert len(result) == 1
        assert result[0]["type"] == "equation"
        assert result[0]["text"] == "E = mc^2"

    def test_unknown_kwarg_rejected(self, docling_parser, docling_mocks, tmp_path):
        """_parse_with_python_api must reject unknown kwargs before side-effects."""
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            docling_parser._parse_with_python_api(
                fake_pdf, output_dir=str(tmp_path), typo_arg="oops"
            )


# ---------------------------------------------------------------------------
# 9. check_installation
# ---------------------------------------------------------------------------

class TestCheckInstallation:

    def test_returns_true_when_docling_installed(
        self, docling_parser, docling_mocks
    ):
        assert docling_parser.check_installation() is True

    def test_returns_false_when_docling_missing(self, monkeypatch):
        _block_docling_imports(monkeypatch)

        from raganything.parser import DoclingParser
        parser = DoclingParser()
        assert parser.check_installation() is False


# ---------------------------------------------------------------------------
# 10. get_parser integration
# ---------------------------------------------------------------------------

class TestGetParser:

    def test_get_parser_returns_docling(self, docling_mocks):
        from raganything.parser import get_parser
        parser = get_parser("docling")
        assert parser.__class__.__name__ == "DoclingParser"

    def test_get_parser_non_docling(self):
        from raganything.parser import get_parser
        parser = get_parser("mineru")
        assert parser.__class__.__name__ == "MineruParser"
