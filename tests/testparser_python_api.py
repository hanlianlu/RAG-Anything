#!/usr/bin/env python3
"""
Tests for DoclingParser Python API mode.

Validates the Python API path using the DocumentConverter directly.
All docling imports are mocked since docling may not be installed in CI.

Usage:
    pytest tests/testparser_python_api.py -v
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock


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

    MockPipelineModule = MagicMock()
    MockPipelineModule.PdfPipelineOptions = MockPdfPipelineOptions
    MockPipelineModule.TableFormerMode = MockTableFormerMode

    # InputFormat
    MockInputFormat = MagicMock()
    MockInputFormat.PDF = "PDF"
    MockInputFormat.DOCX = "DOCX"
    MockInputFormat.PPTX = "PPTX"
    MockInputFormat.HTML = "HTML"
    MockInputFormat.IMAGE = "IMAGE"

    MockBaseModels = MagicMock()
    MockBaseModels.InputFormat = MockInputFormat

    # DocumentConverter / PdfFormatOption
    MockDocumentConverter = MagicMock()
    MockPdfFormatOption = MagicMock()

    MockConverterModule = MagicMock()
    MockConverterModule.DocumentConverter = MockDocumentConverter
    MockConverterModule.PdfFormatOption = MockPdfFormatOption

    return {
        "pipeline_module": MockPipelineModule,
        "base_models_module": MockBaseModels,
        "converter_module": MockConverterModule,
        "DocumentConverter": MockDocumentConverter,
        "PdfFormatOption": MockPdfFormatOption,
        "PdfPipelineOptions": MockPdfPipelineOptions,
        "TableFormerMode": MockTableFormerMode,
        "InputFormat": MockInputFormat,
    }


def _install_docling_mocks(mocks):
    """Inject mock docling modules into sys.modules."""
    sys.modules.setdefault("docling", MagicMock())
    sys.modules["docling.document_converter"] = mocks["converter_module"]
    sys.modules.setdefault("docling.datamodel", MagicMock())
    sys.modules["docling.datamodel.base_models"] = mocks["base_models_module"]
    sys.modules["docling.datamodel.pipeline_options"] = mocks["pipeline_module"]


def _remove_docling_mocks():
    """Remove injected docling modules from sys.modules."""
    for key in list(sys.modules):
        if key == "docling" or key.startswith("docling."):
            del sys.modules[key]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def docling_mocks():
    """Provide docling mock modules and clean up after the test."""
    mocks = _make_docling_mocks()
    _install_docling_mocks(mocks)
    yield mocks
    _remove_docling_mocks()


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

    def test_ensure_imports_cached(self, docling_parser, docling_mocks):
        first = docling_parser._ensure_docling_imports()
        second = docling_parser._ensure_docling_imports()
        assert first is second

    def test_import_error_when_docling_missing(self):
        """When docling is not installed, _ensure_docling_imports raises ImportError."""
        _remove_docling_mocks()
        from raganything.parser import DoclingParser
        parser = DoclingParser()
        with pytest.raises(ImportError, match="docling"):
            parser._ensure_docling_imports()


# ---------------------------------------------------------------------------
# 3. Pipeline options building
# ---------------------------------------------------------------------------

class TestPipelineOptions:

    def test_default_pipeline_options(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options()
        # Should call PdfPipelineOptions constructor
        docling_mocks["PdfPipelineOptions"].assert_called()

    def test_table_mode_accurate(self, docling_parser, docling_mocks):
        opts = docling_parser._build_pipeline_options(table_mode="accurate")
        # Should set table structure mode to ACCURATE
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
        opts = docling_parser._build_pipeline_options(
            backend="pipeline",
            device="cpu",
            source="huggingface",
            formula=False,
            table=False,
            vlm_url="http://localhost",
            start_page=1,
            end_page=2,
        )


# ---------------------------------------------------------------------------
# 4. Converter management (caching & invalidation)
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
        c1 = docling_parser._get_converter(table_mode="accurate")
        c2 = docling_parser._get_converter(table_mode="fast")
        # Converter should be recreated (new instance)
        assert docling_mocks["DocumentConverter"].call_count >= 2


# ---------------------------------------------------------------------------
# 5. Routing: parse_pdf
# ---------------------------------------------------------------------------

class TestRoutingParsePdf:

    def test_uses_python_api(self, docling_parser, docling_mocks, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        # Mock the _parse_with_python_api method
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

        call_kwargs = mock_api.call_args
        assert call_kwargs is not None
        # kwargs should include table_mode and allow_ocr
        _, kw = call_kwargs
        assert kw.get("table_mode") == "accurate"
        assert kw.get("allow_ocr") is False


# ---------------------------------------------------------------------------
# 6. Routing: parse_document / parse_office_doc / parse_html
# ---------------------------------------------------------------------------

class TestRoutingParseDocument:

    def test_pdf(self, docling_parser, docling_mocks, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        with patch.object(docling_parser, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "api", "page_idx": 0}]
            result = docling_parser.parse_document(fake_pdf, output_dir=str(tmp_path))

        mock_api.assert_called_once()

    def test_office(self, docling_parser, docling_mocks, tmp_path):
        fake_docx = tmp_path / "test.docx"
        fake_docx.write_bytes(b"PK\x03\x04")

        with patch.object(docling_parser, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "office", "page_idx": 0}]
            result = docling_parser.parse_office_doc(fake_docx, output_dir=str(tmp_path))

        mock_api.assert_called_once()

    def test_html(self, docling_parser, docling_mocks, tmp_path):
        fake_html = tmp_path / "test.html"
        fake_html.write_text("<html><body>hello</body></html>")

        with patch.object(docling_parser, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "html", "page_idx": 0}]
            result = docling_parser.parse_html(fake_html, output_dir=str(tmp_path))

        mock_api.assert_called_once()


# ---------------------------------------------------------------------------
# 7. Python API content conversion
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


# ---------------------------------------------------------------------------
# 8. check_installation
# ---------------------------------------------------------------------------

class TestCheckInstallation:

    def test_returns_true_when_docling_installed(
        self, docling_parser, docling_mocks
    ):
        assert docling_parser.check_installation() is True

    def test_returns_false_when_docling_missing(self):
        _remove_docling_mocks()
        from raganything.parser import DoclingParser
        parser = DoclingParser()
        assert parser.check_installation() is False


# ---------------------------------------------------------------------------
# 9. get_parser integration
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
