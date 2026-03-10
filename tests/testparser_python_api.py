#!/usr/bin/env python3
"""
Tests for DoclingParser Python API mode.

Validates the Python API path as an alternative to the CLI subprocess approach.
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
def docling_parser_api(docling_mocks):
    """Return a DoclingParser in Python API mode."""
    from raganything.parser import DoclingParser
    return DoclingParser(use_python_api=True)


@pytest.fixture
def docling_parser_cli():
    """Return a DoclingParser in CLI mode (default)."""
    from raganything.parser import DoclingParser
    return DoclingParser()


# ---------------------------------------------------------------------------
# 1. Construction and flag propagation
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_is_cli_mode(self):
        from raganything.parser import DoclingParser
        parser = DoclingParser()
        assert parser._use_python_api is False

    def test_use_python_api_flag(self, docling_mocks):
        from raganything.parser import DoclingParser
        parser = DoclingParser(use_python_api=True)
        assert parser._use_python_api is True

    def test_converter_initially_none(self, docling_mocks):
        from raganything.parser import DoclingParser
        parser = DoclingParser(use_python_api=True)
        assert parser._converter is None


# ---------------------------------------------------------------------------
# 2. Lazy docling import
# ---------------------------------------------------------------------------

class TestLazyImport:

    def test_ensure_imports_returns_modules(self, docling_parser_api, docling_mocks):
        imports = docling_parser_api._ensure_docling_imports()
        assert "DocumentConverter" in imports
        assert "PdfFormatOption" in imports
        assert "InputFormat" in imports
        assert "PdfPipelineOptions" in imports
        assert "TableFormerMode" in imports

    def test_ensure_imports_cached(self, docling_parser_api, docling_mocks):
        first = docling_parser_api._ensure_docling_imports()
        second = docling_parser_api._ensure_docling_imports()
        assert first is second

    def test_import_error_when_docling_missing(self):
        """When docling is not installed, _ensure_docling_imports raises ImportError."""
        _remove_docling_mocks()
        from raganything.parser import DoclingParser
        parser = DoclingParser(use_python_api=True)
        with pytest.raises(ImportError, match="docling"):
            parser._ensure_docling_imports()


# ---------------------------------------------------------------------------
# 3. Pipeline options building
# ---------------------------------------------------------------------------

class TestPipelineOptions:

    def test_default_pipeline_options(self, docling_parser_api, docling_mocks):
        opts = docling_parser_api._build_pipeline_options()
        # Should call PdfPipelineOptions constructor
        docling_mocks["PdfPipelineOptions"].assert_called()

    def test_table_mode_accurate(self, docling_parser_api, docling_mocks):
        opts = docling_parser_api._build_pipeline_options(table_mode="accurate")
        # Should set table structure mode to ACCURATE
        assert opts.table_structure_options.mode == docling_mocks["TableFormerMode"].ACCURATE

    def test_table_mode_fast(self, docling_parser_api, docling_mocks):
        opts = docling_parser_api._build_pipeline_options(table_mode="fast")
        assert opts.table_structure_options.mode == docling_mocks["TableFormerMode"].FAST

    def test_tables_disabled(self, docling_parser_api, docling_mocks):
        opts = docling_parser_api._build_pipeline_options(tables=False)
        assert opts.do_table_structure is False

    def test_tables_enabled(self, docling_parser_api, docling_mocks):
        opts = docling_parser_api._build_pipeline_options(tables=True)
        assert opts.do_table_structure is True

    def test_ocr_disabled(self, docling_parser_api, docling_mocks):
        opts = docling_parser_api._build_pipeline_options(allow_ocr=False)
        assert opts.do_ocr is False

    def test_ocr_enabled(self, docling_parser_api, docling_mocks):
        opts = docling_parser_api._build_pipeline_options(allow_ocr=True)
        assert opts.do_ocr is True

    def test_artifacts_path(self, docling_parser_api, docling_mocks):
        opts = docling_parser_api._build_pipeline_options(artifacts_path="/models/docling")
        assert opts.artifacts_path == "/models/docling"

    def test_ignores_mineru_kwargs(self, docling_parser_api, docling_mocks):
        """MinerU-specific kwargs should be silently ignored."""
        # Should not raise
        opts = docling_parser_api._build_pipeline_options(
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

    def test_get_converter_creates_instance(self, docling_parser_api, docling_mocks):
        converter = docling_parser_api._get_converter()
        assert converter is not None
        assert docling_parser_api._converter is not None

    def test_converter_reused_same_kwargs(self, docling_parser_api, docling_mocks):
        c1 = docling_parser_api._get_converter(table_mode="accurate")
        c2 = docling_parser_api._get_converter(table_mode="accurate")
        assert c1 is c2

    def test_converter_recreated_different_kwargs(self, docling_parser_api, docling_mocks):
        c1 = docling_parser_api._get_converter(table_mode="accurate")
        c2 = docling_parser_api._get_converter(table_mode="fast")
        # Converter should be recreated (new instance)
        assert docling_mocks["DocumentConverter"].call_count >= 2


# ---------------------------------------------------------------------------
# 5. Routing: parse_pdf
# ---------------------------------------------------------------------------

class TestRoutingParsePdf:

    def test_api_mode_uses_python_api(self, docling_parser_api, docling_mocks, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        # Mock the _parse_with_python_api method
        with patch.object(docling_parser_api, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "hello", "page_idx": 0}]
            result = docling_parser_api.parse_pdf(fake_pdf, output_dir=str(tmp_path))

        mock_api.assert_called_once()
        assert result == [{"type": "text", "text": "hello", "page_idx": 0}]

    def test_cli_mode_uses_subprocess(self, docling_parser_cli, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        with patch.object(docling_parser_cli, '_run_docling_command') as mock_cli, \
             patch.object(docling_parser_cli, '_read_output_files') as mock_read:
            mock_read.return_value = (
                [{"type": "text", "text": "hello", "page_idx": 0}],
                "hello",
            )
            result = docling_parser_cli.parse_pdf(fake_pdf, output_dir=str(tmp_path))

        mock_cli.assert_called_once()
        assert result == [{"type": "text", "text": "hello", "page_idx": 0}]

    def test_api_mode_forwards_kwargs(self, docling_parser_api, docling_mocks, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        with patch.object(docling_parser_api, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "ok", "page_idx": 0}]
            docling_parser_api.parse_pdf(
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
# 6. Routing: parse_document
# ---------------------------------------------------------------------------

class TestRoutingParseDocument:

    def test_api_mode_pdf(self, docling_parser_api, docling_mocks, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        with patch.object(docling_parser_api, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "api", "page_idx": 0}]
            result = docling_parser_api.parse_document(fake_pdf, output_dir=str(tmp_path))

        mock_api.assert_called_once()

    def test_api_mode_office(self, docling_parser_api, docling_mocks, tmp_path):
        fake_docx = tmp_path / "test.docx"
        fake_docx.write_bytes(b"PK\x03\x04")

        with patch.object(docling_parser_api, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "office", "page_idx": 0}]
            result = docling_parser_api.parse_office_doc(fake_docx, output_dir=str(tmp_path))

        mock_api.assert_called_once()

    def test_api_mode_html(self, docling_parser_api, docling_mocks, tmp_path):
        fake_html = tmp_path / "test.html"
        fake_html.write_text("<html><body>hello</body></html>")

        with patch.object(docling_parser_api, '_parse_with_python_api') as mock_api:
            mock_api.return_value = [{"type": "text", "text": "html", "page_idx": 0}]
            result = docling_parser_api.parse_html(fake_html, output_dir=str(tmp_path))

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

    def test_simple_text_document(self, docling_parser_api, docling_mocks, tmp_path):
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

        with patch.object(docling_parser_api, '_get_converter', return_value=mock_converter):
            result = docling_parser_api._parse_with_python_api(
                fake_pdf, output_dir=str(tmp_path)
            )

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "First paragraph"
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "Second paragraph"

    def test_table_content(self, docling_parser_api, docling_mocks, tmp_path):
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

        with patch.object(docling_parser_api, '_get_converter', return_value=mock_converter):
            result = docling_parser_api._parse_with_python_api(
                fake_pdf, output_dir=str(tmp_path)
            )

        assert len(result) == 1
        assert result[0]["type"] == "table"
        assert result[0]["table_body"] == [["A", "B"], ["1", "2"]]
        assert result[0]["table_caption"] == "Test table"

    def test_formula_content(self, docling_parser_api, docling_mocks, tmp_path):
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

        with patch.object(docling_parser_api, '_get_converter', return_value=mock_converter):
            result = docling_parser_api._parse_with_python_api(
                fake_pdf, output_dir=str(tmp_path)
            )

        assert len(result) == 1
        assert result[0]["type"] == "equation"
        assert result[0]["text"] == "E = mc^2"


# ---------------------------------------------------------------------------
# 8. check_installation
# ---------------------------------------------------------------------------

class TestCheckInstallation:

    def test_api_mode_returns_true_when_docling_installed(
        self, docling_parser_api, docling_mocks
    ):
        assert docling_parser_api.check_installation() is True

    def test_api_mode_returns_false_when_docling_missing(self):
        _remove_docling_mocks()
        from raganything.parser import DoclingParser
        parser = DoclingParser(use_python_api=True)
        assert parser.check_installation() is False

    @patch("subprocess.run")
    def test_cli_mode_checks_command(self, mock_run, docling_parser_cli):
        mock_run.return_value = MagicMock(stdout="docling 2.0.0")
        assert docling_parser_cli.check_installation() is True
        cmd = mock_run.call_args.args[0]
        assert cmd == ["docling", "--version"]


# ---------------------------------------------------------------------------
# 9. get_parser integration
# ---------------------------------------------------------------------------

class TestGetParser:

    def test_get_parser_default_cli(self):
        from raganything.parser import get_parser
        parser = get_parser("docling")
        assert parser._use_python_api is False

    def test_get_parser_python_api(self, docling_mocks):
        from raganything.parser import get_parser
        parser = get_parser("docling", use_python_api=True)
        assert parser._use_python_api is True

    def test_get_parser_non_docling_ignores_flag(self):
        from raganything.parser import get_parser
        parser = get_parser("mineru", use_python_api=True)
        assert not hasattr(parser, '_use_python_api') or parser._use_python_api is False


# ---------------------------------------------------------------------------
# 10. Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:

    def test_config_has_use_python_api_field(self):
        from raganything.config import RAGAnythingConfig
        config = RAGAnythingConfig()
        assert hasattr(config, "docling_use_python_api")
        assert config.docling_use_python_api is False

    def test_config_use_python_api_true(self):
        from raganything.config import RAGAnythingConfig
        config = RAGAnythingConfig(docling_use_python_api=True)
        assert config.docling_use_python_api is True


# ---------------------------------------------------------------------------
# 11. Processor passes use_python_api to parser
# ---------------------------------------------------------------------------

class TestProcessorIntegration:

    @pytest.mark.asyncio
    async def test_processor_passes_use_python_api(self, monkeypatch, tmp_path):
        import raganything.processor as processor_module

        class FakeLogger:
            def info(self, *a, **kw): pass
            def warning(self, *a, **kw): pass
            def error(self, *a, **kw): pass
            def debug(self, *a, **kw): pass

        captured = {}

        class FakeParser:
            def __init__(self, use_python_api=False):
                captured["use_python_api"] = use_python_api
            def parse_pdf(self, **kwargs):
                return [{"type": "text", "text": "ok", "page_idx": 0}]

        def fake_get_parser(parser_name, *, use_python_api=False):
            return FakeParser(use_python_api=use_python_api)

        monkeypatch.setattr(processor_module, "get_parser", fake_get_parser)

        class DummyProcessor(processor_module.ProcessorMixin):
            pass

        dummy = DummyProcessor()
        dummy.config = type("Config", (), {
            "parser": "docling",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "display_content_stats": False,
            "use_full_path": False,
            "docling_use_python_api": True,
            "docling_table_mode": None,
            "docling_ocr_engine": None,
            "docling_artifacts_path": None,
        })()
        dummy.logger = FakeLogger()
        dummy.parse_cache = None

        monkeypatch.setattr(
            DummyProcessor, "_store_cached_result",
            lambda *a, **kw: None, raising=False,
        )
        monkeypatch.setattr(
            DummyProcessor, "_generate_content_based_doc_id",
            lambda self, cl: "doc-fixed", raising=False,
        )

        # Need an async _store_cached_result
        async def fake_store(*a, **kw):
            return None
        monkeypatch.setattr(DummyProcessor, "_store_cached_result", fake_store, raising=False)

        fake_pdf = tmp_path / "sample.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")

        await dummy.parse_document(str(fake_pdf))
        assert captured["use_python_api"] is True


# ---------------------------------------------------------------------------
# 12. Backward compatibility — existing CLI tests still work
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    @patch("subprocess.run")
    def test_cli_env_propagation_unchanged(self, mock_run, tmp_path):
        """Existing CLI env propagation still works after refactor."""
        from raganything.parser import DoclingParser
        parser = DoclingParser()  # default CLI mode
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        parser._run_docling_command(
            "dummy.pdf", str(tmp_path / "out"), "stem",
            env={"DOCLING_VAR": "value"},
        )

        _, kwargs = mock_run.call_args
        assert kwargs["env"]["DOCLING_VAR"] == "value"

    @patch("subprocess.run")
    def test_cli_table_mode_unchanged(self, mock_run, tmp_path):
        from raganything.parser import DoclingParser
        parser = DoclingParser()
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        parser._run_docling_command(
            "dummy.pdf", str(tmp_path / "out"), "stem",
            table_mode="fast",
        )

        cmd = mock_run.call_args.args[0]
        assert "--table-mode" in cmd
        assert cmd[cmd.index("--table-mode") + 1] == "fast"
