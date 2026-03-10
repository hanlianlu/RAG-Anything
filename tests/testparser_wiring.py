import pytest

from raganything.batch_parser import BatchParser


def test_batch_parser_uses_paddleocr_parser():
    batch_parser = BatchParser(
        parser_type="paddleocr",
        show_progress=False,
        skip_installation_check=True,
    )
    assert batch_parser.parser.__class__.__name__ == "PaddleOCRParser"


def test_raganything_initializes_selected_parser(monkeypatch, tmp_path):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    class StubParser:
        def check_installation(self):
            return True

    captured = {}

    def fake_get_parser(parser_name):
        captured["parser_name"] = parser_name
        return StubParser()

    monkeypatch.setattr(rag_module, "get_parser", fake_get_parser)
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_workdir"),
        parser="paddleocr",
    )
    rag = rag_module.RAGAnything(config=config)

    assert captured["parser_name"] == "paddleocr"
    assert isinstance(rag.doc_parser, StubParser)


@pytest.mark.asyncio
async def test_docling_parser_keeps_lightrag_chunker_overrides(monkeypatch, tmp_path):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    class StubParser:
        def check_installation(self):
            return True

    captured = {}

    class StubParseCache:
        async def initialize(self):
            return None

    class StubLightRAG:
        def __init__(self, **kwargs):
            captured["lightrag_kwargs"] = kwargs
            self.workspace = object()
            self.tokenizer = object()
            self.key_string_value_json_storage_cls = lambda **_: StubParseCache()

        async def initialize_storages(self):
            return None

    async def fake_initialize_pipeline_status():
        return None

    monkeypatch.setattr(rag_module, "LightRAG", StubLightRAG)
    monkeypatch.setattr(rag_module, "get_parser", lambda parser_name: StubParser())
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rag_module.RAGAnything,
        "_initialize_processors",
        lambda self: None,
        raising=False,
    )

    from lightrag.kg import shared_storage

    monkeypatch.setattr(
        shared_storage,
        "initialize_pipeline_status",
        fake_initialize_pipeline_status,
    )

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_workdir"),
        parser="docling",
    )
    rag = rag_module.RAGAnything(
        config=config,
        llm_model_func=lambda *args, **kwargs: None,
        embedding_func=lambda *args, **kwargs: None,
        lightrag_kwargs={
            "chunk_token_size": 1024,
            "chunk_overlap_token_size": 128,
            "tokenizer": "custom-tokenizer",
        },
    )

    result = await rag._ensure_lightrag_initialized()

    assert result == {"success": True}
    assert captured["lightrag_kwargs"]["working_dir"] == str(tmp_path / "rag_workdir")
    assert captured["lightrag_kwargs"]["chunk_token_size"] == 1024
    assert captured["lightrag_kwargs"]["chunk_overlap_token_size"] == 128
    assert captured["lightrag_kwargs"]["tokenizer"] == "custom-tokenizer"


@pytest.mark.asyncio
async def test_processor_parse_document_uses_selected_parser(monkeypatch, tmp_path):
    import raganything.processor as processor_module

    class FakeLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    class FakeParser:
        def parse_pdf(self, **kwargs):
            return [{"type": "text", "text": "parsed by fake parser", "page_idx": 0}]

        def parse_image(self, **kwargs):
            return [{"type": "text", "text": "image parsed", "page_idx": 0}]

        def parse_office_doc(self, **kwargs):
            return [{"type": "text", "text": "office parsed", "page_idx": 0}]

        def parse_document(self, **kwargs):
            return [{"type": "text", "text": "generic parsed", "page_idx": 0}]

    selected = {"calls": 0}

    def fake_get_parser(parser_name):
        selected["parser_name"] = parser_name
        selected["calls"] += 1
        return FakeParser()

    monkeypatch.setattr(processor_module, "get_parser", fake_get_parser)

    class DummyProcessor(processor_module.ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.config = type(
        "Config",
        (),
        {
            "parser": "paddleocr",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()
    dummy.logger = FakeLogger()
    dummy.parse_cache = None

    async def fake_store_cached_result(*args, **kwargs):
        return None

    monkeypatch.setattr(
        DummyProcessor,
        "_store_cached_result",
        fake_store_cached_result,
        raising=False,
    )
    monkeypatch.setattr(
        DummyProcessor,
        "_generate_content_based_doc_id",
        lambda self, content_list: "doc-fixed",
        raising=False,
    )

    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")

    content_list, doc_id = await dummy.parse_document(str(fake_pdf))
    content_list_2, doc_id_2 = await dummy.parse_document(str(fake_pdf))

    assert selected["parser_name"] == "paddleocr"
    assert selected["calls"] == 1
    assert doc_id == "doc-fixed"
    assert doc_id_2 == "doc-fixed"
    assert content_list == [
        {"type": "text", "text": "parsed by fake parser", "page_idx": 0}
    ]
    assert content_list_2 == [
        {"type": "text", "text": "parsed by fake parser", "page_idx": 0}
    ]


@pytest.mark.asyncio
async def test_processor_applies_docling_config_defaults_and_overrides(
    monkeypatch, tmp_path
):
    import raganything.processor as processor_module

    class FakeLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    class FakeParser:
        def __init__(self):
            self.calls = []

        def parse_pdf(self, **kwargs):
            self.calls.append(("pdf", kwargs))
            return [{"type": "text", "text": "pdf parsed", "page_idx": 0}]

        def parse_office_doc(self, **kwargs):
            self.calls.append(("office", kwargs))
            return [{"type": "text", "text": "office parsed", "page_idx": 0}]

        def parse_html(self, **kwargs):
            self.calls.append(("html", kwargs))
            return [{"type": "text", "text": "html parsed", "page_idx": 0}]

        def parse_document(self, **kwargs):
            self.calls.append(("generic", kwargs))
            return [{"type": "text", "text": "generic parsed", "page_idx": 0}]

    fake_parser = FakeParser()

    monkeypatch.setattr(processor_module, "get_parser", lambda parser_name: fake_parser)

    class DummyProcessor(processor_module.ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.config = type(
        "Config",
        (),
        {
            "parser": "docling",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "display_content_stats": False,
            "use_full_path": False,
            "docling_table_mode": "accurate",
            "docling_ocr_engine": "tesseract",
            "docling_artifacts_path": "/models/docling",
        },
    )()
    dummy.logger = FakeLogger()
    dummy.parse_cache = None

    async def fake_store_cached_result(*args, **kwargs):
        return None

    monkeypatch.setattr(
        DummyProcessor,
        "_store_cached_result",
        fake_store_cached_result,
        raising=False,
    )
    monkeypatch.setattr(
        DummyProcessor,
        "_generate_content_based_doc_id",
        lambda self, content_list: "doc-fixed",
        raising=False,
    )

    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")
    fake_docx = tmp_path / "sample.docx"
    fake_docx.write_bytes(b"docx")
    fake_html = tmp_path / "sample.html"
    fake_html.write_text("<html></html>", encoding="utf-8")

    await dummy.parse_document(str(fake_pdf))
    await dummy.parse_document(str(fake_docx))
    await dummy.parse_document(str(fake_html), table_mode="fast")

    assert fake_parser.calls[0][0] == "pdf"
    assert fake_parser.calls[0][1]["table_mode"] == "accurate"
    assert fake_parser.calls[0][1]["ocr_engine"] == "tesseract"
    assert fake_parser.calls[0][1]["artifacts_path"] == "/models/docling"

    assert fake_parser.calls[1][0] == "office"
    assert fake_parser.calls[1][1]["table_mode"] == "accurate"
    assert fake_parser.calls[1][1]["ocr_engine"] == "tesseract"
    assert fake_parser.calls[1][1]["artifacts_path"] == "/models/docling"

    assert fake_parser.calls[2][0] == "html"
    assert fake_parser.calls[2][1]["table_mode"] == "fast"
    assert fake_parser.calls[2][1]["ocr_engine"] == "tesseract"
    assert fake_parser.calls[2][1]["artifacts_path"] == "/models/docling"


@pytest.mark.asyncio
async def test_processor_docling_cache_tracks_config_default_changes(
    monkeypatch, tmp_path
):
    import raganything.processor as processor_module

    class FakeLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    class FakeParser:
        def __init__(self):
            self.calls = []

        def parse_pdf(self, **kwargs):
            self.calls.append(kwargs)
            table_mode = kwargs.get("table_mode", "missing")
            return [{"type": "text", "text": table_mode, "page_idx": 0}]

    class FakeParseCache:
        def __init__(self):
            self.data = {}

        async def get_by_id(self, key):
            return self.data.get(key)

        async def upsert(self, cache_data):
            self.data.update(cache_data)

        async def index_done_callback(self):
            return None

    fake_parser = FakeParser()

    monkeypatch.setattr(processor_module, "get_parser", lambda parser_name: fake_parser)

    class DummyProcessor(processor_module.ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.config = type(
        "Config",
        (),
        {
            "parser": "docling",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "display_content_stats": False,
            "use_full_path": False,
            "docling_table_mode": "accurate",
            "docling_ocr_engine": None,
            "docling_artifacts_path": None,
        },
    )()
    dummy.logger = FakeLogger()
    dummy.parse_cache = FakeParseCache()

    monkeypatch.setattr(
        DummyProcessor,
        "_generate_content_based_doc_id",
        lambda self, content_list: f"doc-{content_list[0]['text']}",
        raising=False,
    )

    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")

    content_list_1, doc_id_1 = await dummy.parse_document(str(fake_pdf))
    dummy.config.docling_table_mode = "fast"
    content_list_2, doc_id_2 = await dummy.parse_document(str(fake_pdf))

    assert content_list_1 == [{"type": "text", "text": "accurate", "page_idx": 0}]
    assert doc_id_1 == "doc-accurate"
    assert content_list_2 == [{"type": "text", "text": "fast", "page_idx": 0}]
    assert doc_id_2 == "doc-fast"
    assert len(fake_parser.calls) == 2
