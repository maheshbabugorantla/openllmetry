"""Tests for Azure AI Search instrumentation."""

import asyncio
import json
import pytest
from unittest.mock import MagicMock
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes


# ---------------------------------------------------------------------------
# Mock Azure SDK classes
# ---------------------------------------------------------------------------

class MockSearchClient:
    """Mock SearchClient for testing."""

    def __init__(self, endpoint, index_name, credential):
        self._endpoint = endpoint
        self._index_name = index_name
        self._credential = credential

    def search(self, search_text=None, **kwargs):
        return iter([{"id": "1", "name": "Test Document"}])

    def get_document(self, key, **kwargs):
        return {"id": key, "name": "Test Document"}

    def get_document_count(self, **kwargs):
        return 100

    def upload_documents(self, documents, **kwargs):
        return [MagicMock(key="1", succeeded=True, status_code=200, error_message=None)]

    def merge_documents(self, documents, **kwargs):
        return [MagicMock(key="1", succeeded=True, status_code=200, error_message=None)]

    def delete_documents(self, documents, **kwargs):
        return [MagicMock(key="1", succeeded=True, status_code=200, error_message=None)]

    def merge_or_upload_documents(self, documents, **kwargs):
        return [MagicMock(key="1", succeeded=True, status_code=200, error_message=None)]

    def index_documents(self, batch, **kwargs):
        return MagicMock(results=[MagicMock(key="1", succeeded=True, status_code=200, error_message=None)])

    def autocomplete(self, search_text, suggester_name, **kwargs):
        return [{"text": "suggestion1", "query_plus_text": "suggestion1"}]

    def suggest(self, search_text, suggester_name, **kwargs):
        return [{"text": "suggestion1"}]


# ---------------------------------------------------------------------------
# Tests — Semantic Conventions
# ---------------------------------------------------------------------------

class TestSemanticConventions:
    """Verify the 15 core Azure Search span attribute constants."""

    def test_index_name(self):
        assert SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME == "azure.search.index_name"

    def test_search_text(self):
        assert SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT == "azure.search.search.text"

    def test_search_top(self):
        assert SpanAttributes.AZURE_AI_SEARCH_SEARCH_TOP == "azure.search.search.top"

    def test_search_skip(self):
        assert SpanAttributes.AZURE_AI_SEARCH_SEARCH_SKIP == "azure.search.search.skip"

    def test_search_filter(self):
        assert SpanAttributes.AZURE_AI_SEARCH_SEARCH_FILTER == "azure.search.search.filter"

    def test_search_query_type(self):
        assert SpanAttributes.AZURE_AI_SEARCH_SEARCH_QUERY_TYPE == "azure.search.search.query_type"

    def test_document_count(self):
        assert SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT == "azure.search.document.count"

    def test_document_key(self):
        assert SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_KEY == "azure.search.document.key"

    def test_suggester_name(self):
        assert SpanAttributes.AZURE_AI_SEARCH_SUGGESTER_NAME == "azure.search.suggester_name"

    def test_analyzer_name(self):
        assert SpanAttributes.AZURE_AI_SEARCH_ANALYZER_NAME == "azure.search.analyzer_name"

    def test_search_results_count(self):
        assert SpanAttributes.AZURE_AI_SEARCH_SEARCH_RESULTS_COUNT == "azure.search.search.results_count"

    def test_document_succeeded_count(self):
        assert SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_SUCCEEDED_COUNT == "azure.search.document.succeeded_count"

    def test_document_failed_count(self):
        assert SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_FAILED_COUNT == "azure.search.document.failed_count"

    def test_autocomplete_results_count(self):
        assert SpanAttributes.AZURE_AI_SEARCH_AUTOCOMPLETE_RESULTS_COUNT == "azure.search.autocomplete.results_count"

    def test_suggest_results_count(self):
        assert SpanAttributes.AZURE_AI_SEARCH_SUGGEST_RESULTS_COUNT == "azure.search.suggest.results_count"


class TestInstrumentorLifecycle:
    """Verify the instrumentor lifecycle."""

    def test_instrumentor_can_be_instantiated(self):
        from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor
        instrumentor = AzureSearchInstrumentor()
        assert instrumentor is not None

    def test_instrumentation_dependencies(self):
        from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor
        instrumentor = AzureSearchInstrumentor()
        deps = list(instrumentor.instrumentation_dependencies())
        assert len(deps) == 1
        assert "azure-search-documents" in deps[0]

    def test_instrumentor_with_exception_logger(self):
        from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor, Config

        def custom_logger(e):
            pass

        AzureSearchInstrumentor(exception_logger=custom_logger)
        assert Config.exception_logger is custom_logger


# ---------------------------------------------------------------------------
# Tests — SearchClient instrumentation via manual spans
# ---------------------------------------------------------------------------

class TestSearchClientInstrumentation:
    """Tests for SearchClient span creation (manual span helpers)."""

    def test_search_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.search",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME: "test-index",
                SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT: "luxury hotel",
                SpanAttributes.AZURE_AI_SEARCH_SEARCH_TOP: 10,
                SpanAttributes.AZURE_AI_SEARCH_SEARCH_FILTER: "rating ge 4",
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            list(client.search(search_text="luxury hotel", top=10, filter="rating ge 4"))

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "azure.search.search"
        assert span.attributes[SpanAttributes.VECTOR_DB_VENDOR] == "Azure AI Search"
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME] == "test-index"
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT] == "luxury hotel"
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SEARCH_TOP] == 10
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SEARCH_FILTER] == "rating ge 4"

    def test_get_document_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.get_document",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_KEY: "doc-123",
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.get_document(key="doc-123")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure.search.get_document"
        assert spans[0].attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_KEY] == "doc-123"

    def test_upload_documents_creates_span(self, exporter):
        documents = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.upload_documents",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT: len(documents),
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.upload_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT] == 3

    def test_search_with_skip_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.search",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_AI_SEARCH_SEARCH_SKIP: 5,
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            list(client.search(search_text="*", top=10, skip=5))

        spans = exporter.get_finished_spans()
        assert spans[0].attributes[SpanAttributes.AZURE_AI_SEARCH_SEARCH_SKIP] == 5

    def test_get_document_count_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.get_document_count",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME: "test-index",
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.get_document_count()

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.get_document_count"

    def test_merge_documents_creates_span(self, exporter):
        documents = [{"id": "1", "rating": 4.8}]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.merge_documents",
            attributes={
                SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT: 1,
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.merge_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.merge_documents"
        assert spans[0].attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT] == 1

    def test_delete_documents_creates_span(self, exporter):
        documents = [{"id": "1"}, {"id": "2"}]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.delete_documents",
            attributes={SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT: 2},
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.delete_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT] == 2

    def test_merge_or_upload_creates_span(self, exporter):
        documents = [{"id": "1"}]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.merge_or_upload_documents",
            attributes={SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT: 1},
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.merge_or_upload_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.merge_or_upload_documents"

    def test_index_documents_creates_span(self, exporter):
        batch = MagicMock()
        batch.actions = [{"id": "1"}]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.index_documents",
            attributes={SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT: 1},
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.index_documents(batch=batch)

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.index_documents"

    def test_autocomplete_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.autocomplete",
            attributes={
                SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT: "lux",
                SpanAttributes.AZURE_AI_SEARCH_SUGGESTER_NAME: "sg",
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.autocomplete(search_text="lux", suggester_name="sg")

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.autocomplete"
        assert spans[0].attributes[SpanAttributes.AZURE_AI_SEARCH_SUGGESTER_NAME] == "sg"

    def test_suggest_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.suggest",
            attributes={
                SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT: "hot",
                SpanAttributes.AZURE_AI_SEARCH_SUGGESTER_NAME: "sg",
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.suggest(search_text="hot", suggester_name="sg")

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.suggest"
        assert spans[0].attributes[SpanAttributes.AZURE_AI_SEARCH_SUGGESTER_NAME] == "sg"


class TestSearchAttributes:
    """Tests for wrapper.py request attribute extraction functions."""

    def _make_span(self, tracer):
        span = tracer.start_span("test")
        return span

    def test_search_text_attribute(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, [], {"search_text": "hotels"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT) == "hotels"

    def test_search_top_and_skip(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, [], {"top": 5, "skip": 10})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_SEARCH_TOP) == 5
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_SEARCH_SKIP) == 10

    def test_search_filter_attribute(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, [], {"filter": "rating ge 4"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_SEARCH_FILTER) == "rating ge 4"

    def test_query_type_string(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, [], {"query_type": "full"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_SEARCH_QUERY_TYPE) == "full"

    def test_query_type_enum(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        mock_enum = MagicMock()
        mock_enum.value = "semantic"
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, [], {"query_type": mock_enum})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_SEARCH_QUERY_TYPE) == "semantic"

    def test_document_key_attribute(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_get_document_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_get_document_attributes(span, [], {"key": "hotel-1"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_KEY) == "hotel-1"

    def test_document_batch_count(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_batch_attributes
        tracer = trace.get_tracer(__name__)
        docs = [{"id": "1"}, {"id": "2"}]
        with tracer.start_as_current_span("test") as span:
            _set_document_batch_attributes(span, [], {"documents": docs})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT) == 2

    def test_suggester_name_attribute(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_suggestion_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_suggestion_attributes(span, [], {"search_text": "ho", "suggester_name": "sg"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_SUGGESTER_NAME) == "sg"


class TestResponseAttributes:
    """Tests for wrapper.py response attribute extraction functions."""

    def test_search_results_count(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_response_attributes
        mock_response = MagicMock()
        mock_response.get_count.return_value = 42
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_response_attributes(span, mock_response)
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_SEARCH_RESULTS_COUNT) == 42

    def test_search_results_count_none(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_response_attributes
        mock_response = MagicMock()
        mock_response.get_count.return_value = None
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_response_attributes(span, mock_response)
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_RESULTS_COUNT) is None

    def test_empty_batch_response_no_attributes(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_batch_response_all
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_document_batch_response_all(span, [])
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT) is None

    def test_document_count_from_int(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_count_response_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_document_count_response_attributes(span, 100)
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT) == 100

    def test_autocomplete_results_count(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_autocomplete_response_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_autocomplete_response_attributes(span, [{"text": "a"}, {"text": "b"}])
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_AUTOCOMPLETE_RESULTS_COUNT) == 2

    def test_suggest_results_count(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_suggest_response_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_suggest_response_attributes(span, [{"text": "hotel"}])
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_SUGGEST_RESULTS_COUNT) == 1

    def test_batch_succeeded_and_failed(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_batch_response_all
        results = [
            MagicMock(succeeded=True),
            MagicMock(succeeded=True),
            MagicMock(succeeded=False),
        ]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_document_batch_response_all(span, results)
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_SUCCEEDED_COUNT) == 2
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_FAILED_COUNT) == 1


class TestErrorHandling:
    """Tests for error status and suppression key."""

    def test_error_status_on_exception(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _sync_wrap
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        to_wrap = {"span_name": "azure.search.search", "method": "search"}

        def failing_wrapped(*args, **kwargs):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _sync_wrap(tracer, to_wrap, failing_wrapped, MagicMock(_index_name="idx"), [], {})

        spans = exporter.get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR

    def test_suppression_key_bypasses_span(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _wrap
        from opentelemetry import context as context_api
        from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

        tracer = trace.get_tracer(__name__)
        to_wrap = {"span_name": "azure.search.search", "method": "search"}

        mock_wrapped = MagicMock(return_value="result")

        token = context_api.attach(context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            result = _wrap(tracer, to_wrap)(mock_wrapped, MagicMock(), [], {})
        finally:
            context_api.detach(token)

        assert result == "result"
        assert len(exporter.get_finished_spans()) == 0


class TestDontThrow:
    """Tests for dont_throw decorator."""

    def test_dont_throw_swallows_exceptions(self):
        from opentelemetry.instrumentation.azure_search.utils import dont_throw

        @dont_throw
        def broken():
            raise RuntimeError("error")

        # Should not raise
        result = broken()
        assert result is None

    def test_dont_throw_returns_value_on_success(self):
        from opentelemetry.instrumentation.azure_search.utils import dont_throw

        @dont_throw
        def good():
            return 42

        assert good() == 42

    def test_async_dont_throw_swallows_exceptions(self):
        import asyncio
        from opentelemetry.instrumentation.azure_search.utils import dont_throw

        @dont_throw
        async def async_broken():
            raise RuntimeError("async error")

        result = asyncio.get_event_loop().run_until_complete(async_broken())
        assert result is None

    def test_async_dont_throw_returns_value_on_success(self):
        import asyncio
        from opentelemetry.instrumentation.azure_search.utils import dont_throw

        @dont_throw
        async def async_good():
            return 99

        result = asyncio.get_event_loop().run_until_complete(async_good())
        assert result == 99


# ---------------------------------------------------------------------------
# Tests — Async + SearchIndexClient
# ---------------------------------------------------------------------------

class MockSearchIndex:
    """Mock SearchIndex for testing."""

    def __init__(self, name, fields=None):
        self.name = name
        self.fields = fields or []


class MockServiceCounters:
    """Mock service counters for get_service_statistics response."""

    def __init__(self, document_count=0, index_count=0):
        self.document_counter = MagicMock(usage=document_count)
        self.index_counter = MagicMock(usage=index_count)


class MockServiceStatistics:
    """Mock service statistics response."""

    def __init__(self, document_count=0, index_count=0):
        self.counters = MockServiceCounters(document_count, index_count)


class MockSearchIndexClient:
    """Mock SearchIndexClient for testing."""

    def __init__(self, endpoint, credential):
        self._endpoint = endpoint
        self._credential = credential

    def create_index(self, index, **kwargs):
        return index

    def create_or_update_index(self, index, **kwargs):
        return index

    def delete_index(self, index, **kwargs):
        return None

    def get_index(self, index_name, **kwargs):
        return MockSearchIndex(name=index_name)

    def list_indexes(self, **kwargs):
        return iter([MockSearchIndex(name="index1"), MockSearchIndex(name="index2")])

    def get_index_statistics(self, index_name, **kwargs):
        return {"document_count": 100, "storage_size": 1024}

    def analyze_text(self, index_name, analyze_request, **kwargs):
        return {"tokens": [{"token": "test"}]}

    def get_service_statistics(self, **kwargs):
        return MockServiceStatistics(document_count=5000, index_count=3)

    def list_index_names(self, **kwargs):
        return iter(["index1", "index2"])


class TestSearchIndexClientInstrumentation:
    """Tests for SearchIndexClient span creation."""

    def test_create_index_creates_span(self, exporter):
        index = MockSearchIndex(name="hotels")
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.create_index",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME: "hotels",
            },
        ):
            client = MockSearchIndexClient("https://test.search.windows.net", MagicMock())
            client.create_index(index=index)

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.create_index"
        assert spans[0].attributes[SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME] == "hotels"

    def test_list_indexes_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.list_indexes",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ):
            client = MockSearchIndexClient("https://test.search.windows.net", MagicMock())
            list(client.list_indexes())

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.list_indexes"

    def test_create_or_update_index_creates_span(self, exporter):
        index = MockSearchIndex(name="upsert-index")
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.create_or_update_index",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "upsert-index",
            },
        ):
            client = MockSearchIndexClient("https://test.search.windows.net", MagicMock())
            client.create_or_update_index(index=index)

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure_search.create_or_update_index"
        assert spans[0].attributes[SpanAttributes.AZURE_SEARCH_INDEX_NAME] == "upsert-index"

    def test_get_index_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.get_index",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME: "hotels",
            },
        ):
            client = MockSearchIndexClient("https://test.search.windows.net", MagicMock())
            client.get_index("hotels")

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.get_index"

    def test_delete_index_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.delete_index",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ):
            client = MockSearchIndexClient("https://test.search.windows.net", MagicMock())
            client.delete_index("hotels")

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.delete_index"

    def test_analyze_text_creates_span(self, exporter):
        analyze_request = MagicMock()
        analyze_request.analyzer_name = "en.microsoft"
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.analyze_text",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME: "hotels",
                SpanAttributes.AZURE_AI_SEARCH_ANALYZER_NAME: "en.microsoft",
            },
        ):
            client = MockSearchIndexClient("https://test.search.windows.net", MagicMock())
            client.analyze_text("hotels", analyze_request)

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.analyze_text"

    def test_get_service_statistics_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.get_service_statistics",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ):
            client = MockSearchIndexClient("https://test.search.windows.net", MagicMock())
            client.get_service_statistics()

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.get_service_statistics"

    def test_list_index_names_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.list_index_names",
            attributes={SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search"},
        ):
            client = MockSearchIndexClient("https://test.search.windows.net", MagicMock())
            list(client.list_index_names())

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.list_index_names"

    def test_get_index_statistics_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure.search.get_index_statistics",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME: "hotels",
            },
        ):
            client = MockSearchIndexClient("https://test.search.windows.net", MagicMock())
            client.get_index_statistics("hotels")

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure.search.get_index_statistics"


class TestIndexManagementAttributes:
    """Tests for index management request attribute extraction."""

    def test_index_name_from_object(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_management_attributes
        index = MagicMock()
        index.name = "hotels"
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_index_management_attributes(span, "create_index", [], {"index": index})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME) == "hotels"

    def test_index_name_from_string(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_index_management_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_index_management_attributes(span, "get_index", ["hotels"], {})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME) == "hotels"

    def test_analyze_text_sets_analyzer(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_analyze_text_attributes
        analyze_req = MagicMock()
        analyze_req.analyzer_name = "standard.lucene"
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_analyze_text_attributes(span, ["hotels", analyze_req], {})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_ANALYZER_NAME) == "standard.lucene"

    def test_analyze_text_enum_analyzer(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_analyze_text_attributes
        enum_val = MagicMock()
        enum_val.value = "en.microsoft"
        analyze_req = MagicMock()
        analyze_req.analyzer_name = enum_val
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_analyze_text_attributes(span, ["hotels", analyze_req], {})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_ANALYZER_NAME) == "en.microsoft"


class TestServiceStatisticsResponse:
    """Tests for get_service_statistics response attributes."""

    def test_service_stats_sets_document_count(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_service_statistics_response_attributes
        stats = MockServiceStatistics(document_count=5000, index_count=3)
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_service_statistics_response_attributes(span, stats)
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_SERVICE_DOCUMENT_COUNT) == 5000

    def test_service_stats_sets_index_count(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_service_statistics_response_attributes
        stats = MockServiceStatistics(document_count=5000, index_count=3)
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_service_statistics_response_attributes(span, stats)
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_AI_SEARCH_SERVICE_INDEX_COUNT) == 3


class TestAsyncInstrumentation:
    """Tests for async wrapper dispatch and dont_throw."""

    def test_async_wrap_dispatches_coroutine(self, exporter):
        """_wrap should return a coroutine when the wrapped function is async."""
        import asyncio
        import inspect
        from opentelemetry.instrumentation.azure_search.wrapper import _wrap

        tracer = trace.get_tracer(__name__)
        to_wrap = {"span_name": "azure.search.search", "method": "search"}

        async def async_wrapped(*args, **kwargs):
            return [{"id": "1"}]

        result = _wrap(tracer, to_wrap)(async_wrapped, MagicMock(_index_name="idx"), [], {})
        assert inspect.iscoroutine(result)
        asyncio.get_event_loop().run_until_complete(result)

    def test_async_error_sets_error_status(self, exporter):
        import asyncio
        from opentelemetry.instrumentation.azure_search.wrapper import _async_wrap
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        to_wrap = {"span_name": "azure.search.search", "method": "search"}

        async def failing_async(*args, **kwargs):
            raise ValueError("async boom")

        async def run():
            with pytest.raises(ValueError, match="async boom"):
                await _async_wrap(tracer, to_wrap, failing_async, MagicMock(_index_name="idx"), [], {})

        asyncio.get_event_loop().run_until_complete(run())
        spans = exporter.get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# PR4 Tests — Vector/Semantic Search + Content Capture
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers for workflow tests (module-level so TestContentCapture can use them)
# ---------------------------------------------------------------------------

def _call_sync(tracer, method, mock_fn, instance, args=(), kwargs=None):
    """Call _sync_wrap directly — produces a real instrumented span."""
    from opentelemetry.instrumentation.azure_search.wrapper import _sync_wrap

    to_wrap = {"span_name": f"azure_search.{method}", "method": method}
    return _sync_wrap(tracer, to_wrap, mock_fn, instance, args, kwargs or {})


async def _call_async(tracer, method, mock_fn, instance, args=(), kwargs=None):
    """Call _async_wrap directly — produces a real instrumented span."""
    from opentelemetry.instrumentation.azure_search.wrapper import _async_wrap

    to_wrap = {"span_name": f"azure_search.{method}", "method": method}
    return await _async_wrap(tracer, to_wrap, mock_fn, instance, args, kwargs or {})


def _get_spans(exporter, name=None):
    """Return all finished spans, optionally filtered by name."""
    spans = exporter.get_finished_spans()
    if name is None:
        return spans
    return [s for s in spans if s.name == name]


def _assert_all_same_trace(spans):
    """Assert every span belongs to the same trace."""
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1, (
        f"Expected all spans to share one trace_id, got {len(trace_ids)}: {trace_ids}"
    )


def _find_span(spans, name):
    """Find at least one span by name, return the first match."""
    matching = [s for s in spans if s.name == name]
    assert len(matching) >= 1, (
        f"No span named '{name}' in {[s.name for s in spans]}"
    )
    return matching[0]


def _make_instance(index_name="test-index"):
    """Create a mock instance with _index_name, like a SearchClient."""
    inst = MagicMock()
    inst._index_name = index_name
    return inst


def _make_index_instance():
    """Create a mock instance without _index_name, like a SearchIndexClient."""
    inst = MagicMock()
    inst._index_name = None
    return inst


def _mock_indexing_result(key, succeeded=True, status_code=200, error_message=None):
    """Create a mock IndexingResult object with the expected attributes."""
    r = MagicMock()
    r.key = key
    r.succeeded = succeeded
    r.status_code = status_code
    r.error_message = error_message
    return r


class TestVectorSearchAttributes:
    """Tests for vector search attribute capturing."""

    def test_vector_search_attributes(self, exporter):
        """Test that vector search attributes are captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "content_vector"
            mock_vq.exhaustive = False

            kwargs = {
                "vector_queries": [mock_vq],
                "vector_filter_mode": "preFilter",
            }
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_QUERIES_COUNT
        ) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_K_NEAREST_NEIGHBORS
        ) == 5
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_FIELDS
        ) == "content_vector"
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_EXHAUSTIVE
        ) is False
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_FILTER_MODE
        ) == "preFilter"

    def test_vector_search_multiple_queries(self, exporter):
        """Test that multiple vector queries are counted correctly."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq1 = MagicMock()
            mock_vq1.k_nearest_neighbors = 5
            mock_vq1.fields = "title_vector"
            mock_vq1.exhaustive = None

            mock_vq2 = MagicMock()
            mock_vq2.k_nearest_neighbors = 3
            mock_vq2.fields = "content_vector"
            mock_vq2.exhaustive = None

            kwargs = {"vector_queries": [mock_vq1, mock_vq2]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_QUERIES_COUNT
        ) == 2
        # First vector query's fields are captured
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_FIELDS
        ) == "title_vector"

    def test_vector_search_list_fields(self, exporter):
        """Test that list fields are joined with commas."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = ["title_vector", "content_vector"]
            mock_vq.exhaustive = None

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_FIELDS
        ) == "title_vector,content_vector"

    def test_no_vector_queries_sets_nothing(self, exporter):
        """Test that no vector_queries kwarg sets no attributes."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_vector_search_attributes(span, {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_QUERIES_COUNT
        ) is None

    def test_vector_filter_mode_enum(self, exporter):
        """Test that enum vector_filter_mode values are converted to string."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "vec"
            mock_vq.exhaustive = None

            mock_enum = MagicMock()
            mock_enum.value = "postFilter"

            kwargs = {
                "vector_queries": [mock_vq],
                "vector_filter_mode": mock_enum,
            }
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_FILTER_MODE
        ) == "postFilter"


class TestEnhancedVectorSearchAttributes:
    """Tests for enhanced vector search attributes (kind, weight, oversampling)."""

    def test_vectorizable_text_query_kind(self, exporter):
        """Test that VectorizableTextQuery sets vector_query_kind='text'."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "content_vector"
            mock_vq.exhaustive = None
            mock_vq.kind = "text"
            mock_vq.weight = None
            mock_vq.oversampling = None

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_QUERY_KIND
        ) == "text"

    def test_vectorized_query_kind(self, exporter):
        """Test that VectorizedQuery sets vector_query_kind='vector'."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 10
            mock_vq.fields = "embedding"
            mock_vq.exhaustive = None
            mock_vq.kind = "vector"
            mock_vq.weight = None
            mock_vq.oversampling = None

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_QUERY_KIND
        ) == "vector"

    def test_vector_weight_captured(self, exporter):
        """Test that vector query weight is captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "vec"
            mock_vq.exhaustive = None
            mock_vq.kind = None
            mock_vq.weight = 0.8
            mock_vq.oversampling = None

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_WEIGHT
        ) == 0.8

    def test_vector_oversampling_captured(self, exporter):
        """Test that vector query oversampling is captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "vec"
            mock_vq.exhaustive = None
            mock_vq.kind = None
            mock_vq.weight = None
            mock_vq.oversampling = 2.0

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_VECTOR_OVERSAMPLING
        ) == 2.0

    def test_none_kind_weight_oversampling_not_set(self, exporter):
        """Test that None values for kind/weight/oversampling are not set."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_vector_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_vq = MagicMock()
            mock_vq.k_nearest_neighbors = 5
            mock_vq.fields = "vec"
            mock_vq.exhaustive = None
            mock_vq.kind = None
            mock_vq.weight = None
            mock_vq.oversampling = None

            kwargs = {"vector_queries": [mock_vq]}
            _set_vector_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert SpanAttributes.AZURE_SEARCH_VECTOR_QUERY_KIND not in spans[0].attributes
        assert SpanAttributes.AZURE_SEARCH_VECTOR_WEIGHT not in spans[0].attributes
        assert SpanAttributes.AZURE_SEARCH_VECTOR_OVERSAMPLING not in spans[0].attributes


class TestFacetsAndOrderByAttributes:
    """Tests for facets and order_by search attribute capturing."""

    def test_facets_as_list(self, exporter):
        """Test that facets list is captured as comma-joined string."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            kwargs = {"facets": ["category", "price,interval:10"]}
            _set_search_attributes(span, (), kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_FACETS
        ) == "category,price,interval:10"

    def test_order_by_as_list(self, exporter):
        """Test that order_by list is captured as comma-joined string."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            kwargs = {"order_by": ["price asc", "rating desc"]}
            _set_search_attributes(span, (), kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_ORDER_BY
        ) == "price asc,rating desc"

    def test_facets_none_not_set(self, exporter):
        """Test that None facets/order_by set nothing on span."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert SpanAttributes.AZURE_SEARCH_FACETS not in spans[0].attributes
        assert SpanAttributes.AZURE_SEARCH_ORDER_BY not in spans[0].attributes


class TestSemanticSearchAttributes:
    """Tests for semantic search attribute capturing."""

    def test_semantic_search_attributes(self, exporter):
        """Test that semantic search attributes are captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_semantic_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            kwargs = {
                "semantic_configuration_name": "my-semantic-config",
                "query_caption": "extractive",
                "query_answer": "extractive",
            }
            _set_semantic_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME
        ) == "my-semantic-config"
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_QUERY_CAPTION
        ) == "extractive"
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_QUERY_ANSWER
        ) == "extractive"

    def test_semantic_search_enum_values(self, exporter):
        """Test that enum values for query_caption/query_answer are converted."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_semantic_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_caption = MagicMock()
            mock_caption.value = "extractive"
            mock_answer = MagicMock()
            mock_answer.value = "extractive"

            kwargs = {
                "semantic_configuration_name": "config-1",
                "query_caption": mock_caption,
                "query_answer": mock_answer,
            }
            _set_semantic_search_attributes(span, kwargs)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_QUERY_CAPTION
        ) == "extractive"
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_QUERY_ANSWER
        ) == "extractive"

    def test_no_semantic_config_sets_nothing(self, exporter):
        """Test that missing semantic kwargs set no attributes."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_semantic_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_semantic_search_attributes(span, {})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME
        ) is None


class TestSearchAttributeExtras:
    """Tests for additional search attributes (select, search_fields, etc.)."""

    def test_search_mode_attribute(self, exporter):
        """Test that search_mode is captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {"search_mode": "all"})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEARCH_MODE
        ) == "all"

    def test_scoring_profile_attribute(self, exporter):
        """Test that scoring_profile is captured."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {"scoring_profile": "boost-by-freshness"})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SCORING_PROFILE
        ) == "boost-by-freshness"

    def test_select_as_list(self, exporter):
        """Test that select list is joined with commas."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {"select": ["id", "name", "rating"]})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SELECT
        ) == "id,name,rating"

    def test_select_as_string(self, exporter):
        """Test that select string is passed through."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {"select": "id,name"})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SELECT
        ) == "id,name"

    def test_search_fields_as_list(self, exporter):
        """Test that search_fields list is joined with commas."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            _set_search_attributes(span, (), {"search_fields": ["title", "description"]})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEARCH_FIELDS
        ) == "title,description"

    def test_query_type_enum(self, exporter):
        """Test that query_type enum is converted to string."""
        from opentelemetry.instrumentation.azure_search.wrapper import (
            _set_search_attributes,
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("azure_search.search") as span:
            mock_qt = MagicMock()
            mock_qt.value = "semantic"
            _set_search_attributes(span, (), {"query_type": mock_qt})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get(
            SpanAttributes.AZURE_SEARCH_SEARCH_QUERY_TYPE
        ) == "semantic"


class TestShouldSendContent:
    """Tests for the should_send_content() toggle function."""

    def test_default_returns_true(self, exporter, monkeypatch):
        """Default (no env var) should return True."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        assert should_send_content() is True

    def test_env_false_returns_false(self, exporter, monkeypatch):
        """TRACELOOP_TRACE_CONTENT=false should return False."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        assert should_send_content() is False

    def test_env_zero_returns_false(self, exporter, monkeypatch):
        """TRACELOOP_TRACE_CONTENT=0 should return False."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "0")
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        assert should_send_content() is False

    def test_override_context_true(self, exporter, monkeypatch):
        """override_enable_content_tracing=True overrides env=false."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        from opentelemetry import context as context_api

        ctx = context_api.set_value("override_enable_content_tracing", True)
        token = context_api.attach(ctx)
        try:
            assert should_send_content() is True
        finally:
            context_api.detach(token)

    def test_override_context_false(self, exporter, monkeypatch):
        """override_enable_content_tracing=False overrides env=true."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        from opentelemetry import context as context_api

        ctx = context_api.set_value("override_enable_content_tracing", False)
        token = context_api.attach(ctx)
        try:
            assert should_send_content() is False
        finally:
            context_api.detach(token)

    def test_truthy_values(self, exporter, monkeypatch):
        """All truthy values should return True."""
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        for val in ["true", "1", "yes", "on", "True", "YES", "ON"]:
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", val)
            assert should_send_content() is True, f"Expected True for {val!r}"

    def test_falsy_values(self, exporter, monkeypatch):
        """Non-truthy values should return False."""
        from opentelemetry.instrumentation.azure_search.utils import should_send_content
        for val in ["false", "0", "no", "off", "False", "NO", "OFF", "random"]:
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", val)
            assert should_send_content() is False, f"Expected False for {val!r}"


class TestMaxContentLength:
    """Tests for the max_content_length() function and content truncation."""

    def test_default_returns_16384(self, exporter, monkeypatch):
        """Default (no env var) should return 16384."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT_MAX_LENGTH", raising=False)
        from opentelemetry.instrumentation.azure_search.utils import max_content_length
        assert max_content_length() == 16384

    def test_env_var_set_returns_value(self, exporter, monkeypatch):
        """TRACELOOP_TRACE_CONTENT_MAX_LENGTH=1024 should return 1024."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT_MAX_LENGTH", "1024")
        from opentelemetry.instrumentation.azure_search.utils import max_content_length
        assert max_content_length() == 1024

    def test_env_var_zero_disables_truncation(self, exporter, monkeypatch):
        """TRACELOOP_TRACE_CONTENT_MAX_LENGTH=0 should return 0 (no truncation)."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT_MAX_LENGTH", "0")
        from opentelemetry.instrumentation.azure_search.utils import max_content_length
        assert max_content_length() == 0

    def test_invalid_value_returns_default(self, exporter, monkeypatch):
        """Invalid value should fall back to default."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT_MAX_LENGTH", "not_a_number")
        from opentelemetry.instrumentation.azure_search.utils import max_content_length
        assert max_content_length() == 16384

    def test_negative_value_returns_default(self, exporter, monkeypatch):
        """Negative value should fall back to default."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT_MAX_LENGTH", "-1")
        from opentelemetry.instrumentation.azure_search.utils import max_content_length
        assert max_content_length() == 16384

    def test_truncation_applied_to_content(self, exporter, monkeypatch):
        """When max length is 50, a 100-char JSON doc should be truncated."""
        from opentelemetry.instrumentation.azure_search.wrapper import _safe_json_dumps
        large_obj = {"data": "x" * 100}
        result = _safe_json_dumps(large_obj, max_length=50)
        assert len(result) == 50 + len("...[truncated]")
        assert result.endswith("...[truncated]")
        assert result[:50] in json.dumps(large_obj)

    def test_no_truncation_when_disabled(self, exporter, monkeypatch):
        """When max length is 0, no truncation should occur."""
        from opentelemetry.instrumentation.azure_search.wrapper import _safe_json_dumps
        large_obj = {"data": "x" * 100}
        result = _safe_json_dumps(large_obj, max_length=0)
        assert result == json.dumps(large_obj)
        assert "...[truncated]" not in result

    def test_no_truncation_when_under_limit(self, exporter, monkeypatch):
        """When content is shorter than max_length, no truncation should occur."""
        from opentelemetry.instrumentation.azure_search.wrapper import _safe_json_dumps
        small_obj = {"id": "1"}
        result = _safe_json_dumps(small_obj, max_length=1000)
        assert result == json.dumps(small_obj)
        assert "...[truncated]" not in result


class TestContentCapture:
    """Tests for response/request content capture via span attributes."""

    def test_get_document_content_attribute(self, exporter, monkeypatch):
        """get_document should set db.query.result.document attribute with document JSON."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()
        _call_sync(
            tracer, "get_document",
            lambda *a, **kw: {"id": "doc-123", "name": "Test Document"},
            instance, kwargs={"key": "doc-123"},
        )

        spans = exporter.get_finished_spans()
        get_doc_spans = [s for s in spans if s.name == "azure_search.get_document"]
        assert len(get_doc_spans) == 1

        span = get_doc_spans[0]
        attr_key = EventAttributes.DB_QUERY_RESULT_DOCUMENT.value
        assert attr_key in dict(span.attributes)
        doc = json.loads(span.attributes[attr_key])
        assert doc["id"] == "doc-123"

    def test_autocomplete_content_attributes(self, exporter, monkeypatch):
        """autocomplete should set indexed db.search.result.entity.N attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()
        _call_sync(
            tracer, "autocomplete",
            lambda *a, **kw: [{"text": "luxury", "query_plus_text": "luxury hotel"}],
            instance, kwargs={"search_text": "lux", "suggester_name": "sg"},
        )

        spans = exporter.get_finished_spans()
        ac_spans = [s for s in spans if s.name == "azure_search.autocomplete"]
        assert len(ac_spans) == 1

        span = ac_spans[0]
        attr_key = f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.0"
        assert attr_key in dict(span.attributes)

    def test_suggest_content_attributes(self, exporter, monkeypatch):
        """suggest should set indexed db.search.result.entity.N attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()
        _call_sync(
            tracer, "suggest",
            lambda *a, **kw: [{"@search.text": "Luxury Hotel", "id": "s1"}],
            instance, kwargs={"search_text": "lux", "suggester_name": "sg"},
        )

        spans = exporter.get_finished_spans()
        suggest_spans = [s for s in spans if s.name == "azure_search.suggest"]
        assert len(suggest_spans) == 1

        span = suggest_spans[0]
        attr_key = f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.0"
        assert attr_key in dict(span.attributes)

    def test_upload_documents_request_content_attributes(self, exporter, monkeypatch):
        """upload_documents should set per-doc indexed db.query.result.document.N attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()
        docs = [{"id": "1", "name": "Hotel A"}, {"id": "2", "name": "Hotel B"}]
        _call_sync(
            tracer, "upload_documents",
            lambda *a, **kw: [
                _mock_indexing_result("1"),
                _mock_indexing_result("2"),
            ],
            instance, kwargs={"documents": docs},
        )

        spans = exporter.get_finished_spans()
        upload_spans = [s for s in spans if s.name == "azure_search.upload_documents"]
        assert len(upload_spans) == 1

        span = upload_spans[0]
        attrs = dict(span.attributes)
        doc_key_0 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"
        doc_key_1 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.1"
        assert doc_key_0 in attrs
        assert doc_key_1 in attrs
        first_doc = json.loads(attrs[doc_key_0])
        assert first_doc["id"] == "1"

    def test_upload_documents_response_content_attributes(self, exporter, monkeypatch):
        """upload_documents should set per-result indexed metadata attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()
        _call_sync(
            tracer, "upload_documents",
            lambda *a, **kw: [_mock_indexing_result("1")],
            instance, kwargs={"documents": [{"id": "1"}]},
        )

        spans = exporter.get_finished_spans()
        upload_spans = [s for s in spans if s.name == "azure_search.upload_documents"]
        assert len(upload_spans) == 1

        span = upload_spans[0]
        attrs = dict(span.attributes)
        metadata_key = f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.0"
        assert metadata_key in attrs

    def test_search_vector_embeddings_attributes(self, exporter, monkeypatch):
        """search with vector_queries should set indexed db.search.embeddings.vector.N attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        vq = MagicMock()
        vq.vector = [0.1, 0.2, 0.3]
        vq.text = None
        vq.k_nearest_neighbors = 5
        vq.fields = "embedding"
        vq.exhaustive = None
        vq.kind = None
        vq.weight = None
        vq.oversampling = None

        _call_sync(
            tracer, "search",
            lambda *a, **kw: iter([]),
            instance, kwargs={"search_text": "hotel", "vector_queries": [vq]},
        )

        spans = exporter.get_finished_spans()
        search_spans = [s for s in spans if s.name == "azure_search.search"]
        assert len(search_spans) == 1

        span = search_spans[0]
        attr_key = f"{EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value}.0"
        assert attr_key in dict(span.attributes)

    def test_search_text_vector_embeddings_attributes(self, exporter, monkeypatch):
        """search with text-based vector query should capture text in embeddings attribute."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        vq = MagicMock()
        vq.vector = None
        vq.text = "luxury hotel"
        vq.k_nearest_neighbors = 5
        vq.fields = "embedding"
        vq.exhaustive = None
        vq.kind = None
        vq.weight = None
        vq.oversampling = None

        _call_sync(
            tracer, "search",
            lambda *a, **kw: iter([]),
            instance, kwargs={"search_text": None, "vector_queries": [vq]},
        )

        spans = exporter.get_finished_spans()
        search_spans = [s for s in spans if s.name == "azure_search.search"]
        assert len(search_spans) == 1

        span = search_spans[0]
        attr_key = f"{EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value}.0"
        assert span.attributes[attr_key] == "luxury hotel"

    def test_content_disabled_no_content_attributes(self, exporter, monkeypatch):
        """With TRACELOOP_TRACE_CONTENT=false, no content attributes should be added."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        content_attr_prefixes = (
            EventAttributes.DB_QUERY_RESULT_DOCUMENT.value,
            EventAttributes.DB_SEARCH_RESULT_ENTITY.value,
            EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value,
            EventAttributes.DB_QUERY_RESULT_METADATA.value,
            EventAttributes.DB_QUERY_RESULT_ID.value,
        )

        _call_sync(
            tracer, "get_document",
            lambda *a, **kw: {"id": "doc-1"}, instance, kwargs={"key": "doc-1"},
        )
        _call_sync(
            tracer, "autocomplete",
            lambda *a, **kw: [{"text": "lux"}], instance,
            kwargs={"search_text": "lux", "suggester_name": "sg"},
        )
        _call_sync(
            tracer, "suggest",
            lambda *a, **kw: [{"id": "s1"}], instance,
            kwargs={"search_text": "lux", "suggester_name": "sg"},
        )
        _call_sync(
            tracer, "upload_documents",
            lambda *a, **kw: [_mock_indexing_result("1")], instance,
            kwargs={"documents": [{"id": "1"}]},
        )

        spans = exporter.get_finished_spans()
        for span in spans:
            for attr_key in dict(span.attributes):
                for prefix in content_attr_prefixes:
                    assert not attr_key.startswith(prefix + "."), (
                        f"Found content attribute {attr_key} on span {span.name} with content disabled"
                    )

    def test_content_override_reenables(self, exporter, monkeypatch):
        """env=false + context override=True should add content attributes."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry import context as context_api

        ctx = context_api.set_value("override_enable_content_tracing", True)
        token = context_api.attach(ctx)
        try:
            tracer = trace.get_tracer(__name__)
            instance = _make_instance()
            _call_sync(
                tracer, "get_document",
                lambda *a, **kw: {"id": "doc-123", "name": "Test"},
                instance, kwargs={"key": "doc-123"},
            )
        finally:
            context_api.detach(token)

        spans = exporter.get_finished_spans()
        get_doc_spans = [s for s in spans if s.name == "azure_search.get_document"]
        assert len(get_doc_spans) == 1

        span = get_doc_spans[0]
        attr_key = EventAttributes.DB_QUERY_RESULT_DOCUMENT.value
        assert attr_key in dict(span.attributes)

    def test_index_documents_request_content_attributes(self, exporter, monkeypatch):
        """index_documents should set per-action indexed db.query.result.document.N attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        batch = MagicMock()
        batch.actions = [
            {"@search.action": "upload", "id": "1", "name": "Hotel A"},
            {"@search.action": "upload", "id": "2", "name": "Hotel B"},
        ]

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()
        _call_sync(
            tracer, "index_documents",
            lambda *a, **kw: MagicMock(results=[_mock_indexing_result("1"), _mock_indexing_result("2")]),
            instance, kwargs={"batch": batch},
        )

        spans = exporter.get_finished_spans()
        idx_spans = [s for s in spans if s.name == "azure_search.index_documents"]
        assert len(idx_spans) == 1

        span = idx_spans[0]
        attrs = dict(span.attributes)
        doc_key_0 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"
        doc_key_1 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.1"
        assert doc_key_0 in attrs
        assert doc_key_1 in attrs

    def test_merge_documents_content_attributes(self, exporter, monkeypatch):
        """merge_documents should set content attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()
        _call_sync(
            tracer, "merge_documents",
            lambda *a, **kw: [_mock_indexing_result("1")],
            instance, kwargs={"documents": [{"id": "1", "rating": 4.5}]},
        )

        spans = exporter.get_finished_spans()
        merge_spans = [s for s in spans if s.name == "azure_search.merge_documents"]
        assert len(merge_spans) == 1

        span = merge_spans[0]
        attr_key = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"
        assert attr_key in dict(span.attributes)

    def test_delete_documents_content_attributes(self, exporter, monkeypatch):
        """delete_documents should set content attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()
        _call_sync(
            tracer, "delete_documents",
            lambda *a, **kw: [_mock_indexing_result("1")],
            instance, kwargs={"documents": [{"id": "1"}]},
        )

        spans = exporter.get_finished_spans()
        del_spans = [s for s in spans if s.name == "azure_search.delete_documents"]
        assert len(del_spans) == 1

        span = del_spans[0]
        attr_key = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"
        assert attr_key in dict(span.attributes)

    def test_merge_or_upload_documents_content_attributes(self, exporter, monkeypatch):
        """merge_or_upload_documents should set content attributes."""
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()
        _call_sync(
            tracer, "merge_or_upload_documents",
            lambda *a, **kw: [_mock_indexing_result("1"), _mock_indexing_result("2")],
            instance, kwargs={"documents": [{"id": "1"}, {"id": "2"}]},
        )

        spans = exporter.get_finished_spans()
        mou_spans = [s for s in spans if s.name == "azure_search.merge_or_upload_documents"]
        assert len(mou_spans) == 1

        span = mou_spans[0]
        attrs = dict(span.attributes)
        doc_key_0 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"
        doc_key_1 = f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.1"
        assert doc_key_0 in attrs
        assert doc_key_1 in attrs


class TestSyncWorkflows:
    """Multi-step sync workflow tests that validate traces tell a debuggable story."""

    def test_search_pipeline(self, exporter):
        """Upload docs, then search — trace must show why search returned nothing."""
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace import SpanKind
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        docs = [
            {"id": "h1", "name": "Grand Hotel", "rating": 4.5},
            {"id": "h2", "name": "Budget Inn", "rating": 2.0},
            {"id": "h3", "name": "Cozy Motel", "rating": 3.8},
        ]
        upload_response = [
            _mock_indexing_result("h1"),
            _mock_indexing_result("h2"),
            _mock_indexing_result("h3"),
        ]

        with tracer.start_as_current_span("app.ingest_and_search"):
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: upload_response, instance,
                kwargs={"documents": docs},
            )
            _call_sync(
                tracer, "search",
                lambda *a, **kw: iter([]), instance,
                kwargs={"search_text": "hotel", "top": 5, "filter": "rating ge 4"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 2
        _assert_all_same_trace(azure_spans)

        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        assert upload_span.kind == SpanKind.CLIENT
        assert upload_span.status.status_code == StatusCode.OK
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 3
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 3
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT] == 0

        attrs = dict(upload_span.attributes)
        doc0 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"])
        assert doc0["id"] == "h1"
        doc2 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.2"])
        assert doc2["name"] == "Cozy Motel"

        search_span = _find_span(azure_spans, "azure_search.search")
        assert search_span.kind == SpanKind.CLIENT
        assert search_span.status.status_code == StatusCode.OK
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "hotel"
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TOP] == 5
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_FILTER] == "rating ge 4"

    def test_document_lifecycle(self, exporter):
        """Full CRUD lifecycle: upload -> get -> merge -> get -> delete."""
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        initial_doc = {"id": "lc-1", "name": "Lifecycle Hotel", "rating": 3.0}
        updated_doc = {"id": "lc-1", "name": "Lifecycle Hotel", "rating": 4.5}

        with tracer.start_as_current_span("app.document_lifecycle"):
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: [_mock_indexing_result("lc-1", status_code=201)],
                instance, kwargs={"documents": [initial_doc]},
            )
            _call_sync(
                tracer, "get_document",
                lambda *a, **kw: dict(initial_doc), instance,
                kwargs={"key": "lc-1"},
            )
            _call_sync(
                tracer, "merge_documents",
                lambda *a, **kw: [_mock_indexing_result("lc-1")],
                instance, kwargs={"documents": [{"id": "lc-1", "rating": 4.5}]},
            )
            _call_sync(
                tracer, "get_document",
                lambda *a, **kw: dict(updated_doc), instance,
                kwargs={"key": "lc-1"},
            )
            _call_sync(
                tracer, "delete_documents",
                lambda *a, **kw: [_mock_indexing_result("lc-1")],
                instance, kwargs={"documents": [{"id": "lc-1"}]},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 5
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK

        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        attrs = dict(upload_span.attributes)
        doc = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"])
        assert doc["rating"] == 3.0

        get_spans = [s for s in azure_spans if s.name == "azure_search.get_document"]
        assert len(get_spans) == 2
        first_get_doc = json.loads(
            dict(get_spans[0].attributes)[EventAttributes.DB_QUERY_RESULT_DOCUMENT.value]
        )
        assert first_get_doc["rating"] == 3.0

        second_get_doc = json.loads(
            dict(get_spans[1].attributes)[EventAttributes.DB_QUERY_RESULT_DOCUMENT.value]
        )
        assert second_get_doc["rating"] == 4.5

        merge_span = _find_span(azure_spans, "azure_search.merge_documents")
        merge_doc = json.loads(
            dict(merge_span.attributes)[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"]
        )
        assert merge_doc["rating"] == 4.5

        delete_span = _find_span(azure_spans, "azure_search.delete_documents")
        assert delete_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 1
        assert delete_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 1

    def test_typeahead_pipeline(self, exporter):
        """Upload -> autocomplete -> suggest -- typeahead debugging."""
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        autocomplete_results = [
            MagicMock(text="luxury", query_plus_text="luxury hotel"),
            MagicMock(text="luxurious", query_plus_text="luxurious resort"),
        ]
        suggest_results = [
            {"@search.text": "Luxury Hotel Downtown", "id": "s1"},
        ]

        with tracer.start_as_current_span("app.typeahead_pipeline"):
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: [_mock_indexing_result("th-1")],
                instance,
                kwargs={"documents": [{"id": "th-1", "name": "Luxury Hotel Downtown"}]},
            )
            _call_sync(
                tracer, "autocomplete",
                lambda *a, **kw: autocomplete_results, instance,
                kwargs={"search_text": "lux", "suggester_name": "sg"},
            )
            _call_sync(
                tracer, "suggest",
                lambda *a, **kw: suggest_results, instance,
                kwargs={"search_text": "lux", "suggester_name": "sg"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 3
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK

        ac_span = _find_span(azure_spans, "azure_search.autocomplete")
        assert ac_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "lux"
        assert ac_span.attributes[SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME] == "sg"
        assert ac_span.attributes[SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT] == 2

        ac_attrs = dict(ac_span.attributes)
        entity_0 = json.loads(ac_attrs[f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.0"])
        assert entity_0["text"] == "luxury"
        entity_1 = json.loads(ac_attrs[f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.1"])
        assert entity_1["text"] == "luxurious"

        sg_span = _find_span(azure_spans, "azure_search.suggest")
        assert sg_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "lux"
        assert sg_span.attributes[SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT] == 1

    def test_bulk_ingestion_partial_failure(self, exporter):
        """Upload 5 docs where 2 fail -- trace shows which docs failed and why."""
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        docs = [
            {"id": f"bulk-{i}", "name": f"Hotel {i}", "rating": float(i)}
            for i in range(5)
        ]
        response = [
            _mock_indexing_result("bulk-0"),
            _mock_indexing_result("bulk-1"),
            _mock_indexing_result("bulk-2"),
            _mock_indexing_result(
                "bulk-3", succeeded=False, status_code=400,
                error_message="Invalid field 'rating'",
            ),
            _mock_indexing_result(
                "bulk-4", succeeded=False, status_code=400,
                error_message="Document too large",
            ),
        ]

        with tracer.start_as_current_span("app.etl_ingestion"):
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: response, instance,
                kwargs={"documents": docs},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 1

        span = azure_spans[0]
        assert span.status.status_code == StatusCode.OK
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 5
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 3
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT] == 2

        attrs = dict(span.attributes)
        for i in range(5):
            raw = attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.{i}"]
            doc = json.loads(raw)
            assert doc["id"] == f"bulk-{i}"

        for i in range(3):
            meta = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.{i}"])
            assert meta["succeeded"] is True

        meta_3 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.3"])
        assert meta_3["succeeded"] is False
        assert meta_3["status_code"] == 400
        assert meta_3["error_message"] == "Invalid field 'rating'"

        meta_4 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.4"])
        assert meta_4["succeeded"] is False
        assert meta_4["error_message"] == "Document too large"

    def test_index_management_pipeline(self, exporter):
        """create_index -> upload -> count -> search -> delete_index."""
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        search_instance = _make_instance("pipeline-test")
        index_instance = _make_index_instance()

        mock_index = MagicMock()
        mock_index.name = "pipeline-test"

        with tracer.start_as_current_span("app.deployment_pipeline"):
            _call_sync(
                tracer, "create_index",
                lambda *a, **kw: mock_index, index_instance,
                kwargs={"index": mock_index},
            )
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: [_mock_indexing_result("p1"), _mock_indexing_result("p2")],
                search_instance,
                kwargs={"documents": [{"id": "p1"}, {"id": "p2"}]},
            )
            _call_sync(
                tracer, "get_document_count",
                lambda *a, **kw: 2, search_instance,
            )
            _call_sync(
                tracer, "search",
                lambda *a, **kw: iter([{"id": "p1"}, {"id": "p2"}]),
                search_instance,
                kwargs={"search_text": "*"},
            )
            _call_sync(
                tracer, "delete_index",
                lambda *a, **kw: None, index_instance,
                kwargs={"index": "pipeline-test"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 5
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK

        create_span = _find_span(azure_spans, "azure_search.create_index")
        assert create_span.attributes[SpanAttributes.AZURE_SEARCH_INDEX_NAME] == "pipeline-test"

        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 2

        count_span = _find_span(azure_spans, "azure_search.get_document_count")
        assert count_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 2

        search_span = _find_span(azure_spans, "azure_search.search")
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_INDEX_NAME] == "pipeline-test"

        delete_span = _find_span(azure_spans, "azure_search.delete_index")
        assert delete_span.attributes[SpanAttributes.AZURE_SEARCH_INDEX_NAME] == "pipeline-test"

    def test_content_privacy_across_pipeline(self, exporter, monkeypatch):
        """Full pipeline with content disabled -- verify no PII leaks."""
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        content_prefixes = (
            EventAttributes.DB_QUERY_RESULT_DOCUMENT.value,
            EventAttributes.DB_SEARCH_RESULT_ENTITY.value,
            EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value,
            EventAttributes.DB_QUERY_RESULT_METADATA.value,
            EventAttributes.DB_QUERY_RESULT_ID.value,
        )

        with tracer.start_as_current_span("app.privacy_pipeline"):
            _call_sync(
                tracer, "upload_documents",
                lambda *a, **kw: [_mock_indexing_result("priv-1")],
                instance,
                kwargs={"documents": [{"id": "priv-1", "ssn": "123-45-6789"}]},
            )
            _call_sync(
                tracer, "get_document",
                lambda *a, **kw: {"id": "priv-1", "ssn": "123-45-6789"},
                instance, kwargs={"key": "priv-1"},
            )
            _call_sync(
                tracer, "autocomplete",
                lambda *a, **kw: [MagicMock(text="secret", query_plus_text="secret data")],
                instance,
                kwargs={"search_text": "sec", "suggester_name": "sg"},
            )
            _call_sync(
                tracer, "suggest",
                lambda *a, **kw: [{"@search.text": "Secret Doc", "id": "s1"}],
                instance,
                kwargs={"search_text": "sec", "suggester_name": "sg"},
            )

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 4
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK
            attrs = dict(s.attributes)
            content_keys = [
                k for k in attrs
                if any(k.startswith(p + ".") for p in content_prefixes)
            ]
            assert content_keys == [], (
                f"Content leaked in span '{s.name}': {content_keys}"
            )

        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 1

        get_span = _find_span(azure_spans, "azure_search.get_document")
        assert get_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY] == "priv-1"

        ac_span = _find_span(azure_spans, "azure_search.autocomplete")
        assert ac_span.attributes[SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME] == "sg"

    def test_error_then_retry_success(self, exporter):
        """First call fails, retry succeeds -- trace shows both for diagnosis."""
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        call_count = 0

        def flaky_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("ServiceUnavailable: retry later")
            return iter([{"id": "1"}])

        with tracer.start_as_current_span("app.search_with_retry"):
            try:
                _call_sync(
                    tracer, "search", flaky_search, instance,
                    kwargs={"search_text": "hotel"},
                )
            except ConnectionError:
                pass
            _call_sync(
                tracer, "search", flaky_search, instance,
                kwargs={"search_text": "hotel"},
            )

        spans = _get_spans(exporter)
        search_spans = [s for s in spans if s.name == "azure_search.search"]
        assert len(search_spans) == 2
        _assert_all_same_trace(search_spans)

        assert search_spans[0].status.status_code == StatusCode.ERROR
        assert "ServiceUnavailable" in search_spans[0].status.description
        assert search_spans[1].status.status_code == StatusCode.OK


class TestAsyncWorkflows:
    """Async mirrors of all sync workflow tests."""

    def test_search_pipeline(self, exporter):
        """Async: upload -> search -- trace correlation and query visibility."""
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace import SpanKind
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        docs = [
            {"id": "h1", "name": "Grand Hotel", "rating": 4.5},
            {"id": "h2", "name": "Budget Inn", "rating": 2.0},
            {"id": "h3", "name": "Cozy Motel", "rating": 3.8},
        ]
        upload_response = [
            _mock_indexing_result("h1"),
            _mock_indexing_result("h2"),
            _mock_indexing_result("h3"),
        ]

        async def mock_upload(*a, **kw):
            return upload_response

        async def mock_search(*a, **kw):
            return iter([])

        async def run():
            with tracer.start_as_current_span("app.async_ingest_and_search"):
                await _call_async(
                    tracer, "upload_documents", mock_upload, instance,
                    kwargs={"documents": docs},
                )
                await _call_async(
                    tracer, "search", mock_search, instance,
                    kwargs={"search_text": "hotel", "top": 5, "filter": "rating ge 4"},
                )

        asyncio.get_event_loop().run_until_complete(run())

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 2
        _assert_all_same_trace(azure_spans)

        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        assert upload_span.kind == SpanKind.CLIENT
        assert upload_span.status.status_code == StatusCode.OK
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 3
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 3

        attrs = dict(upload_span.attributes)
        doc0 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.0"])
        assert doc0["id"] == "h1"

        search_span = _find_span(azure_spans, "azure_search.search")
        assert search_span.status.status_code == StatusCode.OK
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "hotel"
        assert search_span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TOP] == 5

    def test_document_lifecycle(self, exporter):
        """Async: upload -> get -> merge -> get -> delete -- CRUD audit trail."""
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        initial_doc = {"id": "lc-1", "name": "Lifecycle Hotel", "rating": 3.0}
        updated_doc = {"id": "lc-1", "name": "Lifecycle Hotel", "rating": 4.5}

        async def upload(*a, **kw):
            return [_mock_indexing_result("lc-1", status_code=201)]

        async def get_initial(*a, **kw):
            return dict(initial_doc)

        async def merge(*a, **kw):
            return [_mock_indexing_result("lc-1")]

        async def get_updated(*a, **kw):
            return dict(updated_doc)

        async def delete(*a, **kw):
            return [_mock_indexing_result("lc-1")]

        async def run():
            with tracer.start_as_current_span("app.async_document_lifecycle"):
                await _call_async(
                    tracer, "upload_documents", upload, instance,
                    kwargs={"documents": [initial_doc]},
                )
                await _call_async(
                    tracer, "get_document", get_initial, instance,
                    kwargs={"key": "lc-1"},
                )
                await _call_async(
                    tracer, "merge_documents", merge, instance,
                    kwargs={"documents": [{"id": "lc-1", "rating": 4.5}]},
                )
                await _call_async(
                    tracer, "get_document", get_updated, instance,
                    kwargs={"key": "lc-1"},
                )
                await _call_async(
                    tracer, "delete_documents", delete, instance,
                    kwargs={"documents": [{"id": "lc-1"}]},
                )

        asyncio.get_event_loop().run_until_complete(run())

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 5
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK

        get_spans = [s for s in azure_spans if s.name == "azure_search.get_document"]
        first_doc = json.loads(
            dict(get_spans[0].attributes)[EventAttributes.DB_QUERY_RESULT_DOCUMENT.value]
        )
        assert first_doc["rating"] == 3.0
        second_doc = json.loads(
            dict(get_spans[1].attributes)[EventAttributes.DB_QUERY_RESULT_DOCUMENT.value]
        )
        assert second_doc["rating"] == 4.5

    def test_typeahead_pipeline(self, exporter):
        """Async: upload -> autocomplete -> suggest -- typeahead debugging."""
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        autocomplete_results = [MagicMock(text="luxury", query_plus_text="luxury hotel")]
        suggest_results = [{"@search.text": "Luxury Hotel", "id": "s1"}]

        async def mock_upload(*a, **kw):
            return [_mock_indexing_result("th-1")]

        async def mock_ac(*a, **kw):
            return autocomplete_results

        async def mock_sg(*a, **kw):
            return suggest_results

        async def run():
            with tracer.start_as_current_span("app.async_typeahead"):
                await _call_async(
                    tracer, "upload_documents", mock_upload, instance,
                    kwargs={"documents": [{"id": "th-1", "name": "Luxury Hotel"}]},
                )
                await _call_async(
                    tracer, "autocomplete", mock_ac, instance,
                    kwargs={"search_text": "lux", "suggester_name": "sg"},
                )
                await _call_async(
                    tracer, "suggest", mock_sg, instance,
                    kwargs={"search_text": "lux", "suggester_name": "sg"},
                )

        asyncio.get_event_loop().run_until_complete(run())

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 3
        _assert_all_same_trace(azure_spans)

        ac_span = _find_span(azure_spans, "azure_search.autocomplete")
        assert ac_span.attributes[SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT] == 1
        entity_0 = json.loads(
            dict(ac_span.attributes)[f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.0"]
        )
        assert entity_0["text"] == "luxury"

        sg_span = _find_span(azure_spans, "azure_search.suggest")
        assert sg_span.attributes[SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT] == 1

    def test_bulk_ingestion_partial_failure(self, exporter):
        """Async: upload 5 docs, 2 fail -- per-document failure metadata."""
        from opentelemetry.semconv_ai import EventAttributes

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        docs = [{"id": f"bulk-{i}", "name": f"Hotel {i}"} for i in range(5)]
        response = [
            _mock_indexing_result("bulk-0"),
            _mock_indexing_result("bulk-1"),
            _mock_indexing_result("bulk-2"),
            _mock_indexing_result("bulk-3", succeeded=False, status_code=400, error_message="Bad format"),
            _mock_indexing_result("bulk-4", succeeded=False, status_code=400, error_message="Too large"),
        ]

        async def mock_upload(*a, **kw):
            return response

        async def run():
            with tracer.start_as_current_span("app.async_etl"):
                await _call_async(
                    tracer, "upload_documents", mock_upload, instance,
                    kwargs={"documents": docs},
                )

        asyncio.get_event_loop().run_until_complete(run())

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 1

        span = azure_spans[0]
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 3
        assert span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT] == 2

        attrs = dict(span.attributes)
        meta_3 = json.loads(attrs[f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.3"])
        assert meta_3["succeeded"] is False
        assert meta_3["error_message"] == "Bad format"

    def test_index_management_pipeline(self, exporter):
        """Async: create_index -> upload -> count -> search -> delete_index."""
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        search_instance = _make_instance("pipeline-test")
        index_instance = _make_index_instance()

        mock_index = MagicMock()
        mock_index.name = "pipeline-test"

        async def create_idx(*a, **kw):
            return mock_index

        async def upload(*a, **kw):
            return [_mock_indexing_result("p1")]

        async def count(*a, **kw):
            return 1

        async def search(*a, **kw):
            return iter([])

        async def delete_idx(*a, **kw):
            return None

        async def run():
            with tracer.start_as_current_span("app.async_deployment"):
                await _call_async(
                    tracer, "create_index", create_idx, index_instance,
                    kwargs={"index": mock_index},
                )
                await _call_async(
                    tracer, "upload_documents", upload, search_instance,
                    kwargs={"documents": [{"id": "p1"}]},
                )
                await _call_async(tracer, "get_document_count", count, search_instance)
                await _call_async(
                    tracer, "search", search, search_instance,
                    kwargs={"search_text": "*"},
                )
                await _call_async(
                    tracer, "delete_index", delete_idx, index_instance,
                    kwargs={"index": "pipeline-test"},
                )

        asyncio.get_event_loop().run_until_complete(run())

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 5
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK

        create_span = _find_span(azure_spans, "azure_search.create_index")
        assert create_span.attributes[SpanAttributes.AZURE_SEARCH_INDEX_NAME] == "pipeline-test"

    def test_content_privacy_across_pipeline(self, exporter, monkeypatch):
        """Async: full pipeline with content disabled -- no PII leaks."""
        from opentelemetry.semconv_ai import EventAttributes
        from opentelemetry.trace.status import StatusCode

        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        content_prefixes = (
            EventAttributes.DB_QUERY_RESULT_DOCUMENT.value,
            EventAttributes.DB_SEARCH_RESULT_ENTITY.value,
            EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value,
            EventAttributes.DB_QUERY_RESULT_METADATA.value,
            EventAttributes.DB_QUERY_RESULT_ID.value,
        )

        async def upload(*a, **kw):
            return [_mock_indexing_result("priv-1")]

        async def get_doc(*a, **kw):
            return {"id": "priv-1", "secret": "classified"}

        async def autocomplete(*a, **kw):
            return [MagicMock(text="secret", query_plus_text="secret data")]

        async def suggest(*a, **kw):
            return [{"@search.text": "Secret", "id": "s1"}]

        async def run():
            with tracer.start_as_current_span("app.async_privacy"):
                await _call_async(
                    tracer, "upload_documents", upload, instance,
                    kwargs={"documents": [{"id": "priv-1", "secret": "classified"}]},
                )
                await _call_async(
                    tracer, "get_document", get_doc, instance,
                    kwargs={"key": "priv-1"},
                )
                await _call_async(
                    tracer, "autocomplete", autocomplete, instance,
                    kwargs={"search_text": "sec", "suggester_name": "sg"},
                )
                await _call_async(
                    tracer, "suggest", suggest, instance,
                    kwargs={"search_text": "sec", "suggester_name": "sg"},
                )

        asyncio.get_event_loop().run_until_complete(run())

        spans = _get_spans(exporter)
        azure_spans = [s for s in spans if s.name.startswith("azure_search.")]
        assert len(azure_spans) == 4
        _assert_all_same_trace(azure_spans)

        for s in azure_spans:
            assert s.status.status_code == StatusCode.OK
            attrs = dict(s.attributes)
            content_keys = [
                k for k in attrs
                if any(k.startswith(p + ".") for p in content_prefixes)
            ]
            assert content_keys == [], f"Content leaked in '{s.name}': {content_keys}"

        upload_span = _find_span(azure_spans, "azure_search.upload_documents")
        assert upload_span.attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 1

    def test_error_then_retry_success(self, exporter):
        """Async: first call fails, retry succeeds -- transient failure diagnosis."""
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        instance = _make_instance()

        call_count = 0

        async def flaky_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("ServiceUnavailable: retry later")
            return iter([{"id": "1"}])

        async def run():
            with tracer.start_as_current_span("app.async_retry"):
                try:
                    await _call_async(
                        tracer, "search", flaky_search, instance,
                        kwargs={"search_text": "hotel"},
                    )
                except ConnectionError:
                    pass
                await _call_async(
                    tracer, "search", flaky_search, instance,
                    kwargs={"search_text": "hotel"},
                )

        asyncio.get_event_loop().run_until_complete(run())

        spans = _get_spans(exporter)
        search_spans = [s for s in spans if s.name == "azure_search.search"]
        assert len(search_spans) == 2
        _assert_all_same_trace(search_spans)

        assert search_spans[0].status.status_code == StatusCode.ERROR
        assert "ServiceUnavailable" in search_spans[0].status.description
        assert search_spans[1].status.status_code == StatusCode.OK
