"""Tests for Azure AI Search instrumentation."""

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


# ---------------------------------------------------------------------------
# Tests — SearchClient instrumentation via manual spans
# ---------------------------------------------------------------------------

class TestSearchClientInstrumentation:
    """Tests for SearchClient span creation (manual span helpers)."""

    def test_search_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.search",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "test-index",
                SpanAttributes.AZURE_SEARCH_SEARCH_TEXT: "luxury hotel",
                SpanAttributes.AZURE_SEARCH_SEARCH_TOP: 10,
                SpanAttributes.AZURE_SEARCH_SEARCH_FILTER: "rating ge 4",
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            list(client.search(search_text="luxury hotel", top=10, filter="rating ge 4"))

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "azure_search.search"
        assert span.attributes[SpanAttributes.VECTOR_DB_VENDOR] == "Azure AI Search"
        assert span.attributes[SpanAttributes.AZURE_SEARCH_INDEX_NAME] == "test-index"
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TEXT] == "luxury hotel"
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_TOP] == 10
        assert span.attributes[SpanAttributes.AZURE_SEARCH_SEARCH_FILTER] == "rating ge 4"

    def test_get_document_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.get_document",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY: "doc-123",
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.get_document(key="doc-123")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "azure_search.get_document"
        assert spans[0].attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY] == "doc-123"

    def test_upload_documents_creates_span(self, exporter):
        documents = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.upload_documents",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT: len(documents),
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.upload_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 3

    def test_search_with_skip_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.search",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_SEARCH_SKIP: 5,
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            list(client.search(search_text="*", top=10, skip=5))

        spans = exporter.get_finished_spans()
        assert spans[0].attributes[SpanAttributes.AZURE_SEARCH_SEARCH_SKIP] == 5

    def test_get_document_count_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.get_document_count",
            attributes={
                SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
                SpanAttributes.AZURE_SEARCH_INDEX_NAME: "test-index",
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.get_document_count()

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure_search.get_document_count"

    def test_merge_documents_creates_span(self, exporter):
        documents = [{"id": "1", "rating": 4.8}]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.merge_documents",
            attributes={
                SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT: 1,
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.merge_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure_search.merge_documents"
        assert spans[0].attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 1

    def test_delete_documents_creates_span(self, exporter):
        documents = [{"id": "1"}, {"id": "2"}]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.delete_documents",
            attributes={SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT: 2},
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.delete_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes[SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT] == 2

    def test_merge_or_upload_creates_span(self, exporter):
        documents = [{"id": "1"}]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.merge_or_upload_documents",
            attributes={SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT: 1},
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.merge_or_upload_documents(documents=documents)

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure_search.merge_or_upload_documents"

    def test_index_documents_creates_span(self, exporter):
        batch = MagicMock()
        batch.actions = [{"id": "1"}]
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.index_documents",
            attributes={SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT: 1},
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.index_documents(batch=batch)

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure_search.index_documents"

    def test_autocomplete_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.autocomplete",
            attributes={
                SpanAttributes.AZURE_SEARCH_SEARCH_TEXT: "lux",
                SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME: "sg",
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.autocomplete(search_text="lux", suggester_name="sg")

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure_search.autocomplete"
        assert spans[0].attributes[SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME] == "sg"

    def test_suggest_creates_span(self, exporter):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "azure_search.suggest",
            attributes={
                SpanAttributes.AZURE_SEARCH_SEARCH_TEXT: "hot",
                SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME: "sg",
            },
        ):
            client = MockSearchClient("https://test.search.windows.net", "test-index", MagicMock())
            client.suggest(search_text="hot", suggester_name="sg")

        spans = exporter.get_finished_spans()
        assert spans[0].name == "azure_search.suggest"
        assert spans[0].attributes[SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME] == "sg"


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
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "hotels"

    def test_search_top_and_skip(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, [], {"top": 5, "skip": 10})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TOP) == 5
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_SKIP) == 10

    def test_search_filter_attribute(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, [], {"filter": "rating ge 4"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_FILTER) == "rating ge 4"

    def test_query_type_string(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, [], {"query_type": "full"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_QUERY_TYPE) == "full"

    def test_query_type_enum(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_search_attributes
        mock_enum = MagicMock()
        mock_enum.value = "semantic"
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_search_attributes(span, [], {"query_type": mock_enum})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_QUERY_TYPE) == "semantic"

    def test_document_key_attribute(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_get_document_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_get_document_attributes(span, [], {"key": "hotel-1"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY) == "hotel-1"

    def test_document_batch_count(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_batch_attributes
        tracer = trace.get_tracer(__name__)
        docs = [{"id": "1"}, {"id": "2"}]
        with tracer.start_as_current_span("test") as span:
            _set_document_batch_attributes(span, [], {"documents": docs})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 2

    def test_suggester_name_attribute(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_suggestion_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_suggestion_attributes(span, [], {"search_text": "ho", "suggester_name": "sg"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME) == "sg"


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
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_RESULTS_COUNT) == 42

    def test_document_count_from_int(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_document_count_response_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_document_count_response_attributes(span, 100)
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT) == 100

    def test_autocomplete_results_count(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_autocomplete_response_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_autocomplete_response_attributes(span, [{"text": "a"}, {"text": "b"}])
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT) == 2

    def test_suggest_results_count(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _set_suggest_response_attributes
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            _set_suggest_response_attributes(span, [{"text": "hotel"}])
        spans = exporter.get_finished_spans()
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT) == 1

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
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT) == 2
        assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT) == 1


class TestErrorHandling:
    """Tests for error status and suppression key."""

    def test_error_status_on_exception(self, exporter):
        from opentelemetry.instrumentation.azure_search.wrapper import _sync_wrap
        from opentelemetry.trace.status import StatusCode

        tracer = trace.get_tracer(__name__)
        to_wrap = {"span_name": "azure_search.search", "method": "search"}

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
        to_wrap = {"span_name": "azure_search.search", "method": "search"}

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
