"""Integration tests for Azure AI Search instrumentation using VCR cassettes.

Each test verifies that a real Azure SDK call (replayed from a recorded cassette)
produces a span with the correct name, kind, status, vendor, request attributes,
and response attributes.
"""

import os

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import IndexDocumentsBatch, SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchFieldDataType,
    SearchIndex,
    SearchSuggester,
    SearchableField,
    SimpleField,
)
from opentelemetry.semconv_ai import EventAttributes, SpanAttributes
from opentelemetry.trace import SpanKind, StatusCode

INTEGRATION_TEST_INDEX = "otel-integration-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_only_span(exporter, span_name):
    """Return the single span matching span_name, or fail with a clear message."""
    spans = exporter.get_finished_spans()
    matching = [s for s in spans if s.name == span_name]
    assert len(matching) == 1, (
        f"Expected exactly 1 '{span_name}' span, "
        f"got {len(matching)} out of {len(spans)} total spans: "
        f"{[s.name for s in spans]}"
    )
    return matching[0]


def _assert_base_span(span, expected_name, index_name=None):
    """Verify the common properties every instrumented span must have."""
    assert span.name == expected_name
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.OK
    assert span.attributes[SpanAttributes.VECTOR_DB_VENDOR] == SpanAttributes.AZURE_AI_SEARCH_DB_SYSTEM_NAME
    if index_name is not None:
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME] == index_name


def _span_attrs(span):
    """Return span attributes as a plain dict for easier key/value access."""
    return dict(span.attributes)


def _assert_no_content_attributes(span):
    """Assert that zero content-capture attributes are present on a span."""
    content_prefixes = (
        EventAttributes.DB_QUERY_RESULT_DOCUMENT.value,
        EventAttributes.DB_SEARCH_RESULT_ENTITY.value,
        EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value,
        EventAttributes.DB_QUERY_RESULT_METADATA.value,
        EventAttributes.DB_QUERY_RESULT_ID.value,
    )
    attrs = _span_attrs(span)
    content_keys = [
        k for k in attrs
        if any(k.startswith(p + ".") for p in content_prefixes)
    ]
    assert content_keys == [], (
        f"Found content attributes with content tracing disabled: {content_keys}"
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_index_client():
    return SearchIndexClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
    )


def _make_search_client(index_name=INTEGRATION_TEST_INDEX):
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=index_name,
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
    )


def _is_playback_mode():
    return os.environ.get("AZURE_SEARCH_ADMIN_KEY") == "test-api-key"


def _setup_index(index_client, fields, suggesters=None):
    """Create the integration-test index (idempotent). Skips in playback mode."""
    if _is_playback_mode():
        return
    try:
        index_client.delete_index(INTEGRATION_TEST_INDEX)
    except Exception:
        pass
    index = SearchIndex(
        name=INTEGRATION_TEST_INDEX,
        fields=fields,
        suggesters=suggesters or [],
    )
    index_client.create_index(index)


def _teardown_index(index_client):
    """Delete the integration-test index. Skips in playback mode."""
    if _is_playback_mode():
        return
    try:
        index_client.delete_index(INTEGRATION_TEST_INDEX)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# TestSearchClientIntegration
# ---------------------------------------------------------------------------

class TestSearchClientIntegration:
    """Integration tests for SearchClient operations (search, document CRUD, autocomplete, suggest)."""

    @pytest.fixture(scope="class")
    def index_client_setup(self):
        return _make_index_client()

    @pytest.fixture(scope="class", autouse=True)
    def setup_test_index(self, index_client_setup):
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="name", type=SearchFieldDataType.String),
            SearchableField(name="description", type=SearchFieldDataType.String),
            SimpleField(name="rating", type=SearchFieldDataType.Double, filterable=True),
        ]
        suggesters = [SearchSuggester(name="sg", source_fields=["name"])]
        _setup_index(index_client_setup, fields, suggesters)
        yield
        _teardown_index(index_client_setup)

    @pytest.fixture
    def search_client(self):
        return _make_search_client()

    @pytest.fixture
    def index_client(self):
        return _make_index_client()

    # -- Search operations --

    @pytest.mark.vcr
    def test_search(self, exporter, search_client):
        """Search with text, top, and filter captures all query parameters."""
        list(search_client.search(search_text="hotel", top=5, filter="rating ge 3"))

        span = _get_only_span(exporter, "azure.search.search")
        _assert_base_span(span, "azure.search.search", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT] == "hotel"
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SEARCH_TOP] == 5
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SEARCH_FILTER] == "rating ge 3"

    @pytest.mark.vcr
    def test_search_with_skip(self, exporter, search_client):
        """Search with skip parameter captures pagination offset."""
        list(search_client.search(search_text="*", top=10, skip=5))

        span = _get_only_span(exporter, "azure.search.search")
        _assert_base_span(span, "azure.search.search", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SEARCH_SKIP] == 5

    # -- Document retrieval --

    @pytest.mark.vcr
    def test_get_document(self, exporter, search_client):
        """get_document captures the document key."""
        # Cassette includes a prior upload of doc-1, then the GET
        search_client.upload_documents([
            {"id": "doc-1", "name": "Test", "description": "Test", "rating": 4.0},
        ])
        exporter.clear()

        search_client.get_document(key="doc-1")

        span = _get_only_span(exporter, "azure.search.get_document")
        _assert_base_span(span, "azure.search.get_document", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_KEY] == "doc-1"

    @pytest.mark.vcr
    def test_get_document_count(self, exporter, search_client):
        """get_document_count captures the count as a response attribute."""
        count = search_client.get_document_count()

        span = _get_only_span(exporter, "azure.search.get_document_count")
        _assert_base_span(span, "azure.search.get_document_count", INTEGRATION_TEST_INDEX)
        # The cassette returns "1" — verify the response attribute matches
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT] == count

    # -- Document write operations --

    @pytest.mark.vcr
    def test_upload_documents(self, exporter, search_client):
        """upload_documents captures doc count and succeeded/failed counts."""
        documents = [
            {"id": "test-1", "name": "Test Hotel 1", "description": "A test hotel", "rating": 4.0},
            {"id": "test-2", "name": "Test Hotel 2", "description": "Another test hotel", "rating": 3.5},
        ]
        search_client.upload_documents(documents=documents)

        span = _get_only_span(exporter, "azure.search.upload_documents")
        _assert_base_span(span, "azure.search.upload_documents", INTEGRATION_TEST_INDEX)

        # Request attributes
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT] == 2

        # Response attributes — cassette shows both docs succeed with 201
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 2
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_FAILED_COUNT] == 0

    @pytest.mark.vcr
    def test_merge_documents(self, exporter, search_client):
        """merge_documents captures doc count and succeeded/failed counts."""
        # Cassette uploads merge-1 first, then merges it
        search_client.upload_documents([
            {"id": "merge-1", "name": "Merge Test", "description": "Test", "rating": 3.0},
        ])
        exporter.clear()

        search_client.merge_documents(documents=[{"id": "merge-1", "rating": 4.8}])

        span = _get_only_span(exporter, "azure.search.merge_documents")
        _assert_base_span(span, "azure.search.merge_documents", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_FAILED_COUNT] == 0

    @pytest.mark.vcr
    def test_delete_documents(self, exporter, search_client):
        """delete_documents captures doc count and succeeded/failed counts."""
        search_client.delete_documents(documents=[{"id": "test-1"}, {"id": "test-2"}])

        span = _get_only_span(exporter, "azure.search.delete_documents")
        _assert_base_span(span, "azure.search.delete_documents", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT] == 2
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 2
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_FAILED_COUNT] == 0

    @pytest.mark.vcr
    def test_merge_or_upload_documents(self, exporter, search_client):
        """merge_or_upload_documents captures doc count and succeeded/failed counts."""
        search_client.merge_or_upload_documents(
            documents=[{"id": "upsert-1", "name": "Upsert Hotel", "description": "A test upsert", "rating": 4.2}],
        )

        span = _get_only_span(exporter, "azure.search.merge_or_upload_documents")
        _assert_base_span(span, "azure.search.merge_or_upload_documents", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_FAILED_COUNT] == 0

    @pytest.mark.vcr
    def test_index_documents(self, exporter, search_client):
        """index_documents (batch API) captures batch size and succeeded/failed counts."""
        batch = IndexDocumentsBatch()
        batch.add_upload_actions([
            {"id": "batch-1", "name": "Batch Hotel", "description": "A batch test", "rating": 3.9},
        ])
        search_client.index_documents(batch=batch)

        span = _get_only_span(exporter, "azure.search.index_documents")
        _assert_base_span(span, "azure.search.index_documents", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_SUCCEEDED_COUNT] == 1
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_FAILED_COUNT] == 0

    # -- Autocomplete & Suggest --

    @pytest.mark.vcr
    def test_autocomplete(self, exporter, search_client):
        """autocomplete captures search text, suggester, and results count."""
        # Cassette uploads auto-1 first, then autocompletes
        search_client.upload_documents([
            {"id": "auto-1", "name": "Luxury Hotel", "description": "A luxury hotel", "rating": 5.0},
        ])
        exporter.clear()

        list(search_client.autocomplete(search_text="lux", suggester_name="sg"))

        span = _get_only_span(exporter, "azure.search.autocomplete")
        _assert_base_span(span, "azure.search.autocomplete", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT] == "lux"
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SUGGESTER_NAME] == "sg"

        # Cassette returns 1 result: {"text":"luxury","queryPlusText":"luxury"}
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_AUTOCOMPLETE_RESULTS_COUNT] == 1

    @pytest.mark.vcr
    def test_suggest(self, exporter, search_client):
        """suggest captures search text, suggester, and results count."""
        # Cassette uploads sug-1 first, then suggests
        search_client.upload_documents([
            {"id": "sug-1", "name": "Hot Springs Resort", "description": "A hot springs resort", "rating": 4.5},
        ])
        exporter.clear()

        results = list(search_client.suggest(search_text="hot", suggester_name="sg"))

        span = _get_only_span(exporter, "azure.search.suggest")
        _assert_base_span(span, "azure.search.suggest", INTEGRATION_TEST_INDEX)
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT] == "hot"
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SUGGESTER_NAME] == "sg"

        # Cassette returns suggestions
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SUGGEST_RESULTS_COUNT] == len(results)
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_SUGGEST_RESULTS_COUNT] >= 1

    # -- Content toggle --

    @pytest.mark.vcr
    def test_content_disabled_no_content_attributes(self, exporter, search_client, monkeypatch):
        """With TRACELOOP_TRACE_CONTENT=false, get_document still creates a span but omits content."""
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

        search_client.get_document(key="1")

        span = _get_only_span(exporter, "azure.search.get_document")
        _assert_base_span(span, "azure.search.get_document", INTEGRATION_TEST_INDEX)

        # The document key is always captured (metadata, not content)
        assert span.attributes[SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_KEY] == "1"

        # But the document body must NOT be captured
        _assert_no_content_attributes(span)
