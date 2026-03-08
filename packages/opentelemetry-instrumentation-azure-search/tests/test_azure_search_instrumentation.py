"""Tests for Azure AI Search instrumentation — semconv + scaffold."""

from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor


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
    """Verify the instrumentor scaffold."""

    def test_instrumentation_dependencies(self):
        instrumentor = AzureSearchInstrumentor()
        deps = list(instrumentor.instrumentation_dependencies())
        assert len(deps) == 1
        assert "azure-search-documents" in deps[0]

    def test_instrument_uninstrument_noop(self):
        """instrument() / uninstrument() should not raise."""
        instrumentor = AzureSearchInstrumentor()
        # Already instrumented by conftest fixture; uninstrument is safe to call
        instrumentor._uninstrument()
        instrumentor._instrument()
