# OpenTelemetry Azure AI Search Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-azure-search/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-azure-search.svg" alt="PyPI version">
</a>

This library provides automatic instrumentation for the [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/) Python SDK (`azure-search-documents`).

## Installation

```bash
pip install opentelemetry-instrumentation-azure-search
```

## Usage

```python
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

AzureSearchInstrumentor().instrument()
```

The instrumentor hooks into the Azure AI Search Python SDK and automatically generates OpenTelemetry spans for every instrumented operation. No changes to your application code are required beyond calling `instrument()`.

## Status

> **Note:** This is the initial package scaffold. Instrumented operations will be added in subsequent releases.

The `AzureSearchInstrumentor` class is registered and ready to use, but no SDK methods are wrapped yet — all spans will be no-ops until operation instrumentation is added.

## Semantic Conventions

Span attributes follow the `azure.search.*` naming convention and are defined in `opentelemetry-semantic-conventions-ai`. The `db.system` attribute is set to `"azure.ai_search"` on every span.

| Constant | Attribute | Description |
|----------|-----------|-------------|
| `AZURE_AI_SEARCH_INDEX_NAME` | `azure.search.index_name` | Search index targeted by the operation |
| `AZURE_AI_SEARCH_SEARCH_TEXT` | `azure.search.search.text` | Full-text search query string |
| `AZURE_AI_SEARCH_SEARCH_TOP` | `azure.search.search.top` | Maximum number of results requested |
| `AZURE_AI_SEARCH_SEARCH_SKIP` | `azure.search.search.skip` | Number of results to skip (pagination offset) |
| `AZURE_AI_SEARCH_SEARCH_FILTER` | `azure.search.search.filter` | OData filter expression |
| `AZURE_AI_SEARCH_SEARCH_QUERY_TYPE` | `azure.search.search.query_type` | Query type (`simple`, `full`, `semantic`) |
| `AZURE_AI_SEARCH_DOCUMENT_COUNT` | `azure.search.document.count` | Number of documents in the batch operation |
| `AZURE_AI_SEARCH_DOCUMENT_KEY` | `azure.search.document.key` | Document key for single-document retrieval |
| `AZURE_AI_SEARCH_SUGGESTER_NAME` | `azure.search.suggester_name` | Suggester name for autocomplete/suggest operations |
| `AZURE_AI_SEARCH_ANALYZER_NAME` | `azure.search.analyzer_name` | Analyzer name used in text analysis |
| `AZURE_AI_SEARCH_SEARCH_RESULTS_COUNT` | `azure.search.search.results_count` | Total number of results returned by a search |
| `AZURE_AI_SEARCH_DOCUMENT_SUCCEEDED_COUNT` | `azure.search.document.succeeded_count` | Documents successfully indexed in a batch |
| `AZURE_AI_SEARCH_DOCUMENT_FAILED_COUNT` | `azure.search.document.failed_count` | Documents that failed indexing in a batch |
| `AZURE_AI_SEARCH_AUTOCOMPLETE_RESULTS_COUNT` | `azure.search.autocomplete.results_count` | Number of autocomplete suggestions returned |
| `AZURE_AI_SEARCH_SUGGEST_RESULTS_COUNT` | `azure.search.suggest.results_count` | Number of search suggestions returned |
| `AZURE_AI_SEARCH_DB_SYSTEM_NAME` | *(value)* `"azure.ai_search"` | Value used for `db.system` on every span |
