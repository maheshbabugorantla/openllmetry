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

## Instrumented Operations

### SearchClient

Both sync (`azure.search.documents.SearchClient`) and async (`azure.search.documents.aio.SearchClient`) variants are instrumented.

| Method | Span Name |
|--------|-----------|
| `search()` | `azure.search.search` |
| `get_document()` | `azure.search.get_document` |
| `get_document_count()` | `azure.search.get_document_count` |
| `upload_documents()` | `azure.search.upload_documents` |
| `merge_documents()` | `azure.search.merge_documents` |
| `delete_documents()` | `azure.search.delete_documents` |
| `merge_or_upload_documents()` | `azure.search.merge_or_upload_documents` |
| `index_documents()` | `azure.search.index_documents` |
| `autocomplete()` | `azure.search.autocomplete` |
| `suggest()` | `azure.search.suggest` |

### SearchIndexClient

Both sync (`azure.search.documents.indexes.SearchIndexClient`) and async (`azure.search.documents.indexes.aio.SearchIndexClient`) variants are instrumented.

| Method | Span Name |
|--------|-----------|
| `create_index()` | `azure.search.create_index` |
| `create_or_update_index()` | `azure.search.create_or_update_index` |
| `delete_index()` | `azure.search.delete_index` |
| `get_index()` | `azure.search.get_index` |
| `list_indexes()` | `azure.search.list_indexes` |
| `list_index_names()` | `azure.search.list_index_names` |
| `get_index_statistics()` | `azure.search.get_index_statistics` |
| `get_service_statistics()` | `azure.search.get_service_statistics` |
| `analyze_text()` | `azure.search.analyze_text` |

## Span Attributes

Every span includes:

| Attribute | Description |
|-----------|-------------|
| `db.system` | Always `"azure.ai_search"` |
| `db.operation` | Method name (e.g. `search`, `upload_documents`) |
| `db.system` | Index hostname (from client endpoint) |

Operation-specific attributes:

| Constant | Attribute | Set On |
|----------|-----------|--------|
| `AZURE_AI_SEARCH_INDEX_NAME` | `azure.search.index_name` | All operations |
| `AZURE_AI_SEARCH_SEARCH_TEXT` | `azure.search.search.text` | `search`, `autocomplete`, `suggest` |
| `AZURE_AI_SEARCH_SEARCH_TOP` | `azure.search.search.top` | `search` |
| `AZURE_AI_SEARCH_SEARCH_SKIP` | `azure.search.search.skip` | `search` |
| `AZURE_AI_SEARCH_SEARCH_FILTER` | `azure.search.search.filter` | `search` |
| `AZURE_AI_SEARCH_SEARCH_QUERY_TYPE` | `azure.search.search.query_type` | `search` |
| `AZURE_AI_SEARCH_DOCUMENT_COUNT` | `azure.search.document.count` | `upload_documents`, `merge_documents`, `delete_documents`, `merge_or_upload_documents`, `index_documents` |
| `AZURE_AI_SEARCH_DOCUMENT_KEY` | `azure.search.document.key` | `get_document` |
| `AZURE_AI_SEARCH_SUGGESTER_NAME` | `azure.search.suggester_name` | `autocomplete`, `suggest` |
| `AZURE_AI_SEARCH_SEARCH_RESULTS_COUNT` | `azure.search.search.results_count` | `search` (response) |
| `AZURE_AI_SEARCH_DOCUMENT_SUCCEEDED_COUNT` | `azure.search.document.succeeded_count` | Batch indexing (response) |
| `AZURE_AI_SEARCH_DOCUMENT_FAILED_COUNT` | `azure.search.document.failed_count` | Batch indexing (response) |
| `AZURE_AI_SEARCH_AUTOCOMPLETE_RESULTS_COUNT` | `azure.search.autocomplete.results_count` | `autocomplete` (response) |
| `AZURE_AI_SEARCH_SUGGEST_RESULTS_COUNT` | `azure.search.suggest.results_count` | `suggest` (response) |
| `AZURE_AI_SEARCH_ANALYZER_NAME` | `azure.search.analyzer_name` | `analyze_text` |
| `AZURE_AI_SEARCH_SERVICE_DOCUMENT_COUNT` | `azure.search.service.document_count` | `get_service_statistics` (response) |
| `AZURE_AI_SEARCH_SERVICE_INDEX_COUNT` | `azure.search.service.index_count` | `get_service_statistics` (response) |

## Async Usage

The async `SearchClient` and `SearchIndexClient` are instrumented identically to their sync counterparts. Use them in any `async` context:

```python
import asyncio
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

AzureSearchInstrumentor().instrument()

async def main():
    async with SearchClient(
        endpoint="https://my-search.search.windows.net",
        index_name="hotels",
        credential=AzureKeyCredential("api-key"),
    ) as client:
        results = client.search(search_text="luxury")
        async for result in results:
            print(result["hotel_name"])

    async with SearchIndexClient(
        endpoint="https://my-search.search.windows.net",
        credential=AzureKeyCredential("api-key"),
    ) as index_client:
        stats = await index_client.get_service_statistics()
        print(stats)

asyncio.run(main())
```

## Example

```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Configure tracing
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Initialize instrumentation
AzureSearchInstrumentor().instrument(tracer_provider=provider)

# Create client
client = SearchClient(
    endpoint="https://my-search.search.windows.net",
    index_name="hotels",
    credential=AzureKeyCredential("api-key"),
)

# Operations are automatically traced
results = client.search(search_text="luxury hotel", filter="rating ge 4", top=10)
for result in results:
    print(result["hotel_name"])

count = client.get_document_count()
print(f"Total documents: {count}")

client.upload_documents(documents=[{"hotel_id": "1", "hotel_name": "Grand Hotel"}])
```

## Developer Guide

For contributors and developers looking to extend this instrumentation:

📖 **[Span Attribute Extraction Guide](docs/SPAN_ATTRIBUTES_GUIDE.md)**

This guide covers the architecture, extraction function patterns, how to add new SDK methods, and testing strategies.
