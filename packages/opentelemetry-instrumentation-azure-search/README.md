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

**Index management:**

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

**Synonym maps:**

| Method | Span Name |
|--------|-----------|
| `create_synonym_map()` | `azure.search.create_synonym_map` |
| `create_or_update_synonym_map()` | `azure.search.create_or_update_synonym_map` |
| `delete_synonym_map()` | `azure.search.delete_synonym_map` |
| `get_synonym_map()` | `azure.search.get_synonym_map` |
| `get_synonym_maps()` | `azure.search.get_synonym_maps` |
| `get_synonym_map_names()` | `azure.search.get_synonym_map_names` |

### SearchIndexerClient

Both sync (`azure.search.documents.indexes.SearchIndexerClient`) and async (`azure.search.documents.indexes.aio.SearchIndexerClient`) variants are instrumented.

**Indexer management:**

| Method | Span Name |
|--------|-----------|
| `create_indexer()` | `azure.search.create_indexer` |
| `create_or_update_indexer()` | `azure.search.create_or_update_indexer` |
| `delete_indexer()` | `azure.search.delete_indexer` |
| `get_indexer()` | `azure.search.get_indexer` |
| `get_indexers()` | `azure.search.get_indexers` |
| `get_indexer_names()` | `azure.search.get_indexer_names` |
| `run_indexer()` | `azure.search.run_indexer` |
| `reset_indexer()` | `azure.search.reset_indexer` |
| `get_indexer_status()` | `azure.search.get_indexer_status` |

**Data source management:**

| Method | Span Name |
|--------|-----------|
| `create_data_source_connection()` | `azure.search.create_data_source_connection` |
| `create_or_update_data_source_connection()` | `azure.search.create_or_update_data_source_connection` |
| `delete_data_source_connection()` | `azure.search.delete_data_source_connection` |
| `get_data_source_connection()` | `azure.search.get_data_source_connection` |
| `get_data_source_connections()` | `azure.search.get_data_source_connections` |
| `get_data_source_connection_names()` | `azure.search.get_data_source_connection_names` |

**Skillset management:**

| Method | Span Name |
|--------|-----------|
| `create_skillset()` | `azure.search.create_skillset` |
| `create_or_update_skillset()` | `azure.search.create_or_update_skillset` |
| `delete_skillset()` | `azure.search.delete_skillset` |
| `get_skillset()` | `azure.search.get_skillset` |
| `get_skillsets()` | `azure.search.get_skillsets` |
| `get_skillset_names()` | `azure.search.get_skillset_names` |

### SearchIndexingBufferedSender

Both sync (`azure.search.documents.SearchIndexingBufferedSender`) and async (`azure.search.documents.aio.SearchIndexingBufferedSender`) variants are instrumented.

| Method | Span Name |
|--------|-----------|
| `upload_documents()` | `azure.search.upload_documents` |
| `delete_documents()` | `azure.search.delete_documents` |
| `merge_documents()` | `azure.search.merge_documents` |
| `merge_or_upload_documents()` | `azure.search.merge_or_upload_documents` |
| `index_documents()` | `azure.search.index_documents` |
| `flush()` | `azure.search.flush` |

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
| `AZURE_AI_SEARCH_VECTOR_QUERIES_COUNT` | `azure.search.search.vector_queries_count` | `search` with `vector_queries` |
| `AZURE_AI_SEARCH_VECTOR_FIELDS` | `azure.search.search.vector_fields` | `search` with `vector_queries` |
| `AZURE_AI_SEARCH_VECTOR_K_NEAREST_NEIGHBORS` | `azure.search.search.k_nearest_neighbors` | `search` with `vector_queries` |
| `AZURE_AI_SEARCH_VECTOR_QUERY_KIND` | `azure.search.search.vector_query_kind` | `search` with `vector_queries` |
| `AZURE_AI_SEARCH_VECTOR_WEIGHT` | `azure.search.search.vector_weight` | `search` with `vector_queries` |
| `AZURE_AI_SEARCH_VECTOR_OVERSAMPLING` | `azure.search.search.vector_oversampling` | `search` with `vector_queries` |
| `AZURE_AI_SEARCH_VECTOR_FILTER_MODE` | `azure.search.search.vector_filter_mode` | `search` with `vector_queries` |
| `AZURE_AI_SEARCH_VECTOR_EXHAUSTIVE` | `azure.search.search.vector_exhaustive` | `search` with `vector_queries` |
| `AZURE_AI_SEARCH_SEMANTIC_CONFIGURATION_NAME` | `azure.search.search.semantic_configuration_name` | `search` with semantic configuration |
| `AZURE_AI_SEARCH_QUERY_CAPTION` | `azure.search.search.query_caption` | `search` with semantic search |
| `AZURE_AI_SEARCH_QUERY_ANSWER` | `azure.search.search.query_answer` | `search` with semantic search |
| `AZURE_AI_SEARCH_SEARCH_MODE` | `azure.search.search.search_mode` | `search` |
| `AZURE_AI_SEARCH_SCORING_PROFILE` | `azure.search.search.scoring_profile` | `search` |
| `AZURE_AI_SEARCH_SELECT` | `azure.search.search.select` | `search` |
| `AZURE_AI_SEARCH_SEARCH_FIELDS` | `azure.search.search.search_fields` | `search` |
| `AZURE_AI_SEARCH_FACETS` | `azure.search.search.facets` | `search` |
| `AZURE_AI_SEARCH_ORDER_BY` | `azure.search.search.order_by` | `search` |
| `AZURE_AI_SEARCH_SYNONYM_MAP_NAME` | `azure.search.synonym_map.name` | `create/get/delete_synonym_map` |
| `AZURE_AI_SEARCH_SYNONYM_MAP_SYNONYMS_COUNT` | `azure.search.synonym_map.synonyms_count` | `create/update_synonym_map` (response) |
| `AZURE_AI_SEARCH_INDEXER_NAME` | `azure.search.indexer_name` | Indexer management methods |
| `AZURE_AI_SEARCH_DATA_SOURCE_NAME` | `azure.search.data_source_name` | Data source management methods |
| `AZURE_AI_SEARCH_SKILLSET_NAME` | `azure.search.skillset_name` | Skillset management methods |
| `AZURE_AI_SEARCH_INDEXER_STATUS` | `azure.search.indexer.status` | `get_indexer_status` (response) |
| `AZURE_AI_SEARCH_DOCUMENTS_PROCESSED` | `azure.search.indexer.documents_processed` | `get_indexer_status` (response) |
| `AZURE_AI_SEARCH_DOCUMENTS_FAILED` | `azure.search.indexer.documents_failed` | `get_indexer_status` (response) |
| `AZURE_AI_SEARCH_DATA_SOURCE_TYPE` | `azure.search.data_source.type` | Data source management (response) |
| `AZURE_AI_SEARCH_SKILLSET_SKILL_COUNT` | `azure.search.skillset.skill_count` | Skillset management (response) |

## Content Capture

By default, request and response content (documents, autocomplete suggestions, vector embeddings) is captured as **indexed span attributes** — e.g., `db.query.result.document.0`, `db.search.result.entity.0`. This follows the same pattern as LLM instrumentations (`gen_ai.prompt.0.content`) and ensures content is visible in APM backends like Elastic APM.

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TRACELOOP_TRACE_CONTENT` | `true` | Enable/disable content capture |
| `TRACELOOP_TRACE_CONTENT_MAX_ITEMS` | `100` | Max items captured per span (documents, suggestions, etc.) |
| `TRACELOOP_TRACE_CONTENT_MAX_LENGTH` | `16384` | Max characters per serialized content attribute |

Accepted truthy values for `TRACELOOP_TRACE_CONTENT`: `true`, `1`, `yes`, `on` (case-insensitive).

### Per-Request Override

```python
from opentelemetry import context as context_api

ctx = context_api.set_value("override_enable_content_tracing", True)
token = context_api.attach(ctx)
try:
    result = client.get_document(key="hotel-1")
finally:
    context_api.detach(token)
```

### What Content is Captured

| Operation | Attribute Pattern | Content |
|-----------|------------------|---------|
| `get_document()` | `db.query.result.document` | Full document JSON |
| `autocomplete()` | `db.search.result.entity.{i}` | Each suggestion (text + query_plus_text) |
| `suggest()` | `db.search.result.entity.{i}` | Each suggestion item JSON |
| `upload/merge/delete_documents()` | `db.query.result.document.{i}` (request) | Each input document |
| `upload/merge/delete_documents()` | `db.query.result.id.{i}`, `db.query.result.metadata.{i}` (response) | Document key + result metadata |
| `index_documents()` | `db.query.result.document.{i}` (request) | Each batch action |
| `index_documents()` | `db.query.result.id.{i}`, `db.query.result.metadata.{i}` (response) | Document key + result metadata |
| `search()` with `vector_queries` | `db.search.embeddings.vector.{i}` | Vector or text from each vector query |

> **Note:** `search()` result documents are not captured because `SearchItemPaged` is a lazy iterator — consuming it would break user code.

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
