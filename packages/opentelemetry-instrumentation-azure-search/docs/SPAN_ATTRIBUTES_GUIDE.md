# Span Attribute Extraction Guide

This guide explains how span attributes are extracted in the Azure AI Search instrumentation, how to add new SDK methods, and the patterns used throughout the codebase.

---

## Architecture Overview

The instrumentation is split across three modules:

| Module | Responsibility |
|--------|---------------|
| `__init__.py` | Method registries (8 lists, 104 methods), `AzureSearchInstrumentor` class |
| `wrapper.py` | `_sync_wrap`, `_async_wrap`, extraction functions, response functions |
| `utils.py` | `dont_throw` decorator (async-aware), `should_send_content()` stub |

### Module: `__init__.py`

Contains eight method lists:

```python
SEARCH_CLIENT_METHODS              # 10 sync SearchClient methods
ASYNC_SEARCH_CLIENT_METHODS        # 10 async SearchClient methods (aio)
SEARCH_INDEX_CLIENT_METHODS        # 15 sync SearchIndexClient methods (incl. synonym maps)
ASYNC_SEARCH_INDEX_CLIENT_METHODS  # 15 async SearchIndexClient methods (aio)
SEARCH_INDEXER_CLIENT_METHODS      # 21 sync SearchIndexerClient methods
ASYNC_SEARCH_INDEXER_CLIENT_METHODS # 21 async SearchIndexerClient methods (aio)
BUFFERED_SENDER_METHODS            # 6 sync SearchIndexingBufferedSender methods
ASYNC_BUFFERED_SENDER_METHODS      # 6 async SearchIndexingBufferedSender methods (aio)
```

`WRAPPED_METHODS` is the concatenation of all eight lists (104 entries total). `_instrument()` iterates it and calls `wrap_function_wrapper` for each entry.

**Method entry format:**

```python
{
    "module": "azure.search.documents.indexes",
    "object": "SearchIndexClient",
    "method": "create_index",
    "span_name": "azure.search.create_index",
}
```

For async variants, `"module"` becomes `"azure.search.documents.indexes.aio"`.

`_instrument()` iterates `WRAPPED_METHODS` and calls `wrap_function_wrapper` for each entry.

### Module: `wrapper.py`

**Frozensets for O(1) dispatch:**

```python
_DOCUMENT_BATCH_METHODS = frozenset({
    "upload_documents", "merge_documents",
    "delete_documents", "merge_or_upload_documents",
})
_SUGGESTION_METHODS = frozenset({"autocomplete", "suggest"})
_INDEX_MANAGEMENT_METHODS = frozenset({
    "create_index", "create_or_update_index", "delete_index",
    "get_index", "list_indexes", "list_index_names",
    "get_index_statistics", "analyze_text",
})
_INDEXER_MANAGEMENT_METHODS = frozenset({
    "create_indexer", "create_or_update_indexer", "delete_indexer",
    "get_indexer", "get_indexers", "run_indexer",
    "reset_indexer", "get_indexer_status",
})
_DATA_SOURCE_METHODS = frozenset({
    "create_data_source_connection", "create_or_update_data_source_connection",
    "delete_data_source_connection", "get_data_source_connection",
    "get_data_source_connections",
})
_SKILLSET_METHODS = frozenset({
    "create_skillset", "create_or_update_skillset", "delete_skillset",
    "get_skillset", "get_skillsets",
})
_SYNONYM_MAP_METHODS = frozenset({
    "create_synonym_map", "create_or_update_synonym_map",
    "delete_synonym_map", "get_synonym_map",
    "get_synonym_maps", "get_synonym_map_names",
})
```

These replace `if method == "x" or method == "y"` chains with `if method in _FROZENSET`.

**Main entry point — `_sync_wrap`:**

```python
def _sync_wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    name = to_wrap.get("span_name")
    method = to_wrap.get("method")

    with tracer.start_as_current_span(name, kind=SpanKind.CLIENT, ...) as span:
        span.set_attribute(OTelSpanAttributes.DB_SYSTEM, SpanAttributes.AZURE_AI_SEARCH_DB_SYSTEM_NAME)
        span.set_attribute(OTelSpanAttributes.DB_OPERATION, method)
        _set_request_attributes(span, method, instance, args, kwargs)

        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        _set_response_attributes(span, method, response, args, kwargs)

        if method in _DOCUMENT_BATCH_METHODS:
            _set_document_batch_response_all(span, response)
        elif method == "index_documents":
            _set_index_documents_response_all(span, response)

        span.set_status(Status(StatusCode.OK))
        return response
```

### Module: `wrapper.py` — `_async_wrap`

The async wrapper mirrors `_sync_wrap` but uses `await` for the underlying call and for async response methods:

```python
async def _async_wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    name = to_wrap.get("span_name")
    method = to_wrap.get("method")

    with tracer.start_as_current_span(name, kind=SpanKind.CLIENT, ...) as span:
        span.set_attribute(OTelSpanAttributes.DB_SYSTEM, ...)
        _set_request_attributes(span, method, instance, args, kwargs)

        try:
            response = await wrapped(*args, **kwargs)
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        # search response needs async variant — get_count() is a coroutine
        if method == "search":
            await _set_search_response_attributes_async(span, response)
        else:
            _set_response_attributes(span, method, response, args, kwargs)

        span.set_status(Status(StatusCode.OK))
        return response
```

**Key async gotcha:** `AsyncSearchItemPaged.get_count()` is a coroutine and must be `await`ed. `_set_search_response_attributes_async` is the async variant that does this correctly.

### Module: `utils.py`

**`dont_throw` decorator** — wraps attribute extraction functions so that any exception is logged instead of propagating. Detects async functions via `asyncio.iscoroutinefunction` and returns the appropriate wrapper:

```python
def dont_throw(func):
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
```

**`should_send_content()`** — always returns `False` in this release. Full content capture will be added in a future release.

---

## Extraction Functions

### Request Attribute Extractors

| Function | Triggered By | Attributes Set |
|----------|-------------|----------------|
| `_set_index_name_attribute` | All methods | `azure.search.index_name` |
| `_set_search_attributes` | `search` | `search.text`, `search.top`, `search.skip`, `search.filter`, `search.query_type` |
| `_set_get_document_attributes` | `get_document` | `document.key` |
| `_set_document_batch_attributes` | `upload/merge/delete_documents`, `merge_or_upload_documents` | `document.count` |
| `_set_index_documents_attributes` | `index_documents` | `document.count` (from `batch.actions`) |
| `_set_suggestion_attributes` | `autocomplete`, `suggest` | `search.text`, `suggester_name` |
| `_set_index_management_attributes` | `create/update/delete/get/list_index*` | `index_name` (positional or keyword) |
| `_set_analyze_text_attributes` | `analyze_text` | `index_name`, `analyzer_name` (from `AnalyzeTextOptions`) |
| `_set_vector_search_attributes` | `search` (when `vector_queries` present) | `vector_queries_count`, `vector_fields`, `k_nearest_neighbors`, `vector_query_kind`, `vector_weight`, `vector_oversampling`, `vector_filter_mode`, `vector_exhaustive` |
| `_set_semantic_search_attributes` | `search` (when semantic params present) | `semantic_configuration_name`, `query_caption`, `query_answer`, `search_mode`, `scoring_profile`, `select`, `search_fields`, `facets`, `order_by` |
| `_set_indexer_management_attributes` | Indexer management methods | `indexer_name` (from arg or object) |
| `_set_indexer_status_attributes` | `get_indexer_status` | `indexer_name`, `indexer.status`, `indexer.documents_processed`, `indexer.documents_failed` |
| `_set_data_source_attributes` | Data source management methods | `data_source_name`, `data_source.type` |
| `_set_skillset_attributes` | Skillset management methods | `skillset_name`, `skillset.skill_count` |
| `_set_synonym_map_attributes` | Synonym map methods | `synonym_map.name`, `synonym_map.synonyms_count` |

All extractors are decorated with `@dont_throw` — if extraction fails, the span is still created.

### Response Attribute Extractors

| Function | Triggered By | Attributes Set |
|----------|-------------|----------------|
| `_set_search_response_attributes` | `search` (sync) | `search.results_count` (via `response.get_count()`) |
| `_set_search_response_attributes_async` | `search` (async) | `search.results_count` (awaits `response.get_count()`) |
| `_set_document_count_response_attributes` | `get_document_count` | `document.count` |
| `_set_autocomplete_response_attributes` | `autocomplete` | `autocomplete.results_count` |
| `_set_suggest_response_attributes` | `suggest` | `suggest.results_count` |
| `_set_service_statistics_response_attributes` | `get_service_statistics` | `service.document_count`, `service.index_count` (via `_deep_get`) |
| `_set_indexing_response_single_pass` | Called by batch handlers | `document.succeeded_count`, `document.failed_count` |
| `_set_document_batch_response_all` | `upload/merge/delete_documents`, `merge_or_upload_documents` | Delegates to single-pass counter |
| `_set_index_documents_response_all` | `index_documents` | Delegates to single-pass counter |

**`_deep_get(obj, key)`** — traverses a nested object/dict by dot-separated key path:

```python
def _deep_get(obj, key):
    for part in key.split("."):
        if obj is None:
            return None
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj
```

Used to extract `counters.document_count` and `counters.index_count` from `ServiceStatistics` without hard-coding attribute access chains.

**Single-pass batch counting:**

```python
def _set_indexing_response_single_pass(span, results):
    succeeded = 0
    for result in results:
        if getattr(result, "succeeded", False):
            succeeded += 1
    failed = len(results) - succeeded
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_SUCCEEDED_COUNT, succeeded)
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_FAILED_COUNT, failed)
```

This counts succeeded and failed in a single O(n) pass instead of two list comprehensions.

---

## How to Add a New SDK Method

Follow these five steps:

### Step 1 — Define the span attribute constant

In `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py`, add to the `SpanAttributes` class:

```python
# Azure AI Search
AZURE_AI_SEARCH_MY_NEW_ATTR = "azure.search.my_new_attr"
```

Run `uv sync --reinstall-package opentelemetry-semantic-conventions-ai` in the instrumentation package to pick up changes.

### Step 2 — Register the method

In `__init__.py`, add an entry to the relevant method list:

```python
SEARCH_CLIENT_METHODS = [
    # ...existing entries...
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "my_new_method",
        "span_name": "azure.search.my_new_method",
    },
]
```

### Step 3 — Create an extraction function

In `wrapper.py`:

```python
@dont_throw
def _set_my_new_method_attributes(span, args, kwargs):
    value = kwargs.get("my_param") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_MY_NEW_ATTR, value)
```

### Step 4 — Route via frozenset or direct dispatch

Add the method to an existing frozenset (if it shares extraction logic) or add a branch in `_set_request_attributes`:

```python
def _set_request_attributes(span, method, instance, args, kwargs):
    _set_index_name_attribute(span, instance, args, kwargs)

    if method == "search":
        _set_search_attributes(span, args, kwargs)
    elif method == "my_new_method":              # ← add here
        _set_my_new_method_attributes(span, args, kwargs)
    # ...
```

### Step 5 — Add a test

In `tests/test_azure_search_instrumentation.py`, add a test using the `MockSearchClient` pattern:

```python
def test_my_new_method_span_attributes(tracer, exporter):
    client = MockSearchClient(tracer)
    client.my_new_method(my_param="value")

    spans = exporter.get_finished_spans()
    span = next(s for s in spans if s.name == "azure.search.my_new_method")
    assert span.attributes.get("azure.search.my_new_attr") == "value"
```

---

## Common Patterns

### Pattern 1 — Simple Query

For methods with a small number of well-known parameters:

```python
@dont_throw
def _set_search_attributes(span, args, kwargs):
    search_text = kwargs.get("search_text") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT, search_text)
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SEARCH_TOP, kwargs.get("top"))
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SEARCH_SKIP, kwargs.get("skip"))
```

Always check `kwargs.get(name)` first, then fall back to positional `args[i]`. This handles both `client.search("query")` and `client.search(search_text="query")`.

### Pattern 2 — Batch Documents

For methods that accept a `documents` list:

```python
@dont_throw
def _set_document_batch_attributes(span, args, kwargs):
    documents = kwargs.get("documents") or (args[0] if args else None)
    if documents and hasattr(documents, "__len__"):
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT, len(documents))
```

Use `hasattr(documents, "__len__")` rather than `isinstance(documents, list)` to handle any sequence type.

---

## Content Capture

### How it works

Content capture stores request/response payloads as **indexed span attributes** — e.g., `db.query.result.document.0`, `db.search.result.entity.1`. This mirrors the LLM pattern (`gen_ai.prompt.0.content`) and ensures content is indexed by APM backends like Elastic APM (which drops `span.add_event()` data).

### `should_send_content()`, `max_content_items()`, `max_content_length()`

These three helpers in `utils.py` are computed **once per span** at the top of `_sync_wrap`/`_async_wrap`:

```python
content_enabled = should_send_content()     # reads TRACELOOP_TRACE_CONTENT env var
max_items = max_content_items()             # reads TRACELOOP_TRACE_CONTENT_MAX_ITEMS
max_length = max_content_length()           # reads TRACELOOP_TRACE_CONTENT_MAX_LENGTH
```

They are then passed down to every content function — never re-read per item.

`should_send_content()` checks `override_enable_content_tracing` from OpenTelemetry context first, then the env var. Default is `True`.

### `_safe_json_dumps(obj, max_length)`

Serializes an object to JSON and truncates to `max_length` characters:

```python
def _safe_json_dumps(obj, max_length):
    try:
        s = json.dumps(obj, default=str)
    except Exception:
        s = str(obj)
    return s[:max_length] if max_length > 0 else s
```

### Content Dispatchers

Two top-level dispatchers route to operation-specific content functions:

| Dispatcher | Called From | Routes To |
|-----------|------------|-----------|
| `_set_request_content_attributes` | `_sync_wrap` / `_async_wrap` (before call) | `_set_search_vector_embeddings_attributes`, `_set_document_batch_request_content_attributes`, `_set_index_documents_request_content_attributes` |
| `_set_response_content_attributes` | `_sync_wrap` / `_async_wrap` (after call) | `_set_get_document_content_attribute`, `_set_autocomplete_content_attributes`, `_set_suggest_content_attributes`, `_set_document_batch_response_all`, `_set_index_documents_response_all` |

### Per-Operation Content Functions

| Function | Operation | Attribute Pattern |
|----------|-----------|------------------|
| `_set_search_vector_embeddings_attributes` | `search` (request) | `db.search.embeddings.vector.{i}` |
| `_set_document_batch_request_content_attributes` | `upload/merge/delete_documents` (request) | `db.query.result.document.{i}` |
| `_set_index_documents_request_content_attributes` | `index_documents` (request) | `db.query.result.document.{i}` |
| `_set_get_document_content_attribute` | `get_document` (response) | `db.query.result.document` |
| `_set_autocomplete_content_attributes` | `autocomplete` (response) | `db.search.result.entity.{i}` |
| `_set_suggest_content_attributes` | `suggest` (response) | `db.search.result.entity.{i}` |
| `_set_indexing_response_single_pass` (updated) | `upload/merge/delete/index_documents` (response) | `db.query.result.id.{i}`, `db.query.result.metadata.{i}` |

`_set_indexing_response_single_pass` was extended to accept `content_enabled`, `max_items`, and `max_length` parameters so it can write content attributes in the same single pass that counts succeeded/failed.

### `EventAttributes` Enum

`EventAttributes` in `semconv_ai` defines the canonical attribute name prefixes:

```python
class EventAttributes(Enum):
    DB_QUERY_RESULT_ID       = "db.query.result.id"
    DB_QUERY_RESULT_METADATA = "db.query.result.metadata"
    DB_QUERY_RESULT_DOCUMENT = "db.query.result.document"
    DB_SEARCH_EMBEDDINGS_VECTOR = "db.search.embeddings.vector"
    DB_SEARCH_RESULT_ENTITY  = "db.search.result.entity"
```

Use `.value` to get the string and append `.{i}` for indexed attributes:
```python
span.set_attribute(f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.{i}", content)
```

### Content Capture Best Practices

- Always check `content_enabled` before serializing — skip expensive `json.dumps` if not needed.
- Pass `max_items` and `max_length` as parameters; do not re-read env vars inside per-item loops.
- Use `_safe_json_dumps` for all attribute values — never raw `str(obj)` without truncation.
- `search()` result documents are intentionally NOT captured — consuming `SearchItemPaged` would exhaust the iterator before user code can iterate it.

---

## Pattern 3 — Index Management (SearchIndexClient)

For methods that take an index object as the first argument:

```python
@dont_throw
def _set_index_management_attributes(span, method, args, kwargs):
    # Methods like create_index(index=...) or delete_index(index_name=...)
    index_obj = kwargs.get("index") or (args[0] if args else None)
    if index_obj is not None:
        name = getattr(index_obj, "name", None) or index_obj
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME, name)
```

The `name` fallback handles both cases: an `Index` object (with `.name`) and a plain string (`delete_index("hotels")`).

## Pattern 4 — Complex Nested Parameters

For methods like `analyze_text` that use a request model:

```python
@dont_throw
def _set_analyze_text_attributes(span, args, kwargs):
    analyze_request = kwargs.get("analyze_request") or (args[1] if len(args) > 1 else None)
    if analyze_request:
        analyzer = getattr(analyze_request, "analyzer_name", None)
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_ANALYZER_NAME, analyzer)
```

Use `getattr(obj, "field", None)` to safely read from SDK model objects without isinstance checks.

---

## Notes

- **Content capture** (`should_send_content`) is stubbed to `False` in this release. Full content capture (request/response documents as indexed span attributes) will be added in a future release.
- **Async support** is available via `_async_wrap` for `SearchClient.aio` and `SearchIndexClient.aio` methods.
- **Content capture** is now active — controlled by `TRACELOOP_TRACE_CONTENT` (default: `true`). Use `TRACELOOP_TRACE_CONTENT_MAX_ITEMS` and `TRACELOOP_TRACE_CONTENT_MAX_LENGTH` to tune volume.
- **Tests** use `MockSearchClient` / `MockSearchIndexClient` with manual span creation rather than full SDK wrapping. VCR cassettes are not needed for unit tests; they are used for integration-level cassette tests.
