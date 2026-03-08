# Span Attribute Extraction Guide

This guide explains how span attributes are extracted in the Azure AI Search instrumentation, how to add new SDK methods, and the patterns used throughout the codebase.

---

## Architecture Overview

The instrumentation is split across three modules:

| Module | Responsibility |
|--------|---------------|
| `__init__.py` | Method registry (`SEARCH_CLIENT_METHODS`), `AzureSearchInstrumentor` class |
| `wrapper.py` | `_sync_wrap`, extraction functions, response functions |
| `utils.py` | `dont_throw` decorator, `should_send_content()` stub |

### Module: `__init__.py`

Contains one method list:

```python
SEARCH_CLIENT_METHODS = [
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "search",
        "span_name": "azure.search.search",
    },
    # ... 9 more methods
]
```

`_instrument()` iterates `WRAPPED_METHODS` (= `SEARCH_CLIENT_METHODS`) and calls `wrap_function_wrapper` for each entry.

### Module: `wrapper.py`

**Frozensets for O(1) dispatch:**

```python
_DOCUMENT_BATCH_METHODS = frozenset({
    "upload_documents",
    "merge_documents",
    "delete_documents",
    "merge_or_upload_documents",
})

_SUGGESTION_METHODS = frozenset({"autocomplete", "suggest"})
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

### Module: `utils.py`

**`dont_throw` decorator** — wraps attribute extraction functions so that any exception is logged instead of propagating. Works for both sync and async functions (detects via `asyncio.iscoroutinefunction`).

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

All extractors are decorated with `@dont_throw` — if extraction fails, the span is still created.

### Response Attribute Extractors

| Function | Triggered By | Attributes Set |
|----------|-------------|----------------|
| `_set_search_response_attributes` | `search` | `search.results_count` (via `response.get_count()`) |
| `_set_document_count_response_attributes` | `get_document_count` | `document.count` |
| `_set_autocomplete_response_attributes` | `autocomplete` | `autocomplete.results_count` |
| `_set_suggest_response_attributes` | `suggest` | `suggest.results_count` |
| `_set_indexing_response_single_pass` | Called by batch handlers | `document.succeeded_count`, `document.failed_count` |
| `_set_document_batch_response_all` | `upload/merge/delete_documents`, `merge_or_upload_documents` | Delegates to single-pass counter |
| `_set_index_documents_response_all` | `index_documents` | Delegates to single-pass counter |

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

## Notes

- **Content capture** (`should_send_content`) is stubbed to `False` in this release. Full content capture (request/response documents as indexed span attributes) will be added in a future release.
- **No async support yet.** All methods are synchronous. Async instrumentation (`_async_wrap`) will be added in a future release.
- **Tests** use `MockSearchClient` with manual span creation rather than full SDK wrapping. VCR cassettes are not needed for unit tests; they are used for integration-level cassette tests.
