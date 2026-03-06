# Azure AI Search Instrumentation — PR Stacking Playbook

## Overview

You are implementing a 6-PR stacking strategy to land a complete OpenTelemetry instrumentation for Azure AI Search into the OpenLLMetry monorepo. The full feature lives on branch `mbg/traceloop-azure-ai-search-instrumentation` (tip commit `3d88ef3`). Your job is to cherry-pick and restructure the code from that branch into 6 sequential PRs, each targeting `main`.

**Source branch:** `mbg/traceloop-azure-ai-search-instrumentation`
**Total scope:** ~12,000 lines across 54 files, 4 client types, 98 wrapped methods, 42 span attributes

## Repository Conventions

- All packages use `uv` as the package manager. Run commands via `uv run <command>`.
- Tests use VCR cassettes for API calls. Run: `uv run pytest tests/`
- Nx orchestrates the workspace: `nx run <package>:test`, `nx run <package>:lint`
- Ruff is used for linting. Config is in each `pyproject.toml`.
- Follow existing instrumentation patterns (see `packages/opentelemetry-instrumentation-voyageai/` as reference).
- Use `wrapt.wrap_function_wrapper()` for method wrapping.
- Instrumentor classes extend `BaseInstrumentor` from `opentelemetry.instrumentation.instrumentor`.

## Source Code Reference

All code to be landed exists on the source branch. The key files are:

```
packages/opentelemetry-instrumentation-azure-search/
├── opentelemetry/instrumentation/azure_search/
│   ├── __init__.py        # Instrumentor class + method registries (739 lines)
│   ├── wrapper.py         # Span wrapping + attribute extraction (897 lines)
│   ├── utils.py           # dont_throw, content toggle, config helpers (91 lines)
│   ├── config.py          # Exception logger singleton (2 lines)
│   └── version.py         # Version string
├── tests/
│   ├── conftest.py        # Fixtures: exporter, VCR config, environment
│   ├── test_azure_search_instrumentation.py  # Unit tests (4,803 lines)
│   ├── test_azure_search_integration.py      # Integration tests (841 lines)
│   └── cassettes/         # 40+ VCR YAML files
├── pyproject.toml, project.json, .flake8, .python-version, uv.lock
├── README.md, docs/SPAN_ATTRIBUTES_GUIDE.md

packages/opentelemetry-semantic-conventions-ai/
└── opentelemetry/semconv_ai/__init__.py  # +58 AZURE_SEARCH_* constants (lines 249-305)

packages/traceloop-sdk/
├── traceloop/sdk/instruments.py           # +AZURE_SEARCH enum
├── traceloop/sdk/tracing/tracing.py       # +init_azure_search_instrumentor()
├── traceloop/sdk/utils/instrumentation_warnings.py  # New file
├── pyproject.toml                         # +azure-search optional dep
└── README.md                              # Updated instruments list

packages/sample-app/
├── pyproject.toml   # +azure-search-documents dependency
└── .env.example     # +AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY
```

---

## PR Dependency Graph

```
PR1 (scaffold + semconv)
 └─▶ PR2 (SearchClient core)
      └─▶ PR3 (async + IndexClient)
           └─▶ PR4 (vector/semantic + content capture)
                └─▶ PR5 (indexer, synonyms, buffered sender)
                     └─▶ PR6 (SDK integration, docs, integration tests)
```

Each PR builds on the previous. Branch off the prior PR's branch.

---

## PR 1: Semantic Conventions + Package Scaffold

**Branch name:** `azure-search/pr1-scaffold`
**PR Title:** `feat(semconv, azure-search): Add Azure AI Search semantic conventions and package scaffold`
**Complexity:** Small (~400-500 LOC)
**Review time:** ~15 min

### Business Value
> Establishes the foundational contract (span attribute names) that all future Azure Search observability will build on. Users who instrument manually can immediately use these constants. The package scaffold lets CI validate the build.

### What to Include

**1. Semantic conventions — add to `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py`:**

Add these 15 core constants to the `SpanAttributes` class (basic search + response attributes only):

```python
# Azure AI Search attributes
AZURE_SEARCH_INDEX_NAME = "azure_search.index_name"
AZURE_SEARCH_SEARCH_TEXT = "azure_search.search.text"
AZURE_SEARCH_SEARCH_TOP = "azure_search.search.top"
AZURE_SEARCH_SEARCH_SKIP = "azure_search.search.skip"
AZURE_SEARCH_SEARCH_FILTER = "azure_search.search.filter"
AZURE_SEARCH_SEARCH_QUERY_TYPE = "azure_search.search.query_type"
AZURE_SEARCH_DOCUMENT_COUNT = "azure_search.document.count"
AZURE_SEARCH_DOCUMENT_KEY = "azure_search.document.key"
AZURE_SEARCH_SUGGESTER_NAME = "azure_search.suggester_name"
AZURE_SEARCH_ANALYZER_NAME = "azure_search.analyzer_name"

# Azure AI Search response attributes
AZURE_SEARCH_SEARCH_RESULTS_COUNT = "azure_search.search.results_count"
AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT = "azure_search.document.succeeded_count"
AZURE_SEARCH_DOCUMENT_FAILED_COUNT = "azure_search.document.failed_count"
AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT = "azure_search.autocomplete.results_count"
AZURE_SEARCH_SUGGEST_RESULTS_COUNT = "azure_search.suggest.results_count"
```

**2. Package scaffold — create `packages/opentelemetry-instrumentation-azure-search/`:**

- `pyproject.toml` — Full package definition (version `0.51.1`, python `>=3.10,<4`, entry point for `azure_search`)
- `project.json` — Nx config (test, lint targets)
- `.flake8`, `.python-version`
- `opentelemetry/instrumentation/azure_search/__init__.py` — Skeleton instrumentor:
  - `_instruments = ("azure-search-documents >= 11.0.0",)`
  - `AzureSearchInstrumentor(BaseInstrumentor)` with:
    - `instrumentation_dependencies()` → returns `_instruments`
    - `_instrument(**kwargs)` → pass (no-op)
    - `_uninstrument(**kwargs)` → pass (no-op)
- `opentelemetry/instrumentation/azure_search/version.py` — `__version__ = "0.51.1"`
- `opentelemetry/instrumentation/azure_search/config.py` — `class Config: exception_logger = None`
- `tests/__init__.py` — empty
- `tests/conftest.py` — Fixtures (exporter with InMemorySpanExporter, clear_exporter)
- `uv.lock` — Generated lock file

**3. Unit tests in `tests/test_azure_search_instrumentation.py`:**

```python
class TestSemanticConventions:
    """Verify all Azure Search span attribute constants resolve correctly."""
    def test_azure_search_index_name_constant(self):
        assert SpanAttributes.AZURE_SEARCH_INDEX_NAME == "azure_search.index_name"
    # ... one test per constant (15 tests)

class TestInstrumentorLifecycle:
    """Verify instrumentor can be instantiated and has correct dependencies."""
    def test_instrumentation_dependencies(self):
        instrumentor = AzureSearchInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert "azure-search-documents >= 11.0.0" in deps

    def test_instrument_uninstrument_noop(self):
        instrumentor = AzureSearchInstrumentor()
        instrumentor.instrument()
        instrumentor.uninstrument()  # Should not raise
```

### What NOT to Include
- No wrapper.py yet
- No utils.py yet
- No method registries (SEARCH_CLIENT_METHODS, etc.)
- No SDK integration
- No README or docs

---

## PR 2: Core SearchClient Instrumentation

**Branch name:** `azure-search/pr2-search-client`
**PR Title:** `feat(azure-search): Instrument SearchClient with sync search, document ops, and span attributes`
**Complexity:** Medium (~1,200-1,500 LOC)
**Review time:** ~25 min

### Business Value
> Users can now trace their most common Azure Search operations — full-text search queries and document CRUD — seeing query parameters, result counts, and errors in their OpenTelemetry dashboards. This is the bread-and-butter of Azure Search usage.

### What to Include

**1. Method registry in `__init__.py`:**

Add `SEARCH_CLIENT_METHODS` list (10 sync methods for `azure.search.documents.SearchClient`):
- search, get_document, get_document_count
- upload_documents, merge_documents, delete_documents, merge_or_upload_documents, index_documents
- autocomplete, suggest

Wire up `_instrument()` to loop over methods and call `wrap_function_wrapper()`.
Wire up `_uninstrument()` to call `unwrap()` on each.

**2. Create `wrapper.py` with core infrastructure:**

Module-level frozensets:
- `_DOCUMENT_BATCH_METHODS` = frozenset({"upload_documents", "merge_documents", "delete_documents", "merge_or_upload_documents"})
- `_SUGGESTION_METHODS` = frozenset({"autocomplete", "suggest"})

Core wrapper functions:
- `_set_span_attribute(span, name, value)` — Null/empty guard
- `_with_tracer_wrapper(func)` — Decorator providing tracer to wrapper
- `_wrap(tracer, to_wrap, wrapped, instance, args, kwargs)` — Entry point (sync only for now; call `_sync_wrap`)
- `_sync_wrap(tracer, to_wrap, wrapped, instance, args, kwargs)` — Creates span (SpanKind.CLIENT), sets `VECTOR_DB_VENDOR: "Azure AI Search"`, calls `_set_request_attributes`, executes wrapped function, calls `_set_response_attributes`, handles errors

Request attribute dispatch (`_set_request_attributes`):
- `_set_index_name_attribute(span, instance, args, kwargs)`
- `_set_search_attributes(span, args, kwargs)` — search_text, top, skip, filter, query_type (NO vector/semantic yet)
- `_set_get_document_attributes(span, args, kwargs)` — document key
- `_set_document_batch_attributes(span, args, kwargs)` — document count
- `_set_index_documents_attributes(span, args, kwargs)` — batch action count
- `_set_suggestion_attributes(span, args, kwargs)` — search_text, suggester_name

Response attribute dispatch (`_set_response_attributes`):
- `_set_search_response_attributes(span, response)` — results_count from get_count() (sync only)
- `_set_document_count_response_attributes(span, response)` — int count
- `_set_autocomplete_response_attributes(span, response)` — list length
- `_set_suggest_response_attributes(span, response)` — list length
- `_set_document_batch_response_all(span, response, ...)` — succeeded/failed counts
- `_set_index_documents_response_all(span, response, ...)` — succeeded/failed counts
- `_set_indexing_response_single_pass(span, results, ...)` — Single-pass helper (content_enabled=False always in this PR)

**3. Create `utils.py` with:**
- `dont_throw(func)` — Sync-only decorator that logs exceptions instead of raising
- `should_send_content()` → return `False` (stub, content capture comes in PR4)

**4. Unit tests in `test_azure_search_instrumentation.py`:**

Use mocked Azure SDK classes (MockSearchClient, etc.) with `sys.modules` patching.

```python
class TestSearchClientInstrumentation:
    def test_search_creates_span(self): ...
    def test_get_document_creates_span(self): ...
    def test_get_document_count_creates_span(self): ...
    def test_upload_documents_creates_span(self): ...
    def test_merge_documents_creates_span(self): ...
    def test_delete_documents_creates_span(self): ...
    def test_merge_or_upload_documents_creates_span(self): ...
    def test_index_documents_creates_span(self): ...
    def test_autocomplete_creates_span(self): ...
    def test_suggest_creates_span(self): ...

class TestSearchAttributes:
    def test_search_text_extracted(self): ...
    def test_search_top_skip_filter(self): ...
    def test_query_type_enum_handled(self): ...
    def test_document_key_extracted(self): ...
    def test_document_batch_count(self): ...

class TestResponseAttributes:
    def test_search_results_count(self): ...
    def test_document_count_response(self): ...
    def test_autocomplete_results_count(self): ...
    def test_suggest_results_count(self): ...
    def test_document_batch_succeeded_failed(self): ...

class TestErrorHandling:
    def test_exception_sets_error_status(self): ...
    def test_exception_is_reraised(self): ...
    def test_suppression_key_skips_instrumentation(self): ...

class TestDontThrow:
    def test_swallows_attribute_extraction_errors(self): ...
    def test_logs_error_on_failure(self): ...
```

### What NOT to Include
- No async methods (ASYNC_SEARCH_CLIENT_METHODS)
- No SearchIndexClient, SearchIndexerClient, BufferedSender
- No vector/semantic search attributes
- No content capture (should_send_content always returns False)
- Content-related code paths in `_sync_wrap` should be present but inert (since `should_send_content()` returns False)

---

## PR 3: Async Support + SearchIndexClient

**Branch name:** `azure-search/pr3-async-index-client`
**PR Title:** `feat(azure-search): Add async instrumentation and SearchIndexClient index management`
**Complexity:** Medium (~800-1,000 LOC)
**Review time:** ~20 min

### Business Value
> Users running async Azure Search applications (FastAPI, aiohttp, etc.) now get full tracing without blocking their event loop. Index management operations become observable, helping DevOps teams debug index creation, schema changes, and capacity issues in production.

### What to Include

**1. Semantic conventions — add to `SpanAttributes`:**

```python
# Indexer operations
AZURE_SEARCH_INDEXER_NAME = "azure_search.indexer_name"
AZURE_SEARCH_DATA_SOURCE_NAME = "azure_search.data_source_name"
AZURE_SEARCH_SKILLSET_NAME = "azure_search.skillset_name"
AZURE_SEARCH_INDEXER_STATUS = "azure_search.indexer.status"
AZURE_SEARCH_DOCUMENTS_PROCESSED = "azure_search.indexer.documents_processed"
AZURE_SEARCH_DOCUMENTS_FAILED = "azure_search.indexer.documents_failed"
AZURE_SEARCH_DATA_SOURCE_TYPE = "azure_search.data_source.type"
AZURE_SEARCH_SKILLSET_SKILL_COUNT = "azure_search.skillset.skill_count"
```

**2. Method registries in `__init__.py`:**

Add:
- `ASYNC_SEARCH_CLIENT_METHODS` — 10 async SearchClient methods (`azure.search.documents.aio`)
- `SEARCH_INDEX_CLIENT_METHODS` — 9 sync methods (`azure.search.documents.indexes.SearchIndexClient`) — index CRUD + analyze_text + get_service_statistics + list_index_names (NO synonym maps yet)
- `ASYNC_SEARCH_INDEX_CLIENT_METHODS` — 9 async index client methods (NO synonym maps yet)
- Update `WRAPPED_METHODS` tuple to include all 4 lists

**3. Wrapper changes:**

Add frozenset:
- `_INDEX_MANAGEMENT_METHODS` = frozenset({"create_index", "create_or_update_index", "delete_index", "get_index", "get_index_statistics"})

Add to `_wrap()`:
- Runtime detection: `if asyncio.iscoroutinefunction(wrapped): return _async_wrap(...)`

Add `_async_wrap()`:
- Mirror of `_sync_wrap` but with `await wrapped(*args, **kwargs)`
- Special handling: `await _set_search_response_attributes_async(span, response)` for async search

Add attribute functions:
- `_set_index_management_attributes(span, method, args, kwargs)` — Extract index name from object or string arg
- `_set_analyze_text_attributes(span, args, kwargs)` — index_name + analyzer_name (handles enum values)
- `_set_search_response_attributes_async(span, response)` — Async version that awaits `get_count()`
- `_set_service_statistics_response_attributes(span, response)` — (stub, response handling comes with service stats, but the method is in the list)
- `_deep_get(obj, key)` — Helper for dict/object access

Update `_set_response_attributes` dispatch to include:
- `get_service_statistics` → `_set_service_statistics_response_attributes`

**4. Utils changes:**

Make `dont_throw` async-aware:
```python
def dont_throw(func):
    # ... existing sync logic ...
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
```

**5. Unit tests:**

```python
class TestAsyncInstrumentation:
    @pytest.mark.asyncio
    async def test_async_search_creates_span(self): ...
    @pytest.mark.asyncio
    async def test_async_error_handling(self): ...
    @pytest.mark.asyncio
    async def test_async_search_results_count(self): ...

class TestSearchIndexClientInstrumentation:
    def test_create_index_creates_span(self): ...
    def test_create_or_update_index_span(self): ...
    def test_delete_index_span(self): ...
    def test_get_index_span(self): ...
    def test_list_indexes_span(self): ...
    def test_get_index_statistics_span(self): ...
    def test_analyze_text_span(self): ...
    def test_get_service_statistics_span(self): ...

class TestIndexManagementAttributes:
    def test_index_name_from_object(self): ...
    def test_index_name_from_string(self): ...
    def test_analyze_text_analyzer_enum(self): ...

class TestAsyncDontThrow:
    @pytest.mark.asyncio
    async def test_async_dont_throw_swallows_errors(self): ...
```

### What NOT to Include
- No SearchIndexerClient methods
- No synonym map methods
- No BufferedSender methods
- No vector/semantic search attributes
- No content capture

---

## PR 4: Vector Search, Semantic Search + Content Capture

**Branch name:** `azure-search/pr4-vector-semantic-content`
**PR Title:** `feat(azure-search): Add vector/semantic search tracing and configurable content capture`
**Complexity:** Medium (~800-1,000 LOC)
**Review time:** ~25 min

### Business Value
> RAG pipeline developers can now trace their vector search queries — seeing k-NN parameters, vector fields, and oversampling in spans. Semantic search configuration becomes visible in traces. Content capture (gated by `TRACELOOP_TRACE_CONTENT`) lets teams inspect actual documents and embeddings flowing through their search pipeline, with privacy controls and configurable size limits.

### What to Include

**1. Semantic conventions — add to `SpanAttributes`:**

```python
# Vector search
AZURE_SEARCH_VECTOR_QUERIES_COUNT = "azure_search.search.vector_queries_count"
AZURE_SEARCH_VECTOR_FIELDS = "azure_search.search.vector_fields"
AZURE_SEARCH_VECTOR_K_NEAREST_NEIGHBORS = "azure_search.search.k_nearest_neighbors"
AZURE_SEARCH_VECTOR_EXHAUSTIVE = "azure_search.search.vector_exhaustive"
AZURE_SEARCH_VECTOR_FILTER_MODE = "azure_search.search.vector_filter_mode"
AZURE_SEARCH_VECTOR_QUERY_KIND = "azure_search.search.vector_query_kind"
AZURE_SEARCH_VECTOR_WEIGHT = "azure_search.search.vector_weight"
AZURE_SEARCH_VECTOR_OVERSAMPLING = "azure_search.search.vector_oversampling"

# Semantic search
AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME = "azure_search.search.semantic_configuration_name"
AZURE_SEARCH_QUERY_CAPTION = "azure_search.search.query_caption"
AZURE_SEARCH_QUERY_ANSWER = "azure_search.search.query_answer"
AZURE_SEARCH_SEARCH_MODE = "azure_search.search.search_mode"
AZURE_SEARCH_SCORING_PROFILE = "azure_search.search.scoring_profile"
AZURE_SEARCH_SELECT = "azure_search.search.select"
AZURE_SEARCH_SEARCH_FIELDS = "azure_search.search.search_fields"
AZURE_SEARCH_FACETS = "azure_search.search.facets"
AZURE_SEARCH_ORDER_BY = "azure_search.search.order_by"
```

**2. Wrapper additions:**

Vector/semantic attribute functions:
- `_set_vector_search_attributes(span, kwargs)` — Extract from `vector_queries` list: count, k_nearest_neighbors, fields, exhaustive, kind, weight, oversampling, vector_filter_mode
- `_set_semantic_search_attributes(span, kwargs)` — semantic_configuration_name, query_caption, query_answer (handles enum values)

Update `_set_search_attributes` to call:
- `_set_vector_search_attributes(span, kwargs)`
- `_set_semantic_search_attributes(span, kwargs)`
- Also add: search_mode, scoring_profile, select, search_fields, facets, order_by (list→comma-separated string)
- Also add: `VECTOR_DB_QUERY_TOP_K` when top is set

Content capture functions:
- `_safe_json_dumps(obj, max_length)` — JSON serialization with truncation
- `_set_request_content_attributes(span, method, instance, args, kwargs, max_items, max_length)` — Dispatch to per-method content extractors
- `_set_response_content_attributes(span, method, response, args, kwargs, max_items, max_length)` — Dispatch to per-method content extractors
- `_set_search_vector_embeddings_attributes(span, kwargs, max_items)` — Indexed `db.search.embeddings.N.vector` attributes
- `_set_document_batch_request_content_attributes(span, args, kwargs, max_items, max_length)` — Indexed `db.query.result.N.document`
- `_set_index_documents_request_content_attributes(span, args, kwargs, max_items, max_length)`
- `_set_get_document_content_attribute(span, response, max_length)` — `db.query.result.document`
- `_set_autocomplete_content_attributes(span, response, max_items, max_length)` — `db.search.result.N.entity`
- `_set_suggest_content_attributes(span, response, max_items, max_length)` — `db.search.result.N.entity`

Update `_sync_wrap` and `_async_wrap`:
- Add content capture path: `content_enabled = should_send_content()`, compute `max_items`/`max_length`, call `_set_request_content_attributes` and `_set_response_content_attributes`
- Update `_set_document_batch_response_all` and `_set_index_documents_response_all` to pass content_enabled
- Update `_set_indexing_response_single_pass` to write content when enabled

**3. Utils — make `should_send_content()` real:**

```python
TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"
TRACELOOP_TRACE_CONTENT_MAX_ITEMS = "TRACELOOP_TRACE_CONTENT_MAX_ITEMS"
TRACELOOP_TRACE_CONTENT_MAX_LENGTH = "TRACELOOP_TRACE_CONTENT_MAX_LENGTH"
_DEFAULT_MAX_CONTENT_ITEMS = 100
_DEFAULT_MAX_CONTENT_LENGTH = 16384

def should_send_content() -> bool:
    env_setting = os.getenv(TRACELOOP_TRACE_CONTENT, "true")
    override = context_api.get_value("override_enable_content_tracing")
    if override is not None:
        return _is_truthy(override)
    return _is_truthy(env_setting)

def max_content_items() -> int: ...
def max_content_length() -> int: ...
```

**4. Unit tests:**

```python
class TestVectorSearchAttributes:
    def test_vector_queries_count(self): ...
    def test_k_nearest_neighbors(self): ...
    def test_vector_fields_list_to_string(self): ...
    def test_vector_exhaustive(self): ...
    def test_vector_query_kind(self): ...
    def test_vector_weight(self): ...
    def test_vector_oversampling(self): ...
    def test_vector_filter_mode_enum(self): ...

class TestSemanticSearchAttributes:
    def test_semantic_configuration_name(self): ...
    def test_query_caption_enum(self): ...
    def test_query_answer_enum(self): ...

class TestContentCapture:
    def test_content_disabled_by_env(self): ...
    def test_content_enabled_by_default(self): ...
    def test_context_override_disables_content(self): ...
    def test_max_content_items_from_env(self): ...
    def test_max_content_length_from_env(self): ...
    def test_content_truncation(self): ...
    def test_vector_embeddings_captured(self): ...
    def test_document_batch_content_captured(self): ...
    def test_get_document_content_captured(self): ...
    def test_autocomplete_content_captured(self): ...
    def test_suggest_content_captured(self): ...

class TestAdditionalSearchParameters:
    def test_search_mode_enum(self): ...
    def test_scoring_profile(self): ...
    def test_select_list_to_comma_separated(self): ...
    def test_search_fields(self): ...
    def test_facets(self): ...
    def test_order_by(self): ...
```

### What NOT to Include
- No new client types (indexer, synonym, buffered sender)
- No SDK integration
- No docs

---

## PR 5: SearchIndexerClient, Synonym Maps, BufferedSender

**Branch name:** `azure-search/pr5-indexer-synonyms-sender`
**PR Title:** `feat(azure-search): Instrument indexer pipelines, synonym maps, service stats, and BufferedSender`
**Complexity:** Medium (~1,000-1,200 LOC)
**Review time:** ~25 min

### Business Value
> Enterprise users managing complex search infrastructure can now observe their entire indexer pipeline (data sources → skillsets → indexers), synonym map management, and service-level health statistics through OpenTelemetry. BufferedSender instrumentation helps debug batch ingestion throughput and failures.

### What to Include

**1. Semantic conventions — add remaining to `SpanAttributes`:**

```python
# Synonym map operations
AZURE_SEARCH_SYNONYM_MAP_NAME = "azure_search.synonym_map.name"
AZURE_SEARCH_SYNONYM_MAP_SYNONYMS_COUNT = "azure_search.synonym_map.synonyms_count"

# Service statistics
AZURE_SEARCH_SERVICE_DOCUMENT_COUNT = "azure_search.service.document_count"
AZURE_SEARCH_SERVICE_INDEX_COUNT = "azure_search.service.index_count"
```

**2. Method registries in `__init__.py`:**

Add:
- 6 synonym map methods to `SEARCH_INDEX_CLIENT_METHODS` and `ASYNC_SEARCH_INDEX_CLIENT_METHODS`
- `SEARCH_INDEXER_CLIENT_METHODS` — 18 sync methods (indexer CRUD, data sources, skillsets)
- `ASYNC_SEARCH_INDEXER_CLIENT_METHODS` — 18 async methods
- `BUFFERED_SENDER_METHODS` — 6 sync methods (upload, delete, merge, merge_or_upload, index, flush)
- `ASYNC_BUFFERED_SENDER_METHODS` — 6 async methods
- Update `WRAPPED_METHODS` to include all lists

**3. Wrapper additions:**

Add frozensets:
- `_INDEXER_MANAGEMENT_METHODS`
- `_DATA_SOURCE_METHODS`
- `_SKILLSET_METHODS`
- `_SYNONYM_MAP_METHODS`

Add attribute functions:
- `_set_indexer_management_attributes(span, method, args, kwargs)` — Indexer name from object or string
- `_set_data_source_attributes(span, method, args, kwargs)` — Data source name + type (handles enum)
- `_set_skillset_attributes(span, method, args, kwargs)` — Skillset name + skill count
- `_set_synonym_map_attributes(span, method, args, kwargs)` — Synonym map name + synonym count (handles string splitting)
- `_set_indexer_status_attributes(span, args, kwargs, response)` — Status, documents processed/failed from last_result

Update `_set_request_attributes` dispatch to include all new method categories.
Update `_set_response_attributes` dispatch to include `get_indexer_status`.

**4. Unit tests:**

```python
class TestSearchIndexerClientInstrumentation:
    def test_create_indexer_span(self): ...
    def test_get_indexer_status_span(self): ...
    def test_run_indexer_span(self): ...
    # ... all 9 indexer methods

class TestDataSourceInstrumentation:
    def test_create_data_source_span(self): ...
    def test_data_source_type_enum(self): ...
    # ... all 6 data source methods

class TestSkillsetInstrumentation:
    def test_create_skillset_span(self): ...
    def test_skillset_skill_count(self): ...
    # ... all 6 skillset methods

class TestSynonymMapInstrumentation:
    def test_create_synonym_map_span(self): ...
    def test_synonym_count_from_list(self): ...
    def test_synonym_count_from_newline_string(self): ...
    def test_empty_synonyms_string(self): ...
    # ... all 6 synonym map methods

class TestBufferedSenderInstrumentation:
    def test_upload_documents_span(self): ...
    def test_flush_span(self): ...
    # ... all 6 sender methods

class TestIndexerStatusResponse:
    def test_status_extracted(self): ...
    def test_documents_processed_failed(self): ...
    def test_missing_last_result_handled(self): ...

class TestServiceStatisticsResponse:
    def test_document_count(self): ...
    def test_index_count(self): ...
    def test_dict_response_format(self): ...
    def test_object_response_format(self): ...
```

### What NOT to Include
- No SDK integration
- No sample app changes
- No docs/README

---

## PR 6: SDK Integration, Integration Tests, Docs

**Branch name:** `azure-search/pr6-sdk-docs-integration`
**PR Title:** `feat(azure-search): Integrate with Traceloop SDK, add integration tests and documentation`
**Complexity:** Small-to-Medium (~2,500-3,000 LOC, mostly YAML cassettes)
**Review time:** ~20 min

### Business Value
> Users can now enable Azure Search tracing with a single `Traceloop.init()` call — zero manual instrumentor setup. Full documentation guides users through every span attribute. Integration test coverage with VCR cassettes ensures production reliability across all client types.

### What to Include

**1. SDK integration:**

`packages/traceloop-sdk/traceloop/sdk/instruments.py`:
```python
AZURE_SEARCH = "azure_search"
```

`packages/traceloop-sdk/traceloop/sdk/tracing/tracing.py`:
- Add `Instruments.AZURE_SEARCH: ["azure-search-documents"]` to library mapping
- Add `elif instrument == Instruments.AZURE_SEARCH: result = init_azure_search_instrumentor()` to instrumentation loop
- Add `init_azure_search_instrumentor()` function (try import, check `is_package_installed`, instrument)

`packages/traceloop-sdk/traceloop/sdk/utils/instrumentation_warnings.py`:
- New file with `INSTRUMENT_TO_EXTRA` dict including `"azure_search": "azure-search"` mapping

`packages/traceloop-sdk/pyproject.toml`:
- Add `azure-search` optional dependency

`packages/traceloop-sdk/README.md`:
- Add Azure AI Search to supported instruments table

**2. Sample app:**

`packages/sample-app/pyproject.toml` — Add `azure-search-documents>=11.4.0,<12` and instrumentation package
`packages/sample-app/.env.example` — Add `AZURE_SEARCH_ENDPOINT` and `AZURE_SEARCH_ADMIN_KEY`

**3. Integration tests:**

`tests/test_azure_search_integration.py` — VCR-based tests with real API shapes:
- `TestSearchClientIntegration` — search, get_document, get_document_count, upload/merge/delete/index documents, autocomplete, suggest
- `TestSearchIndexClientIntegration` — create/update/delete/get/list indexes, analyze_text, get_index_statistics, get_service_statistics
- `TestSearchIndexerClientIntegration` — get_indexer_names, get_data_source_connection_names, get_skillset_names
- `TestSynonymMapIntegration` — create/update/delete/get synonym maps
- Content capture tests (enabled vs disabled)

`tests/cassettes/*.yaml` — All 40+ VCR cassette files

**4. Documentation:**

`packages/opentelemetry-instrumentation-azure-search/README.md` — Usage, installation, configuration
`packages/opentelemetry-instrumentation-azure-search/docs/SPAN_ATTRIBUTES_GUIDE.md` — Complete attribute reference

### Tests
- VCR integration tests as described above
- Verify `Traceloop.init()` discovers and instruments Azure Search via `is_package_installed` check

---

## Verification Checklist Per PR

For each PR before opening:

```bash
# 1. Run unit tests
cd packages/opentelemetry-instrumentation-azure-search
uv run pytest tests/test_azure_search_instrumentation.py -v

# 2. Run linting
uv run ruff check .

# 3. Verify package builds
uv build
```

After PR6 merges (final verification):
```bash
nx run opentelemetry-instrumentation-azure-search:test
nx run traceloop-sdk:test
nx run-many -t lint
```

## Notes for Implementation

1. **Cherry-pick strategy:** Don't cherry-pick individual commits. Instead, copy the final-state files from the source branch and trim them to match each PR's scope.

2. **Test isolation:** Each PR's tests must pass independently. Tests should only exercise functionality introduced in that PR or prior PRs.

3. **Import guards:** In `_instrument()`, use try/except ImportError for each module to gracefully handle optional async modules.

4. **No dead code:** Don't add code paths that reference functions not yet defined. For example, PR2's `_set_request_attributes` should only dispatch to methods that exist in PR2.

5. **Content stub:** PR2-3 should have `should_send_content() → False` so the content path in `_sync_wrap`/`_async_wrap` is inert but the structure is in place for PR4 to activate.
