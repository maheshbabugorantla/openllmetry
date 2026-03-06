import asyncio
import json
import logging

from opentelemetry import context as context_api
from opentelemetry.instrumentation.azure_search.utils import (
    dont_throw,
    max_content_items,
    max_content_length,
    should_send_content,
)
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import SpanKind
from opentelemetry.semconv_ai import SpanAttributes, EventAttributes

logger = logging.getLogger(__name__)

# Module-level frozensets for O(1) method dispatch
_DOCUMENT_BATCH_METHODS = frozenset({
    "upload_documents",
    "merge_documents",
    "delete_documents",
    "merge_or_upload_documents",
})

_SUGGESTION_METHODS = frozenset({"autocomplete", "suggest"})

_INDEX_MANAGEMENT_METHODS = frozenset({
    "create_index",
    "create_or_update_index",
    "delete_index",
    "get_index",
    "get_index_statistics",
})


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@dont_throw
def _set_request_attributes(span, method, instance, args, kwargs):
    """Set all pre-call span attributes based on the method being called."""
    _set_index_name_attribute(span, instance, args, kwargs)

    if method == "search":
        _set_search_attributes(span, args, kwargs)
    elif method == "get_document":
        _set_get_document_attributes(span, args, kwargs)
    elif method in _DOCUMENT_BATCH_METHODS:
        _set_document_batch_attributes(span, args, kwargs)
    elif method == "index_documents":
        _set_index_documents_attributes(span, args, kwargs)
    elif method in _SUGGESTION_METHODS:
        _set_suggestion_attributes(span, args, kwargs)
    elif method in _INDEX_MANAGEMENT_METHODS:
        _set_index_management_attributes(span, method, args, kwargs)
    elif method == "analyze_text":
        _set_analyze_text_attributes(span, args, kwargs)


@dont_throw
def _set_response_attributes(span, method, response, args, kwargs):
    """Set all post-call span attributes from the response."""
    if response is None:
        return

    if method == "search":
        _set_search_response_attributes(span, response)
    elif method == "get_document_count":
        _set_document_count_response_attributes(span, response)
    elif method == "autocomplete":
        _set_autocomplete_response_attributes(span, response)
    elif method == "suggest":
        _set_suggest_response_attributes(span, response)
    elif method == "get_service_statistics":
        _set_service_statistics_response_attributes(span, response)


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in WRAPPED_METHODS."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    if asyncio.iscoroutinefunction(wrapped):
        return _async_wrap(tracer, to_wrap, wrapped, instance, args, kwargs)

    return _sync_wrap(tracer, to_wrap, wrapped, instance, args, kwargs)


def _sync_wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Synchronous instrumentation wrapper."""
    name = to_wrap.get("span_name")
    method = to_wrap.get("method")

    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
        },
        set_status_on_exception=False,
    ) as span:
        _set_request_attributes(span, method, instance, args, kwargs)

        # Compute content toggle, max items, and max length once per span
        content_enabled = should_send_content()
        if content_enabled:
            max_items = max_content_items()
            max_length = max_content_length()
            _set_request_content_attributes(span, method, instance, args, kwargs, max_items, max_length)

        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        _set_response_attributes(span, method, response, args, kwargs)

        if method in _DOCUMENT_BATCH_METHODS:
            _set_document_batch_response_all(
                span, response, content_enabled,
                max_items if content_enabled else 0,
                max_length if content_enabled else 0,
            )
        elif method == "index_documents":
            _set_index_documents_response_all(
                span, response, content_enabled,
                max_items if content_enabled else 0,
                max_length if content_enabled else 0,
            )
        elif content_enabled:
            _set_response_content_attributes(span, method, response, args, kwargs, max_items, max_length)

        span.set_status(Status(StatusCode.OK))
        return response


async def _async_wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Asynchronous instrumentation wrapper."""
    name = to_wrap.get("span_name")
    method = to_wrap.get("method")

    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
        },
        set_status_on_exception=False,
    ) as span:
        _set_request_attributes(span, method, instance, args, kwargs)

        # Compute content toggle, max items, and max length once per span
        content_enabled = should_send_content()
        if content_enabled:
            max_items = max_content_items()
            max_length = max_content_length()
            _set_request_content_attributes(span, method, instance, args, kwargs, max_items, max_length)

        try:
            response = await wrapped(*args, **kwargs)
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        _set_response_attributes(span, method, response, args, kwargs)

        # For search, get_count() is a coroutine on AsyncSearchItemPaged
        if method == "search":
            await _set_search_response_attributes_async(span, response)

        if method in _DOCUMENT_BATCH_METHODS:
            _set_document_batch_response_all(
                span, response, content_enabled,
                max_items if content_enabled else 0,
                max_length if content_enabled else 0,
            )
        elif method == "index_documents":
            _set_index_documents_response_all(
                span, response, content_enabled,
                max_items if content_enabled else 0,
                max_length if content_enabled else 0,
            )
        elif content_enabled:
            _set_response_content_attributes(span, method, response, args, kwargs, max_items, max_length)

        span.set_status(Status(StatusCode.OK))
        return response


# --- Request attribute extraction ---


@dont_throw
def _set_index_name_attribute(span, instance, args, kwargs):
    index_name = getattr(instance, "_index_name", None)
    if index_name:
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index_name)


@dont_throw
def _set_search_attributes(span, args, kwargs):
    search_text = kwargs.get("search_text") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_TEXT, search_text)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_TOP, kwargs.get("top"))
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_SKIP, kwargs.get("skip"))
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_FILTER, kwargs.get("filter"))

    query_type = kwargs.get("query_type")
    if query_type is not None:
        qt_str = query_type.value if hasattr(query_type, "value") else str(query_type)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_QUERY_TYPE, qt_str)

    top = kwargs.get("top")
    if top:
        _set_span_attribute(span, SpanAttributes.VECTOR_DB_QUERY_TOP_K, top)

    # Vector search attributes (PR4)
    _set_vector_search_attributes(span, kwargs)

    # Semantic search attributes (PR4)
    _set_semantic_search_attributes(span, kwargs)

    # Additional search parameters (PR4)
    search_mode = kwargs.get("search_mode")
    if search_mode is not None:
        sm_str = search_mode.value if hasattr(search_mode, "value") else str(search_mode)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_MODE, sm_str)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SCORING_PROFILE, kwargs.get("scoring_profile"))

    select = kwargs.get("select")
    if select:
        if isinstance(select, (list, tuple)):
            select = ",".join(select)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SELECT, select)

    search_fields = kwargs.get("search_fields")
    if search_fields:
        if isinstance(search_fields, (list, tuple)):
            search_fields = ",".join(search_fields)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_FIELDS, search_fields)

    facets = kwargs.get("facets")
    if facets:
        if isinstance(facets, (list, tuple)):
            facets = ",".join(str(f) for f in facets)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_FACETS, facets)

    order_by = kwargs.get("order_by")
    if order_by:
        if isinstance(order_by, (list, tuple)):
            order_by = ",".join(str(o) for o in order_by)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_ORDER_BY, order_by)


@dont_throw
def _set_vector_search_attributes(span, kwargs):
    """Set attributes for vector search queries."""
    vector_queries = kwargs.get("vector_queries")
    if not vector_queries:
        return

    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_QUERIES_COUNT, len(vector_queries))

    first_vq = vector_queries[0]

    k = getattr(first_vq, "k_nearest_neighbors", None) or getattr(first_vq, "k", None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_K_NEAREST_NEIGHBORS, k)

    fields = getattr(first_vq, "fields", None)
    if fields:
        if isinstance(fields, (list, tuple)):
            fields = ",".join(fields)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_FIELDS, fields)

    exhaustive = getattr(first_vq, "exhaustive", None)
    if exhaustive is not None:
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_EXHAUSTIVE, exhaustive)

    kind = getattr(first_vq, "kind", None)
    if kind is not None:
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_QUERY_KIND, str(kind))

    weight = getattr(first_vq, "weight", None)
    if weight is not None:
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_WEIGHT, weight)

    oversampling = getattr(first_vq, "oversampling", None)
    if oversampling is not None:
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_OVERSAMPLING, oversampling)

    vector_filter_mode = kwargs.get("vector_filter_mode")
    if vector_filter_mode is not None:
        vfm_str = vector_filter_mode.value if hasattr(vector_filter_mode, "value") else str(vector_filter_mode)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_FILTER_MODE, vfm_str)


@dont_throw
def _set_semantic_search_attributes(span, kwargs):
    """Set attributes for semantic search configuration."""
    _set_span_attribute(
        span,
        SpanAttributes.AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME,
        kwargs.get("semantic_configuration_name"),
    )

    query_caption = kwargs.get("query_caption")
    if query_caption is not None:
        qc_str = query_caption.value if hasattr(query_caption, "value") else str(query_caption)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_QUERY_CAPTION, qc_str)

    query_answer = kwargs.get("query_answer")
    if query_answer is not None:
        qa_str = query_answer.value if hasattr(query_answer, "value") else str(query_answer)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_QUERY_ANSWER, qa_str)


@dont_throw
def _set_get_document_attributes(span, args, kwargs):
    key = kwargs.get("key") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY, key)


@dont_throw
def _set_document_batch_attributes(span, args, kwargs):
    documents = kwargs.get("documents") or (args[0] if args else None)
    if documents and hasattr(documents, "__len__"):
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT, len(documents))


@dont_throw
def _set_index_documents_attributes(span, args, kwargs):
    batch = kwargs.get("batch") or (args[0] if args else None)
    if batch:
        actions = getattr(batch, "actions", None)
        if actions and hasattr(actions, "__len__"):
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT, len(actions))


@dont_throw
def _set_suggestion_attributes(span, args, kwargs):
    search_text = kwargs.get("search_text") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_TEXT, search_text)
    suggester_name = kwargs.get("suggester_name") or (args[1] if len(args) > 1 else None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME, suggester_name)


@dont_throw
def _set_index_management_attributes(span, method, args, kwargs):
    if method in ["create_index", "create_or_update_index"]:
        index = kwargs.get("index") or (args[0] if args else None)
        if index:
            index_name = getattr(index, "name", None)
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index_name)
    elif method in ["delete_index", "get_index", "get_index_statistics"]:
        index_name = kwargs.get("index") or kwargs.get("index_name") or (args[0] if args else None)
        if isinstance(index_name, str):
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index_name)
        elif hasattr(index_name, "name"):
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index_name.name)


@dont_throw
def _set_analyze_text_attributes(span, args, kwargs):
    index_name = kwargs.get("index_name") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index_name)

    analyze_request = kwargs.get("analyze_request") or (args[1] if len(args) > 1 else None)
    analyzer_name = None

    if analyze_request:
        analyzer_name = getattr(analyze_request, "analyzer_name", None)

    if not analyzer_name:
        analyzer_name = kwargs.get("analyzer_name") or kwargs.get("analyzer")

    if analyzer_name:
        if hasattr(analyzer_name, "value"):
            analyzer_name = analyzer_name.value
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_ANALYZER_NAME, str(analyzer_name))


# --- Response attribute extraction ---


@dont_throw
def _set_search_response_attributes(span, response):
    """Sync: set results count from SearchItemPaged.get_count()."""
    count_fn = getattr(response, "get_count", None)
    if not callable(count_fn):
        return
    if asyncio.iscoroutinefunction(count_fn):
        return
    total = count_fn()
    if total is not None:
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_RESULTS_COUNT, total)


@dont_throw
async def _set_search_response_attributes_async(span, response):
    """Async: set results count from AsyncSearchItemPaged.get_count()."""
    count_fn = getattr(response, "get_count", None)
    if not callable(count_fn):
        return
    if asyncio.iscoroutinefunction(count_fn):
        total = await count_fn()
    else:
        total = count_fn()
    if total is not None:
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_RESULTS_COUNT, total)


@dont_throw
def _set_document_count_response_attributes(span, response):
    if isinstance(response, int):
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT, response)


@dont_throw
def _set_autocomplete_response_attributes(span, response):
    if isinstance(response, list):
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT, len(response))


@dont_throw
def _set_suggest_response_attributes(span, response):
    if isinstance(response, list):
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT, len(response))


def _deep_get(obj, key):
    """Get a value from an object that may be a dict or an object with attributes."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


@dont_throw
def _set_service_statistics_response_attributes(span, response):
    counters = _deep_get(response, "counters")
    if counters:
        doc_counter = _deep_get(counters, "document_counter")
        if doc_counter:
            usage = _deep_get(doc_counter, "usage")
            if usage is not None:
                _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SERVICE_DOCUMENT_COUNT, usage)

        index_counter = _deep_get(counters, "index_counter")
        if index_counter:
            usage = _deep_get(index_counter, "usage")
            if usage is not None:
                _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SERVICE_INDEX_COUNT, usage)


# --- Content attribute functions (gated by TRACELOOP_TRACE_CONTENT) ---
# Content is stored as indexed span attributes (like gen_ai_prompt.0.content)
# instead of span events, because APM backends (e.g. Elastic) drop span events.


def _safe_json_dumps(obj, max_length=0):
    """Serialize obj to JSON, falling back to str() for non-serializable objects."""
    try:
        result = json.dumps(obj)
    except (TypeError, ValueError):
        result = str(obj)
    if max_length > 0 and len(result) > max_length:
        return result[:max_length] + "...[truncated]"
    return result


@dont_throw
def _set_request_content_attributes(span, method, instance, args, kwargs, max_items, max_length):
    if method == "search":
        _set_search_vector_embeddings_attributes(span, kwargs, max_items)
    elif method in _DOCUMENT_BATCH_METHODS:
        _set_document_batch_request_content_attributes(span, args, kwargs, max_items, max_length)
    elif method == "index_documents":
        _set_index_documents_request_content_attributes(span, args, kwargs, max_items, max_length)


@dont_throw
def _set_response_content_attributes(span, method, response, args, kwargs, max_items, max_length):
    if response is None:
        return

    if method == "get_document":
        _set_get_document_content_attribute(span, response, max_length)
    elif method == "autocomplete":
        _set_autocomplete_content_attributes(span, response, max_items, max_length)
    elif method == "suggest":
        _set_suggest_content_attributes(span, response, max_items, max_length)


@dont_throw
def _set_search_vector_embeddings_attributes(span, kwargs, max_items):
    vector_queries = kwargs.get("vector_queries")
    if not vector_queries:
        return

    for i, vq in enumerate(vector_queries):
        if i >= max_items:
            break
        prefix = f"{EventAttributes.DB_SEARCH_EMBEDDINGS_VECTOR.value}.{i}"
        vector = getattr(vq, "vector", None)
        text = getattr(vq, "text", None)
        if vector is not None:
            _set_span_attribute(span, prefix, str(vector))
        elif text is not None:
            _set_span_attribute(span, prefix, text)


@dont_throw
def _set_document_batch_request_content_attributes(span, args, kwargs, max_items, max_length):
    documents = kwargs.get("documents") or (args[0] if args else None)
    if not documents:
        return

    for i, doc in enumerate(documents):
        if i >= max_items:
            break
        _set_span_attribute(
            span,
            f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.{i}",
            _safe_json_dumps(doc, max_length),
        )


@dont_throw
def _set_index_documents_request_content_attributes(span, args, kwargs, max_items, max_length):
    batch = kwargs.get("batch") or (args[0] if args else None)
    if not batch:
        return

    actions = getattr(batch, "actions", None)
    if not actions:
        return

    for i, action in enumerate(actions):
        if i >= max_items:
            break
        _set_span_attribute(
            span,
            f"{EventAttributes.DB_QUERY_RESULT_DOCUMENT.value}.{i}",
            _safe_json_dumps(action, max_length),
        )


@dont_throw
def _set_get_document_content_attribute(span, response, max_length):
    _set_span_attribute(
        span,
        EventAttributes.DB_QUERY_RESULT_DOCUMENT.value,
        _safe_json_dumps(response, max_length),
    )


@dont_throw
def _set_autocomplete_content_attributes(span, response, max_items, max_length):
    if not isinstance(response, list):
        return

    for i, item in enumerate(response):
        if i >= max_items:
            break
        text = _deep_get(item, "text")
        query_plus_text = _deep_get(item, "query_plus_text")
        entity = _safe_json_dumps({"text": text, "query_plus_text": query_plus_text}, max_length)
        _set_span_attribute(span, f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.{i}", entity)


@dont_throw
def _set_suggest_content_attributes(span, response, max_items, max_length):
    if not isinstance(response, list):
        return

    for i, item in enumerate(response):
        if i >= max_items:
            break
        _set_span_attribute(
            span,
            f"{EventAttributes.DB_SEARCH_RESULT_ENTITY.value}.{i}",
            _safe_json_dumps(item, max_length),
        )


def _set_indexing_response_single_pass(span, results, content_enabled=False, max_items=0, max_length=0):
    """Single-pass: count succeeded/failed AND set content attributes."""
    succeeded = 0
    for i, result in enumerate(results):
        if getattr(result, "succeeded", False):
            succeeded += 1
        if content_enabled and i < max_items:
            key = _deep_get(result, "key")
            if key is not None:
                _set_span_attribute(span, f"{EventAttributes.DB_QUERY_RESULT_ID.value}.{i}", str(key))
            _set_span_attribute(
                span,
                f"{EventAttributes.DB_QUERY_RESULT_METADATA.value}.{i}",
                _safe_json_dumps({
                    "succeeded": _deep_get(result, "succeeded"),
                    "status_code": _deep_get(result, "status_code"),
                    "error_message": _deep_get(result, "error_message"),
                }, max_length),
            )
    failed = len(results) - succeeded
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT, succeeded)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT, failed)


@dont_throw
def _set_document_batch_response_all(span, response, content_enabled=False, max_items=0, max_length=0):
    if not isinstance(response, list) or len(response) == 0:
        return
    _set_indexing_response_single_pass(span, response, content_enabled, max_items, max_length)


@dont_throw
def _set_index_documents_response_all(span, response, content_enabled=False, max_items=0, max_length=0):
    if isinstance(response, list):
        results = response
    else:
        results = getattr(response, "results", None)
    if not results or not isinstance(results, list):
        return
    _set_indexing_response_single_pass(span, results, content_enabled, max_items, max_length)
