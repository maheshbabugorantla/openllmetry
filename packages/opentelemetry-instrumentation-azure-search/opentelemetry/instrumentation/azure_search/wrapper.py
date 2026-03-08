import asyncio
import logging

from opentelemetry import context as context_api
from opentelemetry.instrumentation.azure_search.utils import dont_throw, should_send_content
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import SpanKind
from opentelemetry.semconv.trace import SpanAttributes as OTelSpanAttributes
from opentelemetry.semconv_ai import SpanAttributes

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
        span.set_attribute(OTelSpanAttributes.DB_SYSTEM, SpanAttributes.AZURE_AI_SEARCH_DB_SYSTEM_NAME)
        span.set_attribute(OTelSpanAttributes.DB_OPERATION, method)
        _set_request_attributes(span, method, instance, args, kwargs)

        # Content capture is stubbed (should_send_content() always False in PR3)
        content_enabled = should_send_content()  # noqa: F841

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
        span.set_attribute(OTelSpanAttributes.DB_SYSTEM, SpanAttributes.AZURE_AI_SEARCH_DB_SYSTEM_NAME)
        span.set_attribute(OTelSpanAttributes.DB_OPERATION, method)
        _set_request_attributes(span, method, instance, args, kwargs)

        # Content capture is stubbed (should_send_content() always False in PR3)
        content_enabled = should_send_content()  # noqa: F841

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
            _set_document_batch_response_all(span, response)
        elif method == "index_documents":
            _set_index_documents_response_all(span, response)

        span.set_status(Status(StatusCode.OK))
        return response


# --- Request attribute extraction ---


@dont_throw
def _set_index_name_attribute(span, instance, args, kwargs):
    index_name = getattr(instance, "_index_name", None)
    if index_name:
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME, index_name)


@dont_throw
def _set_search_attributes(span, args, kwargs):
    search_text = kwargs.get("search_text") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT, search_text)
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SEARCH_TOP, kwargs.get("top"))
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SEARCH_SKIP, kwargs.get("skip"))
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SEARCH_FILTER, kwargs.get("filter"))

    query_type = kwargs.get("query_type")
    if query_type is not None:
        qt_str = query_type.value if hasattr(query_type, "value") else str(query_type)
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SEARCH_QUERY_TYPE, qt_str)

    top = kwargs.get("top")
    if top:
        _set_span_attribute(span, SpanAttributes.VECTOR_DB_QUERY_TOP_K, top)


@dont_throw
def _set_get_document_attributes(span, args, kwargs):
    key = kwargs.get("key") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_KEY, key)


@dont_throw
def _set_document_batch_attributes(span, args, kwargs):
    documents = kwargs.get("documents") or (args[0] if args else None)
    if documents and hasattr(documents, "__len__"):
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT, len(documents))


@dont_throw
def _set_index_documents_attributes(span, args, kwargs):
    batch = kwargs.get("batch") or (args[0] if args else None)
    if batch:
        actions = getattr(batch, "actions", None)
        if actions and hasattr(actions, "__len__"):
            _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT, len(actions))


@dont_throw
def _set_suggestion_attributes(span, args, kwargs):
    search_text = kwargs.get("search_text") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SEARCH_TEXT, search_text)
    suggester_name = kwargs.get("suggester_name") or (args[1] if len(args) > 1 else None)
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SUGGESTER_NAME, suggester_name)


@dont_throw
def _set_index_management_attributes(span, method, args, kwargs):
    """Set attributes for index management operations."""
    if method in ["create_index", "create_or_update_index"]:
        index = kwargs.get("index") or (args[0] if args else None)
        if index:
            index_name = getattr(index, "name", None)
            _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME, index_name)
    elif method in ["delete_index", "get_index", "get_index_statistics"]:
        index_name = kwargs.get("index") or kwargs.get("index_name") or (args[0] if args else None)
        if isinstance(index_name, str):
            _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME, index_name)
        elif hasattr(index_name, "name"):
            _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME, index_name.name)


@dont_throw
def _set_analyze_text_attributes(span, args, kwargs):
    """Set attributes for analyze_text operation."""
    index_name = kwargs.get("index_name") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_INDEX_NAME, index_name)

    analyze_request = kwargs.get("analyze_request") or (args[1] if len(args) > 1 else None)
    analyzer_name = None

    if analyze_request:
        analyzer_name = getattr(analyze_request, "analyzer_name", None)

    if not analyzer_name:
        analyzer_name = kwargs.get("analyzer_name") or kwargs.get("analyzer")

    if analyzer_name:
        if hasattr(analyzer_name, "value"):
            analyzer_name = analyzer_name.value
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_ANALYZER_NAME, str(analyzer_name))


# --- Response attribute extraction ---


@dont_throw
def _set_search_response_attributes(span, response):
    """Sync: set results count from SearchItemPaged.get_count()."""
    count_fn = getattr(response, "get_count", None)
    if not callable(count_fn):
        return
    # Skip async coroutines here — handled by _set_search_response_attributes_async
    if asyncio.iscoroutinefunction(count_fn):
        return
    total = count_fn()
    if total is not None:
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SEARCH_RESULTS_COUNT, total)


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
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SEARCH_RESULTS_COUNT, total)


@dont_throw
def _set_document_count_response_attributes(span, response):
    if isinstance(response, int):
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_COUNT, response)


@dont_throw
def _set_autocomplete_response_attributes(span, response):
    if isinstance(response, list):
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_AUTOCOMPLETE_RESULTS_COUNT, len(response))


@dont_throw
def _set_suggest_response_attributes(span, response):
    if isinstance(response, list):
        _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SUGGEST_RESULTS_COUNT, len(response))


def _deep_get(obj, key):
    """Get a value from an object that may be a dict or an object with attributes."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


@dont_throw
def _set_service_statistics_response_attributes(span, response):
    """Set attributes from get_service_statistics response."""
    counters = _deep_get(response, "counters")
    if counters:
        doc_counter = _deep_get(counters, "document_counter")
        if doc_counter:
            usage = _deep_get(doc_counter, "usage")
            if usage is not None:
                _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SERVICE_DOCUMENT_COUNT, usage)

        index_counter = _deep_get(counters, "index_counter")
        if index_counter:
            usage = _deep_get(index_counter, "usage")
            if usage is not None:
                _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_SERVICE_INDEX_COUNT, usage)


def _set_indexing_response_single_pass(span, results):
    """Count succeeded/failed in a single pass over results."""
    succeeded = 0
    for result in results:
        if getattr(result, "succeeded", False):
            succeeded += 1
    failed = len(results) - succeeded
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_SUCCEEDED_COUNT, succeeded)
    _set_span_attribute(span, SpanAttributes.AZURE_AI_SEARCH_DOCUMENT_FAILED_COUNT, failed)


@dont_throw
def _set_document_batch_response_all(span, response):
    if not isinstance(response, list) or len(response) == 0:
        return
    _set_indexing_response_single_pass(span, response)


@dont_throw
def _set_index_documents_response_all(span, response):
    if isinstance(response, list):
        results = response
    else:
        results = getattr(response, "results", None)
    if not results or not isinstance(results, list):
        return
    _set_indexing_response_single_pass(span, results)
