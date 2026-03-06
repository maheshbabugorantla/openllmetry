"""OpenTelemetry Azure AI Search instrumentation"""

import logging
from typing import Collection
from wrapt import wrap_function_wrapper

from opentelemetry.trace import get_tracer
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.instrumentation.azure_search.config import Config
from opentelemetry.instrumentation.azure_search.wrapper import _wrap
from opentelemetry.instrumentation.azure_search.version import __version__


logger = logging.getLogger(__name__)

_instruments = ("azure-search-documents >= 11.0.0",)

# SearchClient methods (azure.search.documents) — 10 sync methods
SEARCH_CLIENT_METHODS = [
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "search",
        "span_name": "azure_search.search",
    },
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "get_document",
        "span_name": "azure_search.get_document",
    },
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "get_document_count",
        "span_name": "azure_search.get_document_count",
    },
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "upload_documents",
        "span_name": "azure_search.upload_documents",
    },
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "merge_documents",
        "span_name": "azure_search.merge_documents",
    },
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "delete_documents",
        "span_name": "azure_search.delete_documents",
    },
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "merge_or_upload_documents",
        "span_name": "azure_search.merge_or_upload_documents",
    },
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "index_documents",
        "span_name": "azure_search.index_documents",
    },
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "autocomplete",
        "span_name": "azure_search.autocomplete",
    },
    {
        "module": "azure.search.documents",
        "object": "SearchClient",
        "method": "suggest",
        "span_name": "azure_search.suggest",
    },
]

WRAPPED_METHODS = SEARCH_CLIENT_METHODS


class AzureSearchInstrumentor(BaseInstrumentor):
    """An instrumentor for Azure AI Search's client library."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                imported_module = __import__(module, fromlist=[wrap_object])
                if getattr(imported_module, wrap_object, None):
                    wrap_function_wrapper(
                        module,
                        f"{wrap_object}.{wrap_method}",
                        _wrap(tracer, wrapped_method),
                    )
            except ImportError:
                logger.debug("Could not wrap %s.%s.%s", module, wrap_object, wrap_method)
                continue

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            module = wrapped_method.get("module")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                unwrap(f"{module}.{wrap_object}", wrap_method)
            except Exception:
                pass
