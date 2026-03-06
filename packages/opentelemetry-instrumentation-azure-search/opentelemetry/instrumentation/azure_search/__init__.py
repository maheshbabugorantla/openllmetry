"""OpenTelemetry Azure AI Search instrumentation"""

import logging
from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from opentelemetry.instrumentation.azure_search.config import Config
from opentelemetry.instrumentation.azure_search.version import __version__  # noqa: F401


logger = logging.getLogger(__name__)

_instruments = ("azure-search-documents >= 11.0.0",)


class AzureSearchInstrumentor(BaseInstrumentor):
    """An instrumentor for Azure AI Search's client library."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        pass

    def _uninstrument(self, **kwargs):
        pass
