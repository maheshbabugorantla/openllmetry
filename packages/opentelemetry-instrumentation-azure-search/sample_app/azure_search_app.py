"""
Azure AI Search instrumentation sample app — SearchClient + SearchIndexClient operations.

This app demonstrates automatic OpenTelemetry tracing for SearchClient and
SearchIndexClient operations. Spans are printed to the console via ConsoleSpanExporter.

Prerequisites:
    pip install opentelemetry-instrumentation-azure-search
    pip install opentelemetry-sdk azure-search-documents

Usage:
    export AZURE_SEARCH_ENDPOINT="https://<your-service>.search.windows.net"
    export AZURE_SEARCH_API_KEY="<your-api-key>"
    export AZURE_SEARCH_INDEX_NAME="hotels"
    python azure_search_app.py
"""

import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AnalyzeTextOptions,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.models import AutocompleteMode
from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# --- Tracing setup ---

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

AzureSearchInstrumentor().instrument(tracer_provider=provider)

# --- Azure AI Search client ---

ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "https://my-search.search.windows.net")
API_KEY = os.environ.get("AZURE_SEARCH_API_KEY", "placeholder-key")
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "hotels")

credential = AzureKeyCredential(API_KEY)

client = SearchClient(
    endpoint=ENDPOINT,
    index_name=INDEX_NAME,
    credential=credential,
)

index_client = SearchIndexClient(
    endpoint=ENDPOINT,
    credential=credential,
)

# --- Sample data ---

# Content capture is controlled by environment variables:
#   TRACELOOP_TRACE_CONTENT=true           (default: true)
#   TRACELOOP_TRACE_CONTENT_MAX_ITEMS=100  (max items per span)
#   TRACELOOP_TRACE_CONTENT_MAX_LENGTH=16384 (max chars per attribute)

# Placeholder 4-dimension embeddings (replace with real embeddings in production)
HOTELS = [
    {
        "hotel_id": "1",
        "hotel_name": "Grand Luxury Hotel",
        "description": "An upscale hotel in the heart of downtown.",
        "category": "Luxury",
        "rating": 4.8,
        "tags": ["wifi", "pool", "spa"],
        "description_vector": [0.12, 0.34, 0.56, 0.78],
    },
    {
        "hotel_id": "2",
        "hotel_name": "Budget Inn",
        "description": "Affordable and comfortable rooms near the airport.",
        "category": "Budget",
        "rating": 3.5,
        "tags": ["wifi", "parking"],
        "description_vector": [0.11, 0.22, 0.33, 0.44],
    },
    {
        "hotel_id": "3",
        "hotel_name": "Seaside Resort",
        "description": "Oceanfront resort with stunning views.",
        "category": "Resort",
        "rating": 4.6,
        "tags": ["beach", "pool", "restaurant"],
        "description_vector": [0.55, 0.66, 0.77, 0.88],
    },
]


def demo_upload_documents():
    """Upload sample hotel documents — generates azure.search.upload_documents span."""
    print("\n--- upload_documents ---")
    result = client.upload_documents(documents=HOTELS)
    for r in result:
        print(f"  hotel_id={r.key} succeeded={r.succeeded}")


def demo_merge_documents():
    """Update a single document field — generates azure.search.merge_documents span."""
    print("\n--- merge_documents ---")
    updates = [{"hotel_id": "1", "rating": 4.9}]
    result = client.merge_documents(documents=updates)
    for r in result:
        print(f"  hotel_id={r.key} succeeded={r.succeeded}")


def demo_get_document():
    """Retrieve a document by key — generates azure.search.get_document span."""
    print("\n--- get_document ---")
    doc = client.get_document(key="1")
    print(f"  Retrieved: {doc.get('hotel_name')}")


def demo_get_document_count():
    """Get total document count — generates azure.search.get_document_count span."""
    print("\n--- get_document_count ---")
    count = client.get_document_count()
    print(f"  Total documents: {count}")


def demo_search():
    """Full-text search — generates azure.search.search span."""
    print("\n--- search ---")
    results = client.search(
        search_text="luxury hotel",
        filter="rating ge 4",
        top=5,
    )
    for r in results:
        print(f"  {r['hotel_name']} (rating={r.get('rating')})")


def demo_vector_search():
    """Vector search — generates azure.search.search span with vector_queries attributes.

    Requires a vector field 'description_vector' configured on the index.
    """
    print("\n--- vector search ---")
    try:
        query_vector = [0.12, 0.35, 0.55, 0.80]  # Placeholder — use real embeddings
        results = client.search(
            search_text=None,
            vector_queries=[
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=3,
                    fields="description_vector",
                )
            ],
        )
        for r in results:
            print(f"  {r['hotel_name']}")
    except Exception as e:
        print(f"  Skipped (vector field not configured): {e}")


def demo_hybrid_search():
    """Hybrid (text + vector) search — generates azure.search.search span."""
    print("\n--- hybrid search ---")
    try:
        query_vector = [0.12, 0.35, 0.55, 0.80]
        results = client.search(
            search_text="luxury",
            vector_queries=[
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=3,
                    fields="description_vector",
                )
            ],
            top=5,
        )
        for r in results:
            print(f"  {r['hotel_name']}")
    except Exception as e:
        print(f"  Skipped (vector field not configured): {e}")


def demo_autocomplete():
    """Autocomplete query — generates azure.search.autocomplete span.

    Requires a suggester named 'sg' configured on the index with 'hotel_name' field.
    """
    print("\n--- autocomplete ---")
    try:
        results = client.autocomplete(
            search_text="lux",
            suggester_name="sg",
            mode=AutocompleteMode.ONE_TERM,
        )
        for r in results:
            print(f"  {r['text']}")
    except Exception as e:
        print(f"  Skipped (no suggester configured): {e}")


def demo_suggest():
    """Search suggestions — generates azure.search.suggest span.

    Requires a suggester named 'sg' configured on the index.
    """
    print("\n--- suggest ---")
    try:
        results = client.suggest(
            search_text="sea",
            suggester_name="sg",
        )
        for r in results:
            print(f"  {r['hotel_name']}")
    except Exception as e:
        print(f"  Skipped (no suggester configured): {e}")


def demo_create_synonym_map():
    """Create a synonym map — generates azure.search.create_synonym_map span.

    Note: SearchIndexerClient and SearchIndexingBufferedSender are instrumented
    but not demonstrated in this sample. Add them following the same pattern.
    """
    print("\n--- create_synonym_map ---")
    from azure.search.documents.indexes.models import SynonymMap
    try:
        synonym_map = SynonymMap(
            name="hotel-synonyms",
            synonyms="luxury, upscale, premium\nbudget, affordable, cheap",
        )
        result = index_client.create_synonym_map(synonym_map)
        print(f"  Created synonym map: {result.name}")
    except Exception as e:
        print(f"  Skipped (synonym map may already exist): {e}")


def demo_delete_synonym_map():
    """Delete a synonym map — generates azure.search.delete_synonym_map span."""
    print("\n--- delete_synonym_map ---")
    try:
        index_client.delete_synonym_map("hotel-synonyms")
        print("  Deleted synonym map: hotel-synonyms")
    except Exception as e:
        print(f"  Skipped: {e}")


def demo_create_index():
    """Create a new search index — generates azure.search.create_index span."""
    print("\n--- create_index ---")
    index = SearchIndex(
        name=INDEX_NAME,
        fields=[
            SimpleField(name="hotel_id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="hotel_name",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="description",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SimpleField(name="rating", type=SearchFieldDataType.Double, filterable=True),
        ],
    )
    try:
        result = index_client.create_index(index)
        print(f"  Created index: {result.name}")
    except Exception as e:
        print(f"  Skipped (index may already exist): {e}")


def demo_get_index_statistics():
    """Get per-index statistics — generates azure.search.get_index_statistics span."""
    print("\n--- get_index_statistics ---")
    try:
        stats = index_client.get_index_statistics(INDEX_NAME)
        print(f"  document_count={stats.document_count}  storage_size={stats.storage_size}")
    except Exception as e:
        print(f"  Skipped: {e}")


def demo_get_service_statistics():
    """Get service-level statistics — generates azure.search.get_service_statistics span."""
    print("\n--- get_service_statistics ---")
    try:
        stats = index_client.get_service_statistics()
        print(f"  counters={stats.counters}")
    except Exception as e:
        print(f"  Skipped: {e}")


def demo_analyze_text():
    """Analyze text with an analyzer — generates azure.search.analyze_text span."""
    print("\n--- analyze_text ---")
    try:
        result = index_client.analyze_text(
            index_name=INDEX_NAME,
            analyze_request=AnalyzeTextOptions(
                text="Luxury hotels with stunning ocean views",
                analyzer_name="en.lucene",
            ),
        )
        tokens = [t.token for t in result.tokens]
        print(f"  Tokens: {tokens}")
    except Exception as e:
        print(f"  Skipped: {e}")


if __name__ == "__main__":
    # SearchIndexClient operations — index and synonym map management
    demo_create_synonym_map()
    demo_delete_synonym_map()
    demo_create_index()
    demo_get_index_statistics()
    demo_get_service_statistics()
    demo_analyze_text()

    # SearchClient operations
    demo_upload_documents()
    demo_get_document()
    demo_get_document_count()
    demo_search()
    demo_vector_search()
    demo_hybrid_search()
    demo_merge_documents()
    demo_autocomplete()
    demo_suggest()
