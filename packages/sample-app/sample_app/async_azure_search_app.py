"""
Async Azure AI Search Sample App with Traceloop

Demonstrates that the instrumentation correctly wraps async methods
and emits identical span attributes as the sync azure_search_app.py.

Prerequisites:
    pip install azure-search-documents traceloop-sdk

Environment variables:
    AZURE_SEARCH_ENDPOINT - Azure Search service endpoint
    AZURE_SEARCH_ADMIN_KEY - Azure Search admin API key

Run:
    python async_azure_search_app.py
"""

import asyncio
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents import IndexDocumentsBatch
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    SearchField,
    SearchSuggester,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
    ScoringProfile,
    TextWeights,
    SynonymMap,
)
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

load_dotenv()

ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
API_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX_NAME = "sample-app-async-hotels"


Traceloop.init(app_name="async_azure_search_app")


def build_index_schema():
    """Build a search index with vector, semantic, scoring profile, and suggester."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchableField(name="description", type=SearchFieldDataType.String),
        SimpleField(name="rating", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3,
            vector_search_profile_name="default-vector-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
        profiles=[VectorSearchProfile(name="default-vector-profile", algorithm_configuration_name="default-hnsw")],
    )
    semantic_config = SemanticConfiguration(
        name="default-semantic",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="description")],
        ),
    )
    scoring_profiles = [
        ScoringProfile(name="boost-title", text_weights=TextWeights(weights={"title": 2.0})),
    ]
    suggesters = [SearchSuggester(name="sg", source_fields=["title"])]

    return SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
        scoring_profiles=scoring_profiles,
        suggesters=suggesters,
    )


HOTEL_DOCUMENTS = [
    {
        "id": "1",
        "title": "Luxury Grand Hotel",
        "description": "A five-star luxury hotel in the heart of downtown with spa and rooftop pool",
        "rating": 4.8,
        "embedding": [0.1, 0.9, 0.3],
    },
    {
        "id": "2",
        "title": "Budget Inn Express",
        "description": "Affordable accommodation near the airport with free breakfast and parking",
        "rating": 3.5,
        "embedding": [0.8, 0.2, 0.5],
    },
    {
        "id": "3",
        "title": "Seaside Resort",
        "description": "Beachfront resort with ocean views, water sports, and fine dining restaurant",
        "rating": 4.5,
        "embedding": [0.4, 0.6, 0.8],
    },
    {
        "id": "4",
        "title": "Mountain Lodge Retreat",
        "description": "Cozy mountain lodge with hiking trails, fireplace lounge, and ski access",
        "rating": 4.2,
        "embedding": [0.3, 0.4, 0.9],
    },
]


# --- Index Management ---


@task(name="async_create_index")
async def create_index(index_client):
    schema = build_index_schema()
    index = await index_client.create_or_update_index(schema)
    print(f"  Created index: {index.name}")
    return index


@task(name="async_get_index_statistics")
async def get_index_statistics(index_client):
    stats = await index_client.get_index_statistics(INDEX_NAME)
    print(f"  Index stats: {stats['document_count']} docs, {stats['storage_size']} bytes")
    return stats


@task(name="async_get_service_statistics")
async def get_service_statistics(index_client):
    stats = await index_client.get_service_statistics()
    print(f"  Service stats: {stats['counters']['document_counter']['usage']} total docs")
    return stats


# --- Document Operations ---


@task(name="async_upload_documents")
async def upload_documents(search_client):
    result = await search_client.upload_documents(documents=HOTEL_DOCUMENTS)
    succeeded = sum(1 for r in result if r.succeeded)
    print(f"  Uploaded: {succeeded}/{len(HOTEL_DOCUMENTS)} documents")
    return result


@task(name="async_merge_documents")
async def merge_documents(search_client):
    updates = [{"id": "1", "description": "Updated: A five-star luxury hotel with new rooftop bar"}]
    result = await search_client.merge_documents(documents=updates)
    print(f"  Merged: {len(updates)} documents")
    return result


@task(name="async_batch_index_documents")
async def batch_index_documents(search_client):
    batch = IndexDocumentsBatch()
    batch.add_upload_actions([
        {"id": "5", "title": "City Center Hostel", "description": "Modern hostel in city center", "rating": 3.8,
         "embedding": [0.6, 0.3, 0.7]},
    ])
    result = await search_client.index_documents(batch)
    print(f"  Batch indexed: {len(result)} actions")
    return result


@task(name="async_get_document")
async def get_document(search_client):
    doc = await search_client.get_document(key="1")
    print(f"  Retrieved: {doc['title']}")
    return doc


@task(name="async_get_document_count")
async def get_document_count(search_client):
    count = await search_client.get_document_count()
    print(f"  Document count: {count}")
    return count


# --- Search Operations ---


@task(name="async_text_search")
async def text_search(search_client):
    results = []
    async for result in await search_client.search(
        search_text="luxury hotel",
        top=5,
        filter="rating ge 4",
        select=["id", "title", "rating"],
        include_total_count=True,
    ):
        results.append(result)
    print(f"  Text search: {len(results)} results")
    return results


@task(name="async_vector_search")
async def vector_search(search_client):
    results = []
    async for result in await search_client.search(
        search_text=None,
        vector_queries=[VectorizedQuery(
            vector=[0.1, 0.9, 0.3],
            k_nearest_neighbors=3,
            fields="embedding",
        )],
    ):
        results.append(result)
    print(f"  Vector search: {len(results)} results")
    return results


@task(name="async_hybrid_search")
async def hybrid_search(search_client):
    results = []
    async for result in await search_client.search(
        search_text="luxury",
        vector_queries=[VectorizedQuery(
            vector=[0.1, 0.9, 0.3],
            k_nearest_neighbors=3,
            fields="embedding",
            exhaustive=True,
        )],
        top=5,
    ):
        results.append(result)
    print(f"  Hybrid search: {len(results)} results")
    return results


@task(name="async_search_with_scoring_profile")
async def search_with_scoring_profile(search_client):
    results = []
    async for result in await search_client.search(
        search_text="hotel",
        scoring_profile="boost-title",
        top=5,
    ):
        results.append(result)
    print(f"  Scored search: {len(results)} results")
    return results


@task(name="async_autocomplete")
async def autocomplete(search_client):
    results = await search_client.autocomplete(search_text="lux", suggester_name="sg")
    print(f"  Autocomplete: {len(results)} suggestions")
    return results


@task(name="async_suggest")
async def suggest(search_client):
    results = await search_client.suggest(search_text="sea", suggester_name="sg")
    print(f"  Suggest: {len(results)} suggestions")
    return results


# --- Synonym Map ---


@task(name="async_create_synonym_map")
async def create_synonym_map(index_client):
    sm = SynonymMap(name="async-hotel-synonyms", synonyms=["hotel, inn, lodge", "luxury, premium, deluxe"])
    result = await index_client.create_or_update_synonym_map(sm)
    print(f"  Created synonym map: {result.name}")
    return result


@task(name="async_delete_synonym_map")
async def delete_synonym_map(index_client):
    await index_client.delete_synonym_map("async-hotel-synonyms")
    print("  Deleted synonym map: async-hotel-synonyms")


# --- Cleanup ---


@task(name="async_delete_documents")
async def delete_documents(search_client):
    result = await search_client.delete_documents(documents=[{"id": "5"}])
    print(f"  Deleted: {len(result)} documents")
    return result


@task(name="async_delete_index")
async def delete_index(index_client):
    await index_client.delete_index(INDEX_NAME)
    print(f"  Deleted index: {INDEX_NAME}")


# --- Main Workflow ---


@workflow(name="azure_hotel_search_async_demo")
async def run_demo():
    credential = AzureKeyCredential(API_KEY)
    index_client = SearchIndexClient(endpoint=ENDPOINT, credential=credential)
    search_client = SearchClient(endpoint=ENDPOINT, index_name=INDEX_NAME, credential=credential)

    try:
        # Index setup
        print("\n--- Index Management (async) ---")
        await create_index(index_client)
        await asyncio.sleep(2)
        await get_service_statistics(index_client)

        # Document operations
        print("\n--- Document Operations (async) ---")
        await upload_documents(search_client)
        await asyncio.sleep(2)
        await merge_documents(search_client)
        await batch_index_documents(search_client)
        await asyncio.sleep(2)
        await get_document(search_client)
        await get_document_count(search_client)
        await get_index_statistics(index_client)

        # Search operations
        print("\n--- Search Operations (async) ---")
        await text_search(search_client)
        await vector_search(search_client)
        await hybrid_search(search_client)
        await search_with_scoring_profile(search_client)
        await autocomplete(search_client)
        await suggest(search_client)

        # Synonym map
        print("\n--- Synonym Map (async) ---")
        await create_synonym_map(index_client)
        await delete_synonym_map(index_client)

        # Cleanup extra doc
        print("\n--- Cleanup (async) ---")
        await delete_documents(search_client)

    finally:
        await delete_index(index_client)
        await index_client.close()
        await search_client.close()

    print("\nDone! Check your observability platform for traces.")


if __name__ == "__main__":
    asyncio.run(run_demo())
