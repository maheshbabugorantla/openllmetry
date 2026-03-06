"""Tests for the instrumentation warnings utility."""

import os
import pytest
import logging

from traceloop.sdk.utils.instrumentation_warnings import (
    warn_missing_instrumentation,
    reset_warnings,
    INSTRUMENT_TO_EXTRA,
)


class TestInstrumentationWarnings:
    """Tests for the instrumentation warning system."""

    def setup_method(self):
        """Reset warnings before each test."""
        reset_warnings()

    def test_warn_missing_instrumentation_logs_message(self, caplog):
        """Verify warning is logged when target library is installed."""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("langchain", target_library_installed=True)

        assert "langchain" in caplog.text
        assert "traceloop-sdk[langchain]" in caplog.text
        assert "pip install" in caplog.text

    def test_warn_missing_instrumentation_logs_once(self, caplog):
        """Verify warnings are only logged once per instrument."""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("cohere", target_library_installed=True)
            warn_missing_instrumentation("cohere", target_library_installed=True)
            warn_missing_instrumentation("cohere", target_library_installed=True)

        # Count occurrences of the warning
        warning_count = caplog.text.count("traceloop-sdk[cohere]")
        assert warning_count == 1

    def test_warn_missing_instrumentation_no_log_when_library_not_installed(self, caplog):
        """Verify no warning when target library is not installed."""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("some_library", target_library_installed=False)

        assert "pip install" not in caplog.text

    def test_warn_missing_instrumentation_suppressed_by_env_var(self, caplog, monkeypatch):
        """Verify warnings can be suppressed via env var."""
        monkeypatch.setenv("TRACELOOP_SUPPRESS_WARNINGS", "true")

        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("openai", target_library_installed=True)

        assert "traceloop-sdk" not in caplog.text

    def test_warn_missing_instrumentation_suppressed_case_insensitive(self, caplog, monkeypatch):
        """Verify env var is case insensitive."""
        monkeypatch.setenv("TRACELOOP_SUPPRESS_WARNINGS", "TRUE")

        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("anthropic", target_library_installed=True)

        assert "traceloop-sdk" not in caplog.text

    def test_correct_extra_name_for_google_generativeai(self, caplog):
        """Verify correct extra name is generated for google_generativeai."""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("google_generativeai", target_library_installed=True)

        assert "traceloop-sdk[google-generativeai]" in caplog.text

    def test_correct_extra_name_for_llama_index(self, caplog):
        """Verify correct extra name is generated for llama_index."""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("llama_index", target_library_installed=True)

        assert "traceloop-sdk[llamaindex]" in caplog.text

    def test_correct_extra_name_for_chroma(self, caplog):
        """Verify correct extra name is generated for chroma."""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("chroma", target_library_installed=True)

        assert "traceloop-sdk[chromadb]" in caplog.text

    def test_reset_warnings_allows_new_warnings(self, caplog):
        """Verify reset_warnings allows the same warning to be logged again."""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("pinecone", target_library_installed=True)

        first_count = caplog.text.count("traceloop-sdk[pinecone]")
        assert first_count == 1

        reset_warnings()
        caplog.clear()

        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("pinecone", target_library_installed=True)

        second_count = caplog.text.count("traceloop-sdk[pinecone]")
        assert second_count == 1

    def test_instrument_to_extra_mapping_completeness(self):
        """Verify all expected instruments have mappings."""
        expected_instruments = [
            "openai",
            "anthropic",
            "mistral",
            "cohere",
            "google_generativeai",
            "bedrock",
            "sagemaker",
            "vertexai",
            "watsonx",
            "ollama",
            "together",
            "groq",
            "replicate",
            "writer",
            "alephalpha",
            "langchain",
            "llama_index",
            "crewai",
            "haystack",
            "agno",
            "openai_agents",
            "mcp",
            "transformers",
            "pinecone",
            "qdrant",
            "lancedb",
            "chroma",
            "milvus",
            "marqo",
            "weaviate",
        ]

        for instrument in expected_instruments:
            assert instrument in INSTRUMENT_TO_EXTRA, f"Missing mapping for {instrument}"

    def test_unknown_instrument_uses_name_as_extra(self, caplog):
        """Verify unknown instruments use their name as the extra."""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("unknown_provider", target_library_installed=True)

        assert "traceloop-sdk[unknown_provider]" in caplog.text
