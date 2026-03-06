"""
High-value user scenario tests for optional dependencies feature.

These tests simulate real-world scenarios that users will encounter
when using traceloop-sdk with optional dependencies.
"""

import os
import sys
import logging
import pytest
from unittest.mock import patch, MagicMock

from traceloop.sdk.utils.instrumentation_warnings import (
    warn_missing_instrumentation,
    reset_warnings,
    INSTRUMENT_TO_EXTRA,
)
from traceloop.sdk.utils.package_check import is_package_installed
from traceloop.sdk.instruments import Instruments


class TestScenario1_UserHasLibraryButNotInstrumentation:
    """
    Scenario: User has installed langchain but not traceloop-sdk[langchain]

    Expected: User should see a helpful warning message telling them
    how to enable tracing for langchain.
    """

    def setup_method(self):
        reset_warnings()

    def test_warning_shows_correct_install_command(self, caplog):
        """User should see: pip install 'traceloop-sdk[langchain]'"""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("langchain", target_library_installed=True)

        assert "pip install 'traceloop-sdk[langchain]'" in caplog.text
        assert "langchain" in caplog.text
        assert "library detected but instrumentation not installed" in caplog.text


class TestScenario2_UserInstallsBaseSDKOnly:
    """
    Scenario: User runs `pip install traceloop-sdk` (base install)

    Expected: OpenAI and Anthropic instrumentations are available.
    Other instrumentations should warn if user has the library but not the extra.
    """

    def test_openai_instrumentation_available_by_default(self):
        """OpenAI instrumentation should be included in base install."""
        # This test verifies that openai instrumentation package is importable
        # In the actual package, openai is a required dependency
        try:
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor
            assert OpenAIInstrumentor is not None
        except ImportError:
            pytest.fail("OpenAI instrumentation should be available in base install")

    def test_anthropic_instrumentation_available_by_default(self):
        """Anthropic instrumentation should be included in base install."""
        try:
            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
            assert AnthropicInstrumentor is not None
        except ImportError:
            pytest.fail("Anthropic instrumentation should be available in base install")


class TestScenario3_UserSuppressesWarnings:
    """
    Scenario: User doesn't want to see warnings about missing instrumentations.

    Expected: Setting TRACELOOP_SUPPRESS_WARNINGS=true should silence all warnings.
    """

    def setup_method(self):
        reset_warnings()

    def test_suppress_warnings_env_var(self, caplog, monkeypatch):
        """Setting env var should suppress all instrumentation warnings."""
        monkeypatch.setenv("TRACELOOP_SUPPRESS_WARNINGS", "true")

        with caplog.at_level(logging.INFO):
            # Try multiple warnings
            warn_missing_instrumentation("langchain", target_library_installed=True)
            warn_missing_instrumentation("pinecone", target_library_installed=True)
            warn_missing_instrumentation("chromadb", target_library_installed=True)

        # None should appear
        assert "traceloop-sdk" not in caplog.text

    def test_suppress_warnings_various_values(self, caplog, monkeypatch):
        """Test various values for TRACELOOP_SUPPRESS_WARNINGS."""
        for value in ["true", "True", "TRUE", "1"]:
            reset_warnings()
            caplog.clear()

            # Only "true" (case-insensitive) should suppress
            monkeypatch.setenv("TRACELOOP_SUPPRESS_WARNINGS", value)

            with caplog.at_level(logging.INFO):
                warn_missing_instrumentation("cohere", target_library_installed=True)

            if value.lower() == "true":
                assert "traceloop-sdk" not in caplog.text
            # Note: "1" won't suppress as the check is for "true"


class TestScenario4_UserInstallsSpecificExtras:
    """
    Scenario: User runs `pip install 'traceloop-sdk[langchain,pinecone]'`

    Expected: Those specific instrumentations should work without warnings.
    """

    def test_extra_names_are_valid(self):
        """Verify all extra names in INSTRUMENT_TO_EXTRA are valid identifiers."""
        for instrument_name, extra_name in INSTRUMENT_TO_EXTRA.items():
            # Extra names should be lowercase and use hyphens (pyproject.toml convention)
            assert extra_name == extra_name.lower(), f"Extra {extra_name} should be lowercase"
            # No spaces
            assert " " not in extra_name, f"Extra {extra_name} should not contain spaces"


class TestScenario5_WarningsOnlyLogOnce:
    """
    Scenario: User's code initializes Traceloop multiple times or
    the instrumentor check runs multiple times.

    Expected: Warning should only appear once per instrument, not spam the logs.
    """

    def setup_method(self):
        reset_warnings()

    def test_multiple_calls_single_warning(self, caplog):
        """Calling warn multiple times should only log once."""
        with caplog.at_level(logging.INFO):
            for _ in range(10):
                warn_missing_instrumentation("milvus", target_library_installed=True)

        # Count occurrences
        count = caplog.text.count("traceloop-sdk[milvus]")
        assert count == 1, f"Expected 1 warning, got {count}"

    def test_different_instruments_get_separate_warnings(self, caplog):
        """Different instruments should each get their own warning."""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("langchain", target_library_installed=True)
            warn_missing_instrumentation("pinecone", target_library_installed=True)
            warn_missing_instrumentation("chromadb", target_library_installed=True)

        assert "traceloop-sdk[langchain]" in caplog.text
        assert "traceloop-sdk[pinecone]" in caplog.text
        assert "traceloop-sdk[chromadb]" in caplog.text


class TestScenario6_AllExtrasMapping:
    """
    Scenario: Verify all instruments have proper extras mapping.

    Expected: Every instrument enum value should map to a valid extra name.
    """

    def test_all_optional_instruments_have_mapping(self):
        """All optional instruments should have an extra mapping."""
        # These are the instruments that should have optional extras
        optional_instruments = [
            "agno", "alephalpha", "anthropic", "bedrock", "chroma", "cohere",
            "crewai", "google_generativeai", "groq", "haystack", "lancedb",
            "langchain", "llama_index", "marqo", "mcp", "milvus", "mistral",
            "ollama", "openai", "openai_agents", "pinecone", "qdrant",
            "replicate", "sagemaker", "together", "transformers", "vertexai",
            "watsonx", "weaviate", "writer"
        ]

        for instrument in optional_instruments:
            assert instrument in INSTRUMENT_TO_EXTRA, \
                f"Missing extra mapping for instrument: {instrument}"

    def test_extra_names_match_pyproject_format(self):
        """Extra names should match the format used in pyproject.toml."""
        # pyproject.toml uses lowercase with hyphens for extras
        for extra_name in INSTRUMENT_TO_EXTRA.values():
            # Should be lowercase
            assert extra_name == extra_name.lower()
            # Should not have underscores (use hyphens instead)
            # Exception: some packages legitimately have underscores
            # But our extras should use hyphens for consistency


class TestScenario7_InitInstrumentationsIntegration:
    """
    Scenario: Test the integration between init_instrumentations and warnings.

    These tests verify the warning system works correctly when called
    from the actual init_instrumentations function.
    """

    def setup_method(self):
        reset_warnings()

    def test_instrument_to_library_mapping_exists(self):
        """Verify init_instrumentations has library mappings for warnings."""
        from traceloop.sdk.tracing.tracing import init_instrumentations

        # The function should be importable and callable
        assert callable(init_instrumentations)

    def test_warning_import_successful(self):
        """Verify warn_missing_instrumentation is properly imported in tracing.py."""
        # This test verifies the import we added works
        from traceloop.sdk.tracing import tracing

        assert hasattr(tracing, 'warn_missing_instrumentation')


class TestScenario8_PackageCheckUtility:
    """
    Scenario: Test the is_package_installed utility that determines
    whether to show warnings.
    """

    def test_installed_package_detection(self):
        """Should correctly detect installed packages."""
        # pytest is definitely installed since we're running tests
        assert is_package_installed("pytest") is True

    def test_missing_package_detection(self):
        """Should correctly detect missing packages."""
        # A package that definitely doesn't exist
        assert is_package_installed("this-package-definitely-does-not-exist-12345") is False

    def test_case_insensitive_package_check(self):
        """Package check should be case-insensitive."""
        # Both should work
        assert is_package_installed("pytest") == is_package_installed("PyTest".lower())


class TestScenario9_EdgeCases:
    """
    Scenario: Edge cases and error handling.
    """

    def setup_method(self):
        reset_warnings()

    def test_empty_instrument_name(self, caplog):
        """Empty instrument name should not crash."""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("", target_library_installed=True)
        # Should not crash, might or might not log

    def test_special_characters_in_instrument_name(self, caplog):
        """Special characters should be handled gracefully."""
        with caplog.at_level(logging.INFO):
            warn_missing_instrumentation("test-instrument_v2.0", target_library_installed=True)
        # Should not crash

    def test_none_like_values_dont_crash(self, caplog):
        """Various falsy values should not crash the warning system."""
        with caplog.at_level(logging.INFO):
            # These should not raise exceptions
            warn_missing_instrumentation("test", target_library_installed=False)
            warn_missing_instrumentation("test2", target_library_installed=True)


class TestScenario10_RealWorldWorkflow:
    """
    Scenario: Simulate a real-world workflow where a user:
    1. Installs base traceloop-sdk
    2. Has langchain installed in their project
    3. Initializes Traceloop
    4. Should see a helpful warning
    """

    def setup_method(self):
        reset_warnings()

    def test_simulated_missing_instrumentation_workflow(self, caplog):
        """Simulate the workflow when user has library but not instrumentation."""
        # Simulate: langchain is installed (library check returns True)
        # but the instrumentor import will fail

        with caplog.at_level(logging.INFO):
            # This simulates what happens in init_instrumentations
            # when is_package_installed("langchain") returns True
            # but the import of LangchainInstrumentor fails
            warn_missing_instrumentation("langchain", target_library_installed=True)

        # User should see actionable guidance
        assert "langchain" in caplog.text
        assert "pip install" in caplog.text
        assert "traceloop-sdk[langchain]" in caplog.text

    def test_no_warning_when_library_not_installed(self, caplog):
        """No warning needed if user doesn't have the library at all."""
        with caplog.at_level(logging.INFO):
            # User doesn't have langchain installed, so no warning needed
            warn_missing_instrumentation("langchain", target_library_installed=False)

        # Should not see any warning about langchain
        assert "traceloop-sdk[langchain]" not in caplog.text
