"""Tests for state.py workflow state management."""
import os
from datetime import datetime
import pytest


class TestStepStatus:
    def test_enum_values(self):
        from state import StepStatus
        assert StepStatus.NOT_STARTED.value == "not_started"
        assert StepStatus.STARTED.value == "started"
        assert StepStatus.COMPLETE.value == "complete"
        assert StepStatus.ERROR.value == "error"
        assert StepStatus.SKIPPED.value == "skipped"


class TestWorkflowStep:
    def test_create_step(self):
        from state import WorkflowStep, StepStatus
        step = WorkflowStep(id="gather_urls", name="Gather URLs", description="Fetch headlines")
        assert step.id == "gather_urls"
        assert step.name == "Gather URLs"
        assert step.status == StepStatus.NOT_STARTED
        assert step.retry_count == 0

    def test_start_step(self):
        from state import WorkflowStep, StepStatus
        step = WorkflowStep(id="test", name="Test", description="desc")
        step.start()
        assert step.status == StepStatus.STARTED
        assert step.started_at is not None

    def test_complete_step(self):
        from state import WorkflowStep, StepStatus
        step = WorkflowStep(id="test", name="Test", description="desc")
        step.start()
        step.complete("Done")
        assert step.status == StepStatus.COMPLETE
        assert step.completed_at is not None
        assert step.status_message == "Done"

    def test_error_step_increments_retry(self):
        from state import WorkflowStep, StepStatus
        step = WorkflowStep(id="test", name="Test", description="desc")
        step.start()
        step.error("Failed")
        assert step.status == StepStatus.ERROR
        assert step.error_message == "Failed"
        assert step.retry_count == 1
        step.error("Failed again")
        assert step.retry_count == 2

    def test_step_serialization(self):
        from state import WorkflowStep
        step = WorkflowStep(id="test", name="Test", description="desc")
        step.start()
        d = step.model_dump()
        assert d["id"] == "test"
        assert d["status"] == "started"
        restored = WorkflowStep.model_validate(d)
        assert restored.id == step.id
        assert restored.status == step.status


class TestWorkflowState:
    def _make_state(self):
        from state import WorkflowState
        ws = WorkflowState()
        ws.add_step("step_a", "Step A", "First step")
        ws.add_step("step_b", "Step B", "Second step")
        ws.add_step("step_c", "Step C", "Third step")
        return ws

    def test_add_step_returns_self(self):
        from state import WorkflowState
        ws = WorkflowState()
        result = ws.add_step("s1", "S1", "desc")
        assert result is ws

    def test_get_step(self):
        ws = self._make_state()
        step = ws.get_step("step_b")
        assert step is not None
        assert step.name == "Step B"

    def test_get_step_missing_returns_none(self):
        ws = self._make_state()
        assert ws.get_step("nonexistent") is None

    def test_start_and_complete_step(self):
        from state import StepStatus
        ws = self._make_state()
        ws.start_step("step_a")
        assert ws.get_step("step_a").status == StepStatus.STARTED
        ws.complete_step("step_a", "Done A")
        assert ws.get_step("step_a").status == StepStatus.COMPLETE

    def test_error_step(self):
        from state import StepStatus
        ws = self._make_state()
        ws.start_step("step_a")
        ws.error_step("step_a", "Boom")
        assert ws.get_step("step_a").status == StepStatus.ERROR
        assert ws.get_step("step_a").error_message == "Boom"

    def test_get_current_step(self):
        ws = self._make_state()
        assert ws.get_current_step().id == "step_a"
        ws.start_step("step_a")
        ws.complete_step("step_a")
        assert ws.get_current_step().id == "step_b"

    def test_all_complete(self):
        ws = self._make_state()
        assert ws.all_complete() is False
        for sid in ["step_a", "step_b", "step_c"]:
            ws.start_step(sid)
            ws.complete_step(sid)
        assert ws.all_complete() is True

    def test_has_errors(self):
        ws = self._make_state()
        assert ws.has_errors() is False
        ws.start_step("step_a")
        ws.error_step("step_a", "fail")
        assert ws.has_errors() is True

    def test_get_completed_steps(self):
        ws = self._make_state()
        ws.start_step("step_a")
        ws.complete_step("step_a")
        assert ws.get_completed_steps() == ["step_a"]

    def test_get_progress_percentage(self):
        ws = self._make_state()
        assert ws.get_progress_percentage() == 0.0
        ws.start_step("step_a")
        ws.complete_step("step_a")
        assert abs(ws.get_progress_percentage() - 33.33) < 1.0

    def test_clear_errors(self):
        from state import StepStatus
        ws = self._make_state()
        ws.start_step("step_a")
        ws.error_step("step_a", "fail")
        ws.clear_errors()
        assert ws.get_step("step_a").status == StepStatus.NOT_STARTED

    def test_reset(self):
        from state import StepStatus
        ws = self._make_state()
        ws.start_step("step_a")
        ws.complete_step("step_a")
        ws.reset()
        assert all(s.status == StepStatus.NOT_STARTED for s in ws.steps)

    def test_get_status_summary(self):
        ws = self._make_state()
        ws.start_step("step_a")
        ws.complete_step("step_a")
        summary = ws.get_status_summary()
        assert summary["complete"] == 1
        assert summary["not_started"] == 2

    def test_serialization_roundtrip(self):
        ws = self._make_state()
        ws.start_step("step_a")
        ws.complete_step("step_a", "ok")
        d = ws.model_dump()
        from state import WorkflowState
        restored = WorkflowState.model_validate(d)
        assert len(restored.steps) == 3
        assert restored.get_step("step_a").status_message == "ok"

    def test_get_workflow_status_report(self):
        ws = self._make_state()
        ws.start_step("step_a")
        ws.complete_step("step_a")
        report = ws.get_workflow_status_report("Test Workflow")
        assert "Test Workflow" in report
        assert "complete" in report.lower()


class TestNewsletterAgentState:
    def test_create_with_session_id(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test_20260213")
        assert s.session_id == "test_20260213"

    def test_nine_workflow_steps(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        assert len(s.steps) == 9
        step_ids = [step.id for step in s.steps]
        assert step_ids == [
            "gather_urls",
            "filter_urls",
            "download_articles",
            "extract_summaries",
            "rate_articles",
            "cluster_topics",
            "select_sections",
            "draft_sections",
            "finalize_newsletter",
        ]

    def test_headline_data_starts_empty(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        assert s.headline_data == []

    def test_add_headlines_deduplicates(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        s.add_headlines([
            {"url": "http://a.com", "title": "A"},
            {"url": "http://b.com", "title": "B"},
        ])
        assert len(s.headline_data) == 2
        s.add_headlines([
            {"url": "http://a.com", "title": "A duplicate"},
            {"url": "http://c.com", "title": "C"},
        ])
        assert len(s.headline_data) == 3

    def test_newsletter_section_data_starts_empty(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        assert s.newsletter_section_data == []

    def test_defaults(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        assert s.db_path == "newsletter_agent.db"
        assert s.sources_file == "sources.yaml"
        assert s.max_edits == 2
        assert s.do_download is True

    def test_get_status_returns_dict(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        status = s.get_status()
        assert "headlines" in status
        assert "workflow" in status
        assert status["headlines"] == 0

    def test_serialization_roundtrip(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test_rt")
        s.add_headlines([{"url": "http://x.com", "title": "X"}])
        s.start_step("gather_urls")
        s.complete_step("gather_urls", "Got 1 url")
        d = s.model_dump()
        restored = NewsletterAgentState.model_validate(d)
        assert restored.session_id == "test_rt"
        assert len(restored.headline_data) == 1
        assert restored.get_step("gather_urls").status_message == "Got 1 url"

    def test_get_unique_sources(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        s.add_headlines([
            {"url": "http://a.com", "title": "A", "source": "TechCrunch"},
            {"url": "http://b.com", "title": "B", "source": "TechCrunch"},
            {"url": "http://c.com", "title": "C", "source": "Ars Technica"},
        ])
        sources = s.get_unique_sources()
        assert sources["TechCrunch"] == 2
        assert sources["Ars Technica"] == 1


DB_PATH = "test_state_integration.db"


@pytest.fixture(autouse=True)
def clean_integration_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    yield
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


class TestStateDBIntegration:
    def test_save_and_load_checkpoint(self):
        from state import NewsletterAgentState
        from db import AgentState as AgentStateDB
        AgentStateDB.create_table(DB_PATH)

        s = NewsletterAgentState(session_id="int_test", db_path=DB_PATH)
        s.add_headlines([{"url": "http://a.com", "title": "A", "source": "s"}])
        s.start_step("gather_urls")
        s.complete_step("gather_urls", "Got 1")
        s.save_checkpoint("gather_urls")

        loaded = NewsletterAgentState.load_latest_from_db("int_test", DB_PATH)
        assert loaded is not None
        assert loaded.session_id == "int_test"
        assert len(loaded.headline_data) == 1
        assert loaded.get_step("gather_urls").status_message == "Got 1"

    def test_save_multiple_checkpoints(self):
        from state import NewsletterAgentState
        from db import AgentState as AgentStateDB
        AgentStateDB.create_table(DB_PATH)

        s = NewsletterAgentState(session_id="multi", db_path=DB_PATH)
        s.start_step("gather_urls")
        s.complete_step("gather_urls")
        s.save_checkpoint("gather_urls")

        s.start_step("filter_urls")
        s.complete_step("filter_urls")
        s.save_checkpoint("filter_urls")

        loaded = NewsletterAgentState.load_latest_from_db("multi", DB_PATH)
        assert loaded.is_step_complete("gather_urls")
        assert loaded.is_step_complete("filter_urls")

    def test_load_specific_step(self):
        from state import NewsletterAgentState, StepStatus
        from db import AgentState as AgentStateDB
        AgentStateDB.create_table(DB_PATH)

        s = NewsletterAgentState(session_id="specific", db_path=DB_PATH)
        s.start_step("gather_urls")
        s.complete_step("gather_urls", "step 1 done")
        s.save_checkpoint("gather_urls")

        s.start_step("filter_urls")
        s.complete_step("filter_urls", "step 2 done")
        s.save_checkpoint("filter_urls")

        loaded = NewsletterAgentState.load_from_db("specific", "gather_urls", DB_PATH)
        assert loaded is not None
        assert loaded.get_step("gather_urls").status_message == "step 1 done"
        assert loaded.get_step("filter_urls").status != StepStatus.COMPLETE

    def test_list_session_steps(self):
        from state import NewsletterAgentState
        from db import AgentState as AgentStateDB
        AgentStateDB.create_table(DB_PATH)

        s = NewsletterAgentState(session_id="list_test", db_path=DB_PATH)
        s.save_checkpoint("gather_urls")
        s.save_checkpoint("filter_urls")

        steps = NewsletterAgentState.list_session_steps("list_test", DB_PATH)
        assert len(steps) == 2
