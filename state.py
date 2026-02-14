"""Workflow state management for the newsletter agent."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Status of a workflow step."""
    NOT_STARTED = "not_started"
    STARTED = "started"
    COMPLETE = "complete"
    ERROR = "error"
    SKIPPED = "skipped"


class WorkflowStep(BaseModel):
    """A single step in the workflow pipeline."""
    id: str
    name: str
    description: str = ""
    status: StepStatus = StepStatus.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str = ""
    status_message: str = ""
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def start(self) -> None:
        self.status = StepStatus.STARTED
        self.started_at = datetime.now()
        self.error_message = ""

    def complete(self, message: str = "") -> None:
        self.status = StepStatus.COMPLETE
        self.completed_at = datetime.now()
        self.status_message = message

    def error(self, message: str) -> None:
        self.status = StepStatus.ERROR
        self.error_message = message
        self.retry_count += 1

    def __str__(self) -> str:
        elapsed = ""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            elapsed = f" ({delta.total_seconds():.1f}s)"
        return f"[{self.status.value}] {self.name}{elapsed}"


class WorkflowState(BaseModel):
    """Base class for managing a multi-step workflow."""
    steps: List[WorkflowStep] = Field(default_factory=list)
    current_step_name: str = ""

    def add_step(self, step_id: str, name: str, description: str = "") -> "WorkflowState":
        self.steps.append(WorkflowStep(id=step_id, name=name, description=description))
        return self

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def start_step(self, step_id: str) -> None:
        step = self.get_step(step_id)
        if step:
            step.start()
            self.current_step_name = step.name

    def complete_step(self, step_id: str, message: str = "") -> None:
        step = self.get_step(step_id)
        if step:
            step.complete(message)

    def error_step(self, step_id: str, error_message: str) -> None:
        step = self.get_step(step_id)
        if step:
            step.error(error_message)

    def get_current_step(self) -> Optional[WorkflowStep]:
        """Return the first step that is not COMPLETE."""
        for step in self.steps:
            if step.status != StepStatus.COMPLETE:
                return step
        return None

    def all_complete(self) -> bool:
        return all(s.status == StepStatus.COMPLETE for s in self.steps)

    def has_errors(self) -> bool:
        return any(s.status == StepStatus.ERROR for s in self.steps)

    def get_completed_steps(self) -> List[str]:
        return [s.id for s in self.steps if s.status == StepStatus.COMPLETE]

    def get_failed_steps(self) -> List[str]:
        return [s.id for s in self.steps if s.status == StepStatus.ERROR]

    def get_progress_percentage(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.status == StepStatus.COMPLETE)
        return (done / len(self.steps)) * 100

    def clear_errors(self) -> None:
        for step in self.steps:
            if step.status == StepStatus.ERROR:
                step.status = StepStatus.NOT_STARTED
                step.error_message = ""

    def reset(self) -> None:
        for step in self.steps:
            step.status = StepStatus.NOT_STARTED
            step.started_at = None
            step.completed_at = None
            step.error_message = ""
            step.status_message = ""

    def get_status_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for step in self.steps:
            key = step.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_workflow_status_report(self, title: str = "Workflow") -> str:
        lines = [f"=== {title} ===", f"Progress: {self.get_progress_percentage():.1f}%", ""]
        for i, step in enumerate(self.steps):
            lines.append(f"  Step {i + 1}: {step}")
        summary = self.get_status_summary()
        lines.append("")
        lines.append(f"Summary: {summary}")
        return "\n".join(lines)


class NewsletterAgentState(WorkflowState):
    """State for the 9-step newsletter generation workflow."""

    # Session
    session_id: str
    db_path: str = "newsletter_agent.db"
    sources_file: str = "sources.yaml"

    # Data
    headline_data: List[Dict[str, Any]] = Field(default_factory=list)
    newsletter_section_data: List[Dict[str, Any]] = Field(default_factory=list)
    newsletter_title: str = ""
    final_newsletter: str = ""

    # Config
    max_edits: int = 2
    concurrency: int = 12
    do_download: bool = True
    reprocess_since: Optional[datetime] = None

    # Sources
    sources: Dict[str, Any] = Field(default_factory=dict)

    # Topics & Clustering
    cluster_names: List[str] = Field(default_factory=list)
    clusters: Dict[str, List[str]] = Field(default_factory=dict)
    common_topics: List[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.steps:
            self._initialize_workflow()

    def _initialize_workflow(self) -> None:
        self.add_step("gather_urls", "Gather URLs", "Collect headlines from sources")
        self.add_step("filter_urls", "Filter URLs", "Deduplicate and filter AI-relevant content")
        self.add_step("download_articles", "Download Articles", "Fetch full article content")
        self.add_step("extract_summaries", "Extract Summaries", "Create bullet-point summaries")
        self.add_step("rate_articles", "Rate Articles", "Evaluate quality and importance")
        self.add_step("cluster_topics", "Cluster Topics", "Group articles by theme")
        self.add_step("select_sections", "Select Sections", "Organize into newsletter sections")
        self.add_step("draft_sections", "Draft Sections", "Write section content")
        self.add_step("finalize_newsletter", "Finalize Newsletter", "Combine into final output")

    def add_headlines(self, new_headlines: List[Dict[str, Any]]) -> None:
        """Add headlines with URL-based deduplication."""
        existing_urls = {h.get("url") for h in self.headline_data}
        for h in new_headlines:
            if h.get("url") not in existing_urls:
                self.headline_data.append(h)
                existing_urls.add(h.get("url"))

    def get_unique_sources(self) -> Dict[str, int]:
        """Return source names with article counts."""
        counts: Dict[str, int] = {}
        for h in self.headline_data:
            src = h.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def is_step_complete(self, step_id: str) -> bool:
        step = self.get_step(step_id)
        return step is not None and step.status == StepStatus.COMPLETE

    def get_status(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "headlines": len(self.headline_data),
            "sources": len(self.get_unique_sources()),
            "topics": len(self.cluster_names),
            "sections": len(self.newsletter_section_data),
            "workflow": {
                "progress": self.get_progress_percentage(),
                "summary": self.get_status_summary(),
            },
        }

    def save_checkpoint(self, step_name: str) -> None:
        """Serialize current state and save to DB."""
        from db import AgentState as AgentStateDB

        state_json = self.model_dump_json()
        existing = AgentStateDB.get_by_session_and_step(self.db_path, self.session_id, step_name)
        if existing:
            existing.state_data = state_json
            existing.updated_at = datetime.now()
            existing.update(self.db_path)
        else:
            record = AgentStateDB(
                session_id=self.session_id,
                step_name=step_name,
                state_data=state_json,
                updated_at=datetime.now(),
            )
            record.insert(self.db_path)

    @classmethod
    def load_from_db(cls, session_id: str, step_name: str, db_path: str = "newsletter_agent.db") -> Optional["NewsletterAgentState"]:
        """Load state from a specific checkpoint."""
        from db import AgentState as AgentStateDB

        record = AgentStateDB.get_by_session_and_step(db_path, session_id, step_name)
        if record is None:
            return None
        return cls.model_validate_json(record.state_data)

    @classmethod
    def load_latest_from_db(cls, session_id: str, db_path: str = "newsletter_agent.db") -> Optional["NewsletterAgentState"]:
        """Load the most recent checkpoint for a session."""
        from db import AgentState as AgentStateDB

        record = AgentStateDB.get_latest_by_session(db_path, session_id)
        if record is None:
            return None
        return cls.model_validate_json(record.state_data)

    @classmethod
    def list_session_steps(cls, session_id: str, db_path: str = "newsletter_agent.db") -> List[Dict[str, Any]]:
        """List all checkpoints for a session."""
        from db import AgentState as AgentStateDB

        records = AgentStateDB.get_all_by_session(db_path, session_id)
        return [
            {"step_name": r.step_name, "updated_at": r.updated_at, "id": r.id}
            for r in records
        ]
