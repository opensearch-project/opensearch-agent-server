"""
AG-UI Data Persistence Service

Provides database persistence for AG-UI conversations, runs, messages, and events.
Uses SQLAlchemy ORM (like Chainlit's approach) for better maintainability and type safety.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

from utils.logging_helpers import (
    get_logger,
    log_debug_event,
    log_error_event,
    log_info_event,
    log_warning_event,
)

logger = get_logger(__name__)

Base = declarative_base()


class Thread(Base):
    """Thread (conversation) table."""

    __tablename__ = "threads"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )
    user_id = Column(String, nullable=True)
    metadata_json = Column(
        "metadata", Text, nullable=True
    )  # Use metadata_json as Python attr, "metadata" as DB column

    # Relationships
    runs = relationship("Run", back_populates="thread", cascade="all, delete-orphan")
    messages = relationship(
        "Message", back_populates="thread", cascade="all, delete-orphan"
    )


class Run(Base):
    """Run (agent execution session) table."""

    __tablename__ = "runs"

    id = Column(String, primary_key=True)
    thread_id = Column(
        String, ForeignKey("threads.id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(DateTime, default=func.now(), nullable=False)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String, nullable=False)  # running, completed, error, canceled
    error_message = Column(Text, nullable=True)
    metadata_json = Column(
        "metadata", Text, nullable=True
    )  # Use metadata_json as Python attr, "metadata" as DB column

    # Relationships
    thread = relationship("Thread", back_populates="runs")
    messages = relationship(
        "Message", back_populates="run", cascade="all, delete-orphan"
    )
    events = relationship("Event", back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (Index("idx_runs_thread_id", "thread_id"),)


class Message(Base):
    """Message table."""

    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    thread_id = Column(
        String, ForeignKey("threads.id", ondelete="CASCADE"), nullable=False
    )
    run_id = Column(String, ForeignKey("runs.id", ondelete="CASCADE"), nullable=True)
    role = Column(String, nullable=False)  # user, assistant, tool
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    metadata_json = Column(
        "metadata", Text, nullable=True
    )  # Use metadata_json as Python attr, "metadata" as DB column

    # Relationships
    thread = relationship("Thread", back_populates="messages")
    run = relationship("Run", back_populates="messages")

    __table_args__ = (
        Index("idx_messages_thread_id", "thread_id"),
        Index("idx_messages_run_id", "run_id"),
    )


class Event(Base):
    """Event (AG-UI protocol event) table."""

    __tablename__ = "events"

    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    event_type = Column(String, nullable=False)
    event_data = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    run = relationship("Run", back_populates="events")

    __table_args__ = (
        Index("idx_events_run_id", "run_id"),
        Index("idx_events_type", "event_type"),
    )


class AGUIPersistence:
    """
    Persistence service for AG-UI conversations and events.

    Uses SQLAlchemy ORM (similar to Chainlit's SQLAlchemyDataLayer pattern)
    for better maintainability and type safety.

    Stores:
    - Threads (conversations)
    - Runs (agent execution sessions)
    - Messages (user and assistant messages)
    - Events (AG-UI protocol events)
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialize persistence service.

        Args:
            db_path: Path to SQLite database file. Defaults to .ag-ui/chat_history.db
        """
        if db_path is None:
            db_path = os.getenv("AG_UI_DB_PATH", ".ag-ui/chat_history.db")

        self.db_path = db_path
        self._ensure_db_directory()

        # Create SQLAlchemy engine (sync SQLite; async could use aiosqlite later if needed)
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url, echo=False)

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        log_info_event(
            logger,
            "✓ AG-UI database initialized.",
            "persistence.db_initialized",
            db_path=self.db_path,
        )

    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    def save_thread(
        self,
        thread_id: str,
        user_id: str | None = None,
        metadata: dict[str, Any | None] = None,
    ) -> None:
        """
        Save or update a thread.

        When updating an existing thread, only provided arguments are applied;
        passing user_id=None or metadata=None leaves the existing value unchanged.

        Args:
            thread_id: Thread identifier
            user_id: Optional user identifier
            metadata: Optional metadata dictionary
        """
        session = self._get_session()
        try:
            thread = session.query(Thread).filter(Thread.id == thread_id).first()
            if thread:
                thread.updated_at = datetime.now(UTC)
                if user_id:
                    thread.user_id = user_id
                if metadata:
                    thread.metadata_json = json.dumps(metadata)
            else:
                thread = Thread(
                    id=thread_id,
                    user_id=user_id,
                    metadata_json=json.dumps(metadata) if metadata else None,
                )
                session.add(thread)
            session.commit()
        except Exception as e:
            session.rollback()
            log_error_event(
                logger,
                "Failed to save thread.",
                "persistence.save_thread_failed",
                error=e,
            )
            raise
        finally:
            session.close()

    def save_run_start(
        self, run_id: str, thread_id: str, metadata: dict[str, Any | None] = None
    ) -> None:
        """
        Save a run start event.

        Args:
            run_id: Run identifier
            thread_id: Thread identifier
            metadata: Optional metadata dictionary
        """
        session = self._get_session()
        try:
            run = Run(
                id=run_id,
                thread_id=thread_id,
                status="running",
                metadata_json=json.dumps(metadata) if metadata else None,
            )
            session.add(run)
            session.commit()
            log_debug_event(
                logger,
                "Saved run start.",
                "persistence.run_start_saved",
                run_id=run_id,
                thread_id=thread_id,
            )
        except Exception as e:
            session.rollback()
            log_error_event(
                logger,
                "Failed to save run start.",
                "persistence.save_run_start_failed",
                error=e,
            )
            raise
        finally:
            session.close()

    def save_run_finish(
        self,
        run_id: str,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        """
        Save a run finish event.

        Args:
            run_id: Run identifier
            status: Run status (completed, error, canceled)
            error_message: Optional error message if status is error
        """
        session = self._get_session()
        try:
            run = session.query(Run).filter(Run.id == run_id).first()
            if run:
                run.finished_at = datetime.now(UTC)
                run.status = status
                run.error_message = error_message
                session.commit()
                log_debug_event(
                    logger,
                    "Saved run finish.",
                    "persistence.run_finish_saved",
                    run_id=run_id,
                    status=status,
                )
            else:
                log_warning_event(
                    logger,
                    "Run not found.",
                    "persistence.run_not_found",
                    run_id=run_id,
                )
        except Exception as e:
            session.rollback()
            log_error_event(
                logger,
                "Failed to save run finish.",
                "persistence.save_run_finish_failed",
                error=e,
            )
            raise
        finally:
            session.close()

    def save_message(
        self,
        message_id: str,
        thread_id: str,
        role: str,
        content: str,
        run_id: str | None = None,
        metadata: dict[str, Any | None] = None,
    ) -> None:
        """
        Save a message.

        Args:
            message_id: Message identifier
            thread_id: Thread identifier
            role: Message role (user, assistant, tool)
            content: Message content
            run_id: Optional run identifier
            metadata: Optional metadata dictionary
        """
        session = self._get_session()
        try:
            message = Message(
                id=message_id,
                thread_id=thread_id,
                run_id=run_id,
                role=role,
                content=content,
                metadata_json=json.dumps(metadata) if metadata else None,
            )
            session.add(message)
            session.commit()
            log_debug_event(
                logger,
                "Saved message.",
                "persistence.message_saved",
                message_id=message_id,
                role=role,
                thread_id=thread_id,
            )
        except Exception as e:
            session.rollback()
            log_error_event(
                logger,
                "Failed to save message.",
                "persistence.save_message_failed",
                error=e,
            )
            raise
        finally:
            session.close()

    def save_event(
        self, event_id: str, run_id: str, event_type: str, event_data: dict[str, Any]
    ) -> None:
        """
        Save an AG-UI event.

        Args:
            event_id: Event identifier
            run_id: Run identifier
            event_type: Event type (e.g., TEXT_MESSAGE_START, TOOL_CALL_START)
            event_data: Event data dictionary
        """
        session = self._get_session()
        try:
            event = Event(
                id=event_id,
                run_id=run_id,
                event_type=event_type,
                event_data=json.dumps(event_data),
            )
            session.add(event)
            session.commit()
            log_debug_event(
                logger,
                "Saved event.",
                "persistence.event_saved",
                event_id=event_id,
                event_type=event_type,
                run_id=run_id,
            )
        except Exception as e:
            session.rollback()
            log_error_event(
                logger,
                "Failed to save event.",
                "persistence.save_event_failed",
                error=e,
            )
            raise
        finally:
            session.close()

    def get_thread(self, thread_id: str) -> dict | None:
        """
        Get a thread by ID.

        Args:
            thread_id: Thread identifier

        Returns:
            Thread dictionary or None if not found
        """
        session = self._get_session()
        try:
            thread = session.query(Thread).filter(Thread.id == thread_id).first()
            if thread:
                result = {
                    "id": thread.id,
                    "created_at": thread.created_at.isoformat()
                    if thread.created_at
                    else None,
                    "updated_at": thread.updated_at.isoformat()
                    if thread.updated_at
                    else None,
                    "user_id": thread.user_id,
                    "metadata": json.loads(thread.metadata_json)
                    if thread.metadata_json
                    else {},
                }
                return result
            return None
        finally:
            session.close()

    def get_threads(
        self, user_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        """
        Get threads, optionally filtered by user_id.

        Args:
            user_id: Optional user identifier to filter by
            limit: Maximum number of threads to return
            offset: Offset for pagination

        Returns:
            List of thread dictionaries
        """
        session = self._get_session()
        try:
            query = session.query(Thread)
            if user_id:
                query = query.filter(Thread.user_id == user_id)
            threads = (
                query.order_by(Thread.updated_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            results = []
            for thread in threads:
                result = {
                    "id": thread.id,
                    "created_at": thread.created_at.isoformat()
                    if thread.created_at
                    else None,
                    "updated_at": thread.updated_at.isoformat()
                    if thread.updated_at
                    else None,
                    "user_id": thread.user_id,
                    "metadata": json.loads(thread.metadata_json)
                    if thread.metadata_json
                    else {},
                }
                results.append(result)
            return results
        finally:
            session.close()

    def _run_to_dict(self, run: Run) -> dict:
        """
        Convert Run ORM object to dictionary.

        Args:
            run: Run ORM object

        Returns:
            Dictionary representation of the run
        """
        return {
            "id": run.id,
            "thread_id": run.thread_id,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "status": run.status,
            "error_message": run.error_message,
            "metadata": json.loads(run.metadata_json) if run.metadata_json else {},
        }

    def get_run(self, run_id: str) -> dict | None:
        """
        Get a run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run dictionary or None if not found
        """
        session = self._get_session()
        try:
            run = session.query(Run).filter(Run.id == run_id).first()
            if run:
                return self._run_to_dict(run)
            return None
        finally:
            session.close()

    def get_run_with_ownership_check(self, run_id: str, user_id: str) -> dict | None:
        """
        Get a run by ID if the user owns its thread. Single query when possible.

        This method combines run retrieval with ownership verification in a single
        database query using a join, eliminating the N+1 query pattern where we would
        first fetch the run, then fetch the thread to check ownership.

        Args:
            run_id: Run identifier
            user_id: User identifier to verify ownership

        Returns:
            Run dictionary if found and user owns the thread, None otherwise
        """
        session = self._get_session()
        try:
            # Join Run with Thread and filter by run_id and user_id in a single query
            run = (
                session.query(Run)
                .join(Thread, Run.thread_id == Thread.id)
                .filter(Run.id == run_id)
                .filter(Thread.user_id == user_id)
                .first()
            )
            if run:
                return self._run_to_dict(run)
            return None
        finally:
            session.close()

    def get_runs(self, thread_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
        """
        Get runs for a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum number of runs to return
            offset: Offset for pagination

        Returns:
            List of run dictionaries
        """
        session = self._get_session()
        try:
            runs = (
                session.query(Run)
                .filter(Run.thread_id == thread_id)
                .order_by(Run.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            results = []
            for run in runs:
                results.append(self._run_to_dict(run))
            return results
        finally:
            session.close()

    def get_messages(
        self,
        thread_id: str,
        run_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Get messages for a thread, optionally filtered by run_id.

        Args:
            thread_id: Thread identifier
            run_id: Optional run identifier to filter by
            limit: Maximum number of messages to return
            offset: Offset for pagination

        Returns:
            List of message dictionaries
        """
        session = self._get_session()
        try:
            query = session.query(Message).filter(Message.thread_id == thread_id)
            if run_id:
                query = query.filter(Message.run_id == run_id)
            messages = (
                query.order_by(Message.created_at.asc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            results = []
            for msg in messages:
                result = {
                    "id": msg.id,
                    "thread_id": msg.thread_id,
                    "run_id": msg.run_id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat()
                    if msg.created_at
                    else None,
                    "metadata": json.loads(msg.metadata_json)
                    if msg.metadata_json
                    else {},
                }
                results.append(result)
            return results
        finally:
            session.close()

    def get_events(
        self,
        run_id: str,
        event_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict]:
        """
        Get events for a run, optionally filtered by event_type.

        Args:
            run_id: Run identifier
            event_type: Optional event type to filter by
            limit: Maximum number of events to return
            offset: Offset for pagination

        Returns:
            List of event dictionaries
        """
        session = self._get_session()
        try:
            query = session.query(Event).filter(Event.run_id == run_id)
            if event_type:
                query = query.filter(Event.event_type == event_type)
            events = (
                query.order_by(Event.created_at.asc()).limit(limit).offset(offset).all()
            )

            results = []
            for event in events:
                result = {
                    "id": event.id,
                    "run_id": event.run_id,
                    "event_type": event.event_type,
                    "event_data": json.loads(event.event_data),
                    "created_at": event.created_at.isoformat()
                    if event.created_at
                    else None,
                }
                results.append(result)
            return results
        finally:
            session.close()

    def delete_thread(self, thread_id: str) -> None:
        """
        Delete a thread and all associated data.

        Due to CASCADE foreign key constraints, deleting a thread will automatically
        delete all associated runs, messages, and events.

        Args:
            thread_id: Thread identifier to delete

        Raises:
            Exception: If deletion fails
        """
        session = self._get_session()
        try:
            thread = session.query(Thread).filter(Thread.id == thread_id).first()
            if thread:
                session.delete(thread)
                session.commit()
                log_info_event(
                    logger,
                    "Deleted thread.",
                    "persistence.thread_deleted",
                    thread_id=thread_id,
                )
            else:
                log_warning_event(
                    logger,
                    "Thread not found for deletion.",
                    "persistence.thread_not_found",
                    thread_id=thread_id,
                )
        except Exception as e:
            session.rollback()
            log_error_event(
                logger,
                "Failed to delete thread.",
                "persistence.delete_thread_failed",
                error=e,
            )
            raise
        finally:
            session.close()
