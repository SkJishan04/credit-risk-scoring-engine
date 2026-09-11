"""ORM models for businesses, transactions, and scoring history."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from credit_risk.db.base import Base


class Business(Base):
    __tablename__ = "businesses"

    business_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sector: Mapped[str] = mapped_column(String(32), nullable=False)
    months_active: Mapped[int] = mapped_column(Integer, nullable=False)
    declared_monthly_revenue: Mapped[float] = mapped_column(Float, nullable=False)
    n_active_vendors: Mapped[int] = mapped_column(Integer, nullable=False)
    n_active_buyers: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    transactions: Mapped[list[TransactionRecord]] = relationship(back_populates="business")
    scores: Mapped[list[ScoreRecord]] = relationship(back_populates="business")


class TransactionRecord(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.business_id"), nullable=False)
    counterparty_id: Mapped[str] = mapped_column(String(64), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    transaction_date: Mapped[date] = mapped_column(Date, nullable=False)
    invoice_due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    settled_on_time: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    business: Mapped[Business] = relationship(back_populates="transactions")

    __table_args__ = (
        Index("ix_transactions_business_id_date", "business_id", "transaction_date"),
    )


class ScoreRecord(Base):
    __tablename__ = "scores"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    business_id: Mapped[str] = mapped_column(ForeignKey("businesses.business_id"), nullable=False)
    default_probability: Mapped[float] = mapped_column(Float, nullable=False)
    credit_score: Mapped[int] = mapped_column(Integer, nullable=False)
    recommended_interest_rate_pct: Mapped[float] = mapped_column(Float, nullable=False)
    scored_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    business: Mapped[Business] = relationship(back_populates="scores")

    __table_args__ = (Index("ix_scores_business_id_scored_at", "business_id", "scored_at"),)