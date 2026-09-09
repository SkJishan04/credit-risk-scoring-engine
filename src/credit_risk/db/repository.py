"""Repository layer isolating raw SQL/ORM access from services."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from credit_risk.data.schemas import BusinessProfile, Transaction
from credit_risk.db.models import Business, ScoreRecord, TransactionRecord
from credit_risk.models.ensemble import ScoringResult


class BusinessRepository:
    def __init__(self, db: Session):
        self.db = db

    def upsert_business(self, profile: BusinessProfile) -> Business:
        business = self.db.get(Business, profile.business_id)
        if business is None:
            business = Business(business_id=profile.business_id)
            self.db.add(business)

        business.sector = profile.sector.value
        business.months_active = profile.months_active
        business.declared_monthly_revenue = profile.declared_monthly_revenue
        business.n_active_vendors = profile.n_active_vendors
        business.n_active_buyers = profile.n_active_buyers
        self.db.flush()
        return business

    def replace_transactions(self, business_id: str, transactions: list[Transaction]) -> None:
        self.db.query(TransactionRecord).filter(
            TransactionRecord.business_id == business_id
        ).delete()
        for tx in transactions:
            self.db.add(
                TransactionRecord(
                    transaction_id=tx.transaction_id,
                    business_id=business_id,
                    counterparty_id=tx.counterparty_id,
                    amount=tx.amount,
                    direction=tx.direction,
                    transaction_date=tx.transaction_date,
                    invoice_due_date=tx.invoice_due_date,
                    settled_on_time=tx.settled_on_time,
                )
            )
        self.db.flush()

    def get_transactions(self, business_id: str) -> list[TransactionRecord]:
        stmt = select(TransactionRecord).where(TransactionRecord.business_id == business_id)
        return list(self.db.execute(stmt).scalars().all())

    def save_score(self, result: ScoringResult) -> ScoreRecord:
        record = ScoreRecord(
            business_id=result.business_id,
            default_probability=result.default_probability,
            credit_score=result.credit_score,
            recommended_interest_rate_pct=result.recommended_interest_rate_pct,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_latest_score(self, business_id: str) -> ScoreRecord | None:
        stmt = (
            select(ScoreRecord)
            .where(ScoreRecord.business_id == business_id)
            .order_by(ScoreRecord.scored_at.desc())
            .limit(1)
        )
        return self.db.execute(stmt).scalars().first()