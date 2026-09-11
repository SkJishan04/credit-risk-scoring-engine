"""Core data contracts shared across the pipeline."""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class SectorEnum(StrEnum):
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    SERVICES = "services"
    LOGISTICS = "logistics"
    GIG_ECONOMY = "gig_economy"


class Transaction(BaseModel):
    """A single vendor <-> business cash flow event."""

    transaction_id: str
    business_id: str
    counterparty_id: str
    amount: float = Field(gt=0)
    direction: str = Field(pattern="^(inflow|outflow)$")
    transaction_date: date
    invoice_due_date: date | None = None
    settled_on_time: bool | None = None


class BusinessProfile(BaseModel):
    """Static/slow-changing attributes of a borrower."""

    business_id: str
    sector: SectorEnum
    months_active: int = Field(ge=0)
    declared_monthly_revenue: float = Field(ge=0)
    n_active_vendors: int = Field(ge=0)
    n_active_buyers: int = Field(ge=0)
    defaulted: bool | None = None  # label, None at inference time


class ScoringRequest(BaseModel):
    business_id: str
    transactions: list[Transaction]
    profile: BusinessProfile