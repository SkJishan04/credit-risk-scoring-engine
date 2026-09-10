"""initial schema: businesses, transactions, scores

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00
"""

import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "businesses",
        sa.Column("business_id", sa.String(64), primary_key=True),
        sa.Column("sector", sa.String(32), nullable=False),
        sa.Column("months_active", sa.Integer, nullable=False),
        sa.Column("declared_monthly_revenue", sa.Float, nullable=False),
        sa.Column("n_active_vendors", sa.Integer, nullable=False),
        sa.Column("n_active_buyers", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "transactions",
        sa.Column("transaction_id", sa.String(64), primary_key=True),
        sa.Column(
            "business_id", sa.String(64), sa.ForeignKey("businesses.business_id"), nullable=False
        ),
        sa.Column("counterparty_id", sa.String(64), nullable=False),
        sa.Column("amount", sa.Float, nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("transaction_date", sa.Date, nullable=False),
        sa.Column("invoice_due_date", sa.Date, nullable=True),
        sa.Column("settled_on_time", sa.Boolean, nullable=True),
    )
    op.create_index(
        "ix_transactions_business_id_date", "transactions", ["business_id", "transaction_date"]
    )

    op.create_table(
        "scores",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "business_id", sa.String(64), sa.ForeignKey("businesses.business_id"), nullable=False
        ),
        sa.Column("default_probability", sa.Float, nullable=False),
        sa.Column("credit_score", sa.Integer, nullable=False),
        sa.Column("recommended_interest_rate_pct", sa.Float, nullable=False),
        sa.Column("scored_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_scores_business_id_scored_at", "scores", ["business_id", "scored_at"])


def downgrade() -> None:
    op.drop_index("ix_scores_business_id_scored_at", table_name="scores")
    op.drop_table("scores")
    op.drop_index("ix_transactions_business_id_date", table_name="transactions")
    op.drop_table("transactions")
    op.drop_table("businesses")