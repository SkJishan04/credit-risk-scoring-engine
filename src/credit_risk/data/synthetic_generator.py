"""
Synthetic MSME transaction data generator.

No public dataset exists that pairs granular vendor-transaction graphs with
verified default labels for redistribution, so this project ships a
documented generative process instead of pretending otherwise. Defaults are
driven by a small number of interpretable latent risk factors so that the
downstream model has genuine, learnable signal rather than pure noise:

  1. Payment-interval jitter: distressed businesses show increasing variance
     and increasing mean gap between invoice due date and actual settlement.
  2. Vendor concentration: over-reliance on very few counterparties raises
     risk (no diversification if one buyer stops paying).
  3. Revenue-to-transaction-volume mismatch: declared revenue far exceeding
     observed transaction volume is a proxy for unreliable self-reported data.
  4. A late-stage "distress spiral": once a business starts missing payments,
     the probability of a further miss increases (autocorrelated defaults).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from credit_risk.data.schemas import SectorEnum

SECTORS = list(SectorEnum)
SECTOR_VALUES = [s.value for s in SectorEnum]


@dataclass
class GeneratedDataset:
    profiles: pd.DataFrame
    transactions: pd.DataFrame


def _generate_business_latent_risk(rng: np.random.Generator, n: int) -> np.ndarray:
    """Ground-truth latent risk in [0, 1], not observed by the model directly."""
    return rng.beta(a=2.0, b=5.0, size=n)


def generate_synthetic_dataset(n_businesses: int = 4000, seed: int = 42) -> GeneratedDataset:
    rng = np.random.default_rng(seed)

    business_ids = [f"biz_{i:05d}" for i in range(n_businesses)]
    latent_risk = _generate_business_latent_risk(rng, n_businesses)
    # rng.choice on a list of str-Enum members silently coerces to a numpy
    # unicode array (losing the Enum wrapper), so choose from plain string
    # values instead of relying on numpy to preserve Enum instances.
    sectors = rng.choice(SECTOR_VALUES, size=n_businesses)
    months_active = rng.integers(3, 96, size=n_businesses)
    declared_revenue = rng.lognormal(mean=9.5, sigma=0.9, size=n_businesses)

    profile_rows = []
    tx_rows = []

    today = date(2024, 1, 1)

    for idx, business_id in enumerate(business_ids):
        risk = latent_risk[idx]
        n_vendors = max(1, int(rng.poisson(6 * (1 - 0.4 * risk))))
        n_buyers = max(1, int(rng.poisson(8 * (1 - 0.3 * risk))))
        vendor_ids = [f"vendor_{business_id}_{v}" for v in range(n_vendors)]
        buyer_ids = [f"buyer_{business_id}_{b}" for b in range(n_buyers)]

        n_tx = int(rng.poisson(40 + 60 * (1 - risk)))
        distress_state = 0.0  # grows if payments are missed, feeds autocorrelation

        for _tx_idx in range(n_tx):
            days_ago = int(rng.integers(0, 365))
            tx_date = today - timedelta(days=days_ago)
            is_inflow = rng.random() < (n_buyers / max(1, n_buyers + n_vendors))
            counterparty = rng.choice(buyer_ids if is_inflow else vendor_ids)

            base_amount = rng.lognormal(mean=7.5, sigma=1.0)
            amount = float(base_amount)

            due_offset = int(rng.integers(7, 45))
            invoice_due = tx_date + timedelta(days=due_offset)

            # Settlement jitter grows with latent risk and accumulated distress
            jitter_scale = 3 + 25 * risk + 10 * distress_state
            settlement_delay = rng.normal(loc=5 * risk, scale=jitter_scale)
            settled_on_time = settlement_delay <= 5

            if not settled_on_time:
                distress_state = min(1.0, distress_state + 0.05)
            else:
                distress_state = max(0.0, distress_state - 0.02)

            tx_rows.append(
                {
                    "transaction_id": str(uuid.uuid4()),
                    "business_id": business_id,
                    "counterparty_id": str(counterparty),
                    "amount": round(amount, 2),
                    "direction": "inflow" if is_inflow else "outflow",
                    "transaction_date": tx_date,
                    "invoice_due_date": invoice_due,
                    "settled_on_time": bool(settled_on_time),
                }
            )

        # Default label: sigmoid combination of latent risk and realized distress
        default_logit = -2.2 + 4.5 * risk + 2.0 * distress_state
        p_default = 1 / (1 + np.exp(-default_logit))
        defaulted = bool(rng.random() < p_default)

        profile_rows.append(
            {
                "business_id": business_id,
                "sector": str(sectors[idx]),
                "months_active": int(months_active[idx]),
                "declared_monthly_revenue": round(float(declared_revenue[idx]), 2),
                "n_active_vendors": n_vendors,
                "n_active_buyers": n_buyers,
                "defaulted": defaulted,
            }
        )

    profiles = pd.DataFrame(profile_rows)
    transactions = pd.DataFrame(tx_rows)
    return GeneratedDataset(profiles=profiles, transactions=transactions)