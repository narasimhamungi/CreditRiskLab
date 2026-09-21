"""Canonical fundamentals schema.

One schema, two producers (Trellis adapter, direct EDGAR client). Everything downstream
sees the same columns regardless of source, so the modelling code never learns which
ingestion path was used.

XBRL tag choice matters for a credit model: issuers report the same economics under
different tags across years and filers. Each canonical field lists candidate us-gaap tags
in preference order; the first tag with a usable value for the period wins, and the tag
actually used is recorded so the provenance is auditable.
"""

from __future__ import annotations

# Balance-sheet items: instantaneous, matched on period end date.
INSTANT_CONCEPTS: dict[str, list[str]] = {
    "total_assets": ["Assets"],
    # Deliberately no fallback to LiabilitiesAndStockholdersEquity: that tag equals total
    # assets, and using it as "liabilities" silently zeroes every solvency ratio. Where
    # `Liabilities` is untagged (common — many filers omit the subtotal), total liabilities
    # is derived as assets minus equity in `features.point_in_time`, and flagged as derived.
    "total_liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "inventory": ["InventoryNet"],
    # LongTermDebt INCLUDES current maturities; LongTermDebtNoncurrent excludes them.
    # `features.ratios.total_debt` reads the tag actually used to avoid double counting.
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt"],
    "current_debt": [
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
        "OtherShortTermBorrowings",
    ],
    # A reported all-in debt total. Used only when the long-term/current components are
    # both absent (Hertz, which reports a single debt line for an unclassified balance sheet).
    "total_debt_reported": [
        "DebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
        "DebtLongtermAndShorttermCombinedAmount",
    ],
}

# Flow items: duration facts, restricted to ~annual periods.
DURATION_CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",  # common pre-2018 tag for goods sellers; last, as it can exclude services
    ],
    "ebit": ["OperatingIncomeLoss"],
    # Several issuers (J&J, Nike, Rite Aid, Hertz) never report an operating-income line.
    # EBIT is then derived as pre-tax income plus interest in features.point_in_time.
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ],
    # Last fallback: net income to common. Equal to net income without preferred stock
    # (Bed Bath & Beyond); lower by preferred dividends otherwise (Frontier, which only tagged
    # NetIncomeLoss for 2015 in a filing made in 2018 — invisible at the time, so the
    # point-in-time filter correctly refuses it). Use is flagged in derived_fields.
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "depreciation_amortisation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ],
    # Gross expense tags only. InterestIncomeExpenseNet is excluded on purpose: it nets
    # interest income, so cash-rich issuers would show a coverage ratio computed on income.
    # Last fallback excludes amortisation of discount and issuance costs (Revlon 2020), so it
    # slightly understates gross interest. Use is flagged in derived_fields.
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestExpenseNonoperating",
        "InterestAndDebtExpense",
        "InterestExpenseDebtExcludingAmortization",
    ],
    # Net interest (income minus expense). Never used as a gross expense directly: see the
    # rule in features.ratios — it is admitted for coverage only when it is a net EXPENSE.
    "net_interest": ["InterestIncomeExpenseNonoperatingNet", "InterestIncomeExpenseNet"],
}

CANONICAL_FIELDS: list[str] = list(INSTANT_CONCEPTS) + list(DURATION_CONCEPTS)

# Tags that are approximations of the canonical field, not the field itself. A snapshot that
# uses one records it in derived_fields so the approximation is visible in the audit trail.
APPROXIMATE_TAGS: dict[str, str] = {
    "NetIncomeLossAvailableToCommonStockholdersBasic": "net_income~available_to_common",
    "InterestExpenseDebtExcludingAmortization": "interest_expense~excl_amortisation",
}

# Tags that already include current maturities of long-term debt.
LONG_TERM_DEBT_INCLUSIVE_TAGS = {"LongTermDebt"}
# Current-debt tags that are current maturities of LTD (double count against inclusive tags).
CURRENT_MATURITY_TAGS = {"LongTermDebtCurrent", "DebtCurrent"}

# Columns every fundamentals frame carries, in addition to the canonical fields.
META_COLUMNS: list[str] = ["cik", "ticker", "period_end", "fiscal_year", "filed", "form", "source"]

ANNUAL_MIN_DAYS = 300
ANNUAL_MAX_DAYS = 430
