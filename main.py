import pandas as pd
import streamlit as st
from pathlib import Path

# Data locations
DATA_DIR = Path(__file__).parent
GLOBAL_FILE = DATA_DIR / "global_factors.xlsx"
SPX_FILE = DATA_DIR / "spx_factors.xlsx"
WITHDRAWALS_FILE = DATA_DIR / "withdrawals.csv"
MONTHS_PER_YEAR = 12

# Allocation label -> (global column, spx column)
ALLOCATION_CHOICES = [
    ("100% equity", "LBM 100E", "spx100e"),
    ("90% equity / 10% fixed income", "LBM 90E", "spx90e"),
    ("80% equity / 20% fixed income", "LBM 80E", "spx80e"),
    ("70% equity / 30% fixed income", "LBM 70E", "spx70e"),
    ("60% equity / 40% fixed income", "LBM 60E", "spx60e"),
    ("50% equity / 50% fixed income", "LBM 50E", "spx50e"),
    ("40% equity / 60% fixed income", "LBM 40E", "spx40e"),
    ("30% equity / 70% fixed income", "LBM 30E", "spx30e"),
    ("20% equity / 80% fixed income", "LBM 20E", "spx20e"),
    ("10% equity / 90% fixed income", "LBM 10E", "spx10e"),
    ("100% fixed income", "LBM 100F", "spx0e"),
]


def format_currency(value: float) -> str:
    """Return a currency-formatted string for UI display."""
    return f"${value:,.0f}"


@st.cache_data(show_spinner=False)
def load_factors(path: Path) -> pd.DataFrame:
    """Read the factor workbook once and reuse the cached dataframe."""
    return pd.read_excel(path)


@st.cache_data(show_spinner=False)
def load_withdrawal_rates(path: Path) -> pd.DataFrame:
    """Return withdrawal rates indexed by years."""
    df = pd.read_csv(path)
    return df.set_index("Years")


def rolling_balances_lump(factors: pd.Series, years: int, principal: float) -> tuple[list[float], list[object]]:
    """Compute future balances for every rolling window assuming an upfront investment."""
    if years < 1:
        raise ValueError("Years must be at least 1.")
    if principal <= 0:
        return [], []

    numeric = pd.to_numeric(factors, errors="coerce").dropna()
    values = numeric.to_numpy(dtype=float)

    required_rows = (years - 1) * MONTHS_PER_YEAR + 1
    if len(values) < required_rows:
        raise ValueError("Not enough factor history for the selected horizon.")

    balances: list[float] = []
    starts: list[object] = []
    stride = MONTHS_PER_YEAR

    for start in range(0, len(values) - (years - 1) * stride):
        balance = principal
        for year_idx in range(years):
            factor = values[start + year_idx * stride]
            balance *= factor
        balances.append(balance)
        starts.append(numeric.index[start])

    return balances, starts


def summarize_lump_outcomes(
    series: pd.Series, years: int, principal: float
) -> dict[str, object]:
    """Return the ending balance distribution plus min/median stats for a lump sum."""
    balances, starts = rolling_balances_lump(series, years, principal)
    if not balances:
        empty_series = pd.Series(dtype=float, name="ending_balance")
        empty_table = pd.DataFrame(columns=["window_start_row", "ending_balance"])
        return {"values": empty_series, "table": empty_table, "min": 0.0, "median": 0.0}

    ending_series = pd.Series(balances, index=starts, name="ending_balance")
    sorted_table = (
        ending_series.sort_values()
        .reset_index()
        .rename(columns={"index": "window_start_row"})
    )
    return {
        "values": ending_series,
        "table": sorted_table,
        "min": float(ending_series.min()),
        "median": float(ending_series.median()),
    }


st.set_page_config(page_title="One-Time Spend Opportunity Cost", layout="wide")
st.title("One-Time Spend Opportunity Cost")

st.write(
    "Evaluate what a single discretionary purchase could grow into if it stays invested instead."
)

with st.sidebar:
    st.header("Inputs")
    spend_amount = st.number_input(
        "One-time spend amount ($)",
        min_value=0.0,
        value=10000.0,
        step=500.0,
        format="%.2f",
    )
    years = st.slider("Years until using the money", 1, 60,35)
    retirement_years = st.slider("Years in retirement", 1, 60, 30)
    allocation_label = st.selectbox(
        "Portfolio allocation",
        options=[label for label, _, _ in ALLOCATION_CHOICES],
        index=0,
    )
    expense_ratio_pct = st.slider(
        "Annual expense ratio (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        format="%.1f%%",
        help="Net returns reduce by this fee each year (10 basis point steps).",
    )

selected_global_col = next(
    global_col for label, global_col, _ in ALLOCATION_CHOICES if label == allocation_label
)
selected_spx_col = next(
    spx_col for label, _, spx_col in ALLOCATION_CHOICES if label == allocation_label
)

if spend_amount <= 0:
    st.warning("Increase the one-time spend amount to see its opportunity cost.")
else:
    st.subheader("Investment Equivalent")
    st.write(f"Initial investment: **{format_currency(spend_amount)}**")
    st.write(f"Years invested: **{years}**")
    st.write(f"Annual expenses applied: **{expense_ratio_pct:.1f}%**")

    try:
        global_df = load_factors(GLOBAL_FILE)
        spx_df = load_factors(SPX_FILE)
        withdrawals = load_withdrawal_rates(WITHDRAWALS_FILE)

        expense_ratio = expense_ratio_pct / 100.0
        # Apply an annual net-of-fee adjustment by multiplying each factor by (1 - fee).
        global_factors = pd.to_numeric(global_df[selected_global_col], errors="coerce")
        spx_factors = pd.to_numeric(spx_df[selected_spx_col], errors="coerce")

        net_multiplier = 1.0 - expense_ratio
        net_global_series = global_factors * net_multiplier
        net_spx_series = spx_factors * net_multiplier

        global_outcomes = summarize_lump_outcomes(net_global_series, years, spend_amount)
        spx_outcomes = summarize_lump_outcomes(net_spx_series, years, spend_amount)

        st.subheader("Investment Outcomes")
        cols = st.columns(2)

        with cols[0]:
            st.markdown(f"**Global Factors ({selected_global_col})**")
            st.metric("Median ending balance", format_currency(global_outcomes["median"]))
            st.metric("Worst-case ending balance", format_currency(global_outcomes["min"]))

        with cols[1]:
            st.markdown(f"**S&P 500 Factors ({selected_spx_col})**")
            st.metric("Median ending balance", format_currency(spx_outcomes["median"]))
            st.metric("Worst-case ending balance", format_currency(spx_outcomes["min"]))

        st.caption("Returns use rolling windows of the selected allocation's annual factors.")

        if years >= 20 and spend_amount > 0 and not withdrawals.empty:
            withdrawal_row = withdrawals.reindex([years]).ffill().bfill().iloc[0]
            median_rate = float(withdrawal_row.get("Median", 0.0))

            st.subheader("Impact on Retirement Income (60% Stock Study)")
            st.write(
                f"Using a median withdrawal rate of **{median_rate:.1%}** from the 60% stock allocation study."
            )

            withdraw_cols = st.columns(2)

            median_withdrawal_global = global_outcomes["median"] * median_rate
            total_income_global = median_withdrawal_global * retirement_years
            median_withdrawal_spx = spx_outcomes["median"] * median_rate
            total_income_spx = median_withdrawal_spx * retirement_years

            with withdraw_cols[0]:
                st.markdown("**Global Portfolio**")
                st.metric(
                    "Median average annual withdrawal",
                    format_currency(median_withdrawal_global),
                    help="Median ending balance × median withdrawal rate",
                )
                st.metric(
                    f"Total Lifetime Retirement Spending over {retirement_years} years",
                    format_currency(total_income_global),
                    help="Median average annual withdrawal × years in retirement",
                )

            with withdraw_cols[1]:
                st.markdown("**S&P 500 Portfolio**")
                st.metric(
                    "Median average annual withdrawal",
                    format_currency(median_withdrawal_spx),
                    help="Median ending balance × median withdrawal rate",
                )
                st.metric(
                    f"Total over {retirement_years} years",
                    format_currency(total_income_spx),
                    help="Median average annual withdrawal × years in retirement",
                )
    except FileNotFoundError as err:
        st.error(f"Missing factor file: {err}")
    except ValueError as err:
        st.error(str(err))
