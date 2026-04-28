"""
E-Commerce Analytics Dashboard
Streamlit + Plotly | Deployable on Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta, date

# ── Page config must be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* KPI card polish */
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }
    [data-testid="metric-container"] label {
        font-size: .75rem !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .05em;
        color: #64748B !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 800;
        color: #0F172A !important;
    }
    /* Insight callout */
    .insight {
        background: #EEF2FF;
        border-left: 4px solid #4F46E5;
        border-radius: 0 8px 8px 0;
        padding: 11px 16px;
        margin: 10px 0 22px 0;
        font-size: .88rem;
        color: #374151;
        line-height: 1.55;
    }
    /* Section sub-title */
    h3 { color: #0F172A; }
    /* Tighten tab font */
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: .87rem; }
    /* Sidebar separator */
    [data-testid="stSidebar"] hr { border-color: #E2E8F0; margin: 14px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str = "data.csv") -> pd.DataFrame:
    """Read and type-cast the raw CSV. Returns augmented DataFrame."""
    df = pd.read_csv(path, encoding="latin1", dtype=str)

    df["Quantity"]    = pd.to_numeric(df["Quantity"],  errors="coerce")
    df["UnitPrice"]   = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], format="%m/%d/%Y %H:%M", errors="coerce"
    )

    # Normalise CustomerID: "17850.0" → "17850", blanks/NaN → np.nan
    def _clean_cid(val):
        if pd.isna(val) or str(val).strip() in ("", "nan"):
            return np.nan
        try:
            return str(int(float(val)))
        except (ValueError, OverflowError):
            return np.nan

    df["CustomerID"]  = df["CustomerID"].apply(_clean_cid)
    df["Revenue"]     = df["Quantity"] * df["UnitPrice"]
    df["IsCancelled"] = df["InvoiceNo"].str.startswith("C", na=False)
    df["Date"]        = df["InvoiceDate"].dt.normalize()
    return df


@st.cache_data(show_spinner=False)
def split_data(df: pd.DataFrame):
    """Split raw rows into (valid_sales, returns_cancellations)."""
    sales = df[
        (~df["IsCancelled"]) &
        (df["Quantity"]  > 0) &
        (df["UnitPrice"] > 0)
    ].copy()

    returns = df[
        df["IsCancelled"] | (df["Quantity"] < 0)
    ].copy()

    return sales, returns


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple,
    countries: list,
    search: str,
) -> pd.DataFrame:
    """Apply sidebar filters to any DataFrame that has Date / Country / Description / StockCode."""
    if df.empty:
        return df

    start = pd.Timestamp(date_range[0])
    end   = pd.Timestamp(date_range[1])
    mask  = (df["Date"] >= start) & (df["Date"] <= end)

    if countries:
        mask &= df["Country"].isin(countries)

    if search.strip():
        q = search.strip().lower()
        mask &= (
            df["Description"].str.lower().str.contains(q, na=False) |
            df["StockCode"].str.lower().str.contains(q,  na=False)
        )

    return df[mask].copy()


# ════════════════════════════════════════════════════════════════════════════
#  ANALYTICS HELPERS
# ════════════════════════════════════════════════════════════════════════════

def fmt_currency(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "£0.00"
    v = float(v)
    if abs(v) >= 1_000_000:
        return f"£{v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"£{v / 1_000:.1f}K"
    return f"£{v:,.2f}"


def fmt_number(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "0"
    v = float(v)
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{int(v):,}"


def calculate_kpis(sales: pd.DataFrame, returns: pd.DataFrame, raw: pd.DataFrame) -> dict:
    zero = dict(revenue=0, orders=0, units=0, aov=0, customers=0, products=0, cancel_rate=0)
    if sales.empty:
        return zero

    total_rev   = sales["Revenue"].sum()
    n_orders    = sales["InvoiceNo"].nunique()
    units_sold  = sales["Quantity"].sum()
    aov         = total_rev / n_orders if n_orders else 0
    n_customers = sales["CustomerID"].nunique()
    n_products  = sales["StockCode"].nunique()

    total_inv   = raw["InvoiceNo"].nunique()
    cancel_inv  = raw[raw["IsCancelled"]]["InvoiceNo"].nunique()
    cancel_rate = (cancel_inv / total_inv * 100) if total_inv else 0

    return dict(
        revenue=total_rev, orders=n_orders, units=units_sold,
        aov=aov, customers=n_customers, products=n_products,
        cancel_rate=cancel_rate,
    )


def build_rfm(sales: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM scores and assign segments. Requires CustomerID column."""
    df = sales.dropna(subset=["CustomerID"]).copy()
    if df.empty:
        return pd.DataFrame()

    ref = df["Date"].max() + timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency   = ("Date",       lambda x: (ref - x.max()).days),
            Frequency = ("InvoiceNo",  "nunique"),
            Monetary  = ("Revenue",    "sum"),
        )
        .reset_index()
    )

    # Use rank(method="first") so qcut always gets n unique bins
    def score(series, n=5, ascending=True):
        ranked = series.rank(method="first")
        labels = list(range(1, n + 1))
        s = pd.qcut(ranked, q=n, labels=labels).astype(int)
        return s if ascending else (n + 1 - s)

    rfm["R_Score"] = score(rfm["Recency"],   ascending=False)  # lower recency = better
    rfm["F_Score"] = score(rfm["Frequency"], ascending=True)
    rfm["M_Score"] = score(rfm["Monetary"],  ascending=True)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def _segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 3 and f >= 3:
            return "Loyal Customers"
        if r <= 2 and f >= 3:
            return "At Risk"
        if r >= 4 and f <= 2:
            return "New Customers"
        return "Low Value"

    rfm["Segment"] = rfm.apply(_segment, axis=1)
    return rfm


def insight(text: str):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def empty_guard(df: pd.DataFrame, label: str = "this view"):
    if df.empty:
        st.warning(f"No data matches the current filters for {label}.")
        return True
    return False


# ════════════════════════════════════════════════════════════════════════════
#  BOOTSTRAP — load & split once
# ════════════════════════════════════════════════════════════════════════════

raw = load_data()
sales_all, returns_all = split_data(raw)


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛒 E-Commerce Analytics")
    st.markdown("---")

    # Date range
    min_d = sales_all["Date"].min().date() if not sales_all.empty else date(2010, 1, 1)
    max_d = sales_all["Date"].max().date() if not sales_all.empty else date(2011, 12, 31)

    st.markdown("**📅 Date Range**")
    dr = st.date_input(
        "date_range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        label_visibility="collapsed",
    )
    d_start, d_end = (dr[0], dr[1]) if (isinstance(dr, (list, tuple)) and len(dr) == 2) else (min_d, max_d)

    # Country filter
    countries_all = sorted(sales_all["Country"].dropna().unique().tolist())
    st.markdown("**🌍 Country**")
    selected_countries = st.multiselect(
        "countries",
        options=countries_all,
        default=[],
        placeholder="All countries",
        label_visibility="collapsed",
    )

    # Product search
    st.markdown("**🔎 Product Search**")
    search_query = st.text_input(
        "search",
        value="",
        placeholder="e.g. candle, 85123A…",
        label_visibility="collapsed",
    )

    # Include cancelled toggle
    st.markdown("**⚙️ Options**")
    include_cancelled = st.toggle("Include cancelled / returned rows", value=False)

    st.markdown("---")
    st.caption(
        f"Dataset: **{len(raw):,}** rows  \n"
        f"Date span: {min_d} → {max_d}  \n"
        f"Countries: {len(countries_all)}"
    )


# ════════════════════════════════════════════════════════════════════════════
#  FILTER APPLICATION
# ════════════════════════════════════════════════════════════════════════════

sales           = apply_filters(sales_all,   (d_start, d_end), selected_countries, search_query)
returns_filt    = apply_filters(returns_all,  (d_start, d_end), selected_countries, search_query)
display_df      = apply_filters(raw,         (d_start, d_end), selected_countries, search_query)

kpis = calculate_kpis(sales, returns_filt, raw)


# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("# 🛒 E-Commerce Analytics Dashboard")
st.markdown(
    "Real-time business intelligence across sales, customers, products, and fulfilment. "
    "Use the **sidebar** to slice by date, country, or product."
)

# ── KPI row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Total Revenue",     fmt_currency(kpis["revenue"]))
c2.metric("Orders",            fmt_number(kpis["orders"]))
c3.metric("Units Sold",        fmt_number(kpis["units"]))
c4.metric("Avg Order Value",   fmt_currency(kpis["aov"]))
c5.metric("Unique Customers",  fmt_number(kpis["customers"]))
c6.metric("Unique Products",   fmt_number(kpis["products"]))
c7.metric("Cancellation Rate", f"{kpis['cancel_rate']:.1f}%")

st.markdown("---")

# Hard stop if nothing passes the filters
if sales.empty:
    st.warning(
        "No sales data matches the current filters. "
        "Try widening the date range or removing country / product filters."
    )
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════

(
    tab_overview,
    tab_trends,
    tab_products,
    tab_customers,
    tab_countries,
    tab_returns,
    tab_quality,
) = st.tabs([
    "📊 Executive Overview",
    "📈 Sales Trends",
    "📦 Products",
    "👥 Customers",
    "🌍 Countries",
    "↩️ Returns",
    "🔍 Data Quality",
])

PLOTLY_TEMPLATE = "plotly_white"
COLOR_PRIMARY   = "#4F46E5"
COLOR_GREEN     = "#10B981"
COLOR_AMBER     = "#F59E0B"
COLOR_RED       = "#EF4444"
COLOR_SLATE     = "#94A3B8"

SEG_COLORS = {
    "Champions":       COLOR_PRIMARY,
    "Loyal Customers": COLOR_GREEN,
    "At Risk":         COLOR_RED,
    "New Customers":   COLOR_AMBER,
    "Low Value":       COLOR_SLATE,
}


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.subheader("Revenue at a Glance")

    rev_monthly = (
        sales.set_index("InvoiceDate")
        .resample("ME")["Revenue"]
        .sum()
        .reset_index()
    )
    rev_monthly.columns = ["Month", "Revenue"]

    fig_rev_area = px.area(
        rev_monthly, x="Month", y="Revenue",
        title="Monthly Revenue",
        color_discrete_sequence=[COLOR_PRIMARY],
        template=PLOTLY_TEMPLATE,
    )
    fig_rev_area.update_layout(
        xaxis_title="", yaxis_title="Revenue (£)",
        hovermode="x unified",
        yaxis_tickprefix="£", yaxis_tickformat=",.0f",
    )
    st.plotly_chart(fig_rev_area, use_container_width=True)

    peak_month = rev_monthly.loc[rev_monthly["Revenue"].idxmax()]
    insight(
        f"Peak monthly revenue was **{fmt_currency(peak_month['Revenue'])}** "
        f"({peak_month['Month'].strftime('%B %Y')}). "
        "Open the **Sales Trends** tab for daily, weekly, and heatmap breakdowns."
    )

    col_l, col_r = st.columns(2)

    with col_l:
        top5_prod = (
            sales.groupby("Description")["Revenue"]
            .sum().nlargest(5).reset_index().sort_values("Revenue")
        )
        top5_prod.columns = ["Product", "Revenue"]
        fig_top5 = px.bar(
            top5_prod, x="Revenue", y="Product",
            orientation="h", title="Top 5 Products by Revenue",
            color="Revenue", color_continuous_scale="Purples",
            template=PLOTLY_TEMPLATE,
        )
        fig_top5.update_layout(
            xaxis_tickprefix="£", xaxis_tickformat=",.0f",
            coloraxis_showscale=False, yaxis_title="",
        )
        st.plotly_chart(fig_top5, use_container_width=True)

    with col_r:
        top5_cntry = (
            sales.groupby("Country")["Revenue"]
            .sum().nlargest(5).reset_index().sort_values("Revenue")
        )
        top5_cntry.columns = ["Country", "Revenue"]
        fig_top5c = px.bar(
            top5_cntry, x="Revenue", y="Country",
            orientation="h", title="Top 5 Countries by Revenue",
            color="Revenue", color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE,
        )
        fig_top5c.update_layout(
            xaxis_tickprefix="£", xaxis_tickformat=",.0f",
            coloraxis_showscale=False, yaxis_title="",
        )
        st.plotly_chart(fig_top5c, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SALES TRENDS
# ════════════════════════════════════════════════════════════════════════════

with tab_trends:
    st.subheader("Sales Trends")

    granularity = st.radio(
        "Time granularity",
        ["Daily", "Weekly", "Monthly"],
        index=2,
        horizontal=True,
    )
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    freq = freq_map[granularity]

    trend_rev = (
        sales.set_index("InvoiceDate")
        .resample(freq)["Revenue"]
        .sum()
        .reset_index()
    )
    trend_rev.columns = ["Period", "Revenue"]

    trend_orders = (
        sales.set_index("InvoiceDate")
        .resample(freq)["InvoiceNo"]
        .nunique()
        .reset_index()
    )
    trend_orders.columns = ["Period", "Orders"]

    trend_units = (
        sales.set_index("InvoiceDate")
        .resample(freq)["Quantity"]
        .sum()
        .reset_index()
    )
    trend_units.columns = ["Period", "Units"]

    # Revenue line
    fig_line = px.line(
        trend_rev, x="Period", y="Revenue",
        title=f"{granularity} Revenue",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[COLOR_PRIMARY],
    )
    fig_line.update_layout(
        xaxis_title="", yaxis_title="Revenue (£)",
        yaxis_tickprefix="£", yaxis_tickformat=",.0f",
        hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        fig_orders = px.bar(
            trend_orders, x="Period", y="Orders",
            title=f"{granularity} Orders",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[COLOR_GREEN],
        )
        fig_orders.update_layout(xaxis_title="", yaxis_title="Orders", hovermode="x unified")
        st.plotly_chart(fig_orders, use_container_width=True)

    with col_r:
        fig_units = px.bar(
            trend_units, x="Period", y="Units",
            title=f"{granularity} Units Sold",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[COLOR_AMBER],
        )
        fig_units.update_layout(xaxis_title="", yaxis_title="Units", hovermode="x unified")
        st.plotly_chart(fig_units, use_container_width=True)

    # Monthly revenue heatmap
    st.subheader("Monthly Revenue Heatmap")

    s_heat = sales.copy()
    s_heat["Year"]  = s_heat["InvoiceDate"].dt.year.astype(str)
    s_heat["Month"] = s_heat["InvoiceDate"].dt.month

    pivot = (
        s_heat.groupby(["Year", "Month"])["Revenue"]
        .sum()
        .reset_index()
        .pivot(index="Year", columns="Month", values="Revenue")
        .fillna(0)
    )
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_labels[m - 1] for m in pivot.columns]

    fig_heat = px.imshow(
        pivot,
        title="Revenue Heatmap (Year × Month)",
        color_continuous_scale="Purples",
        aspect="auto",
        template=PLOTLY_TEMPLATE,
        text_auto=".2s",
    )
    fig_heat.update_layout(
        xaxis_title="Month", yaxis_title="Year",
        coloraxis_colorbar_title="Revenue (£)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    insight(
        "Darker cells mark peak revenue periods. "
        "Use the heatmap to align marketing campaigns and stock replenishment with historical demand spikes."
    )


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — PRODUCTS
# ════════════════════════════════════════════════════════════════════════════

with tab_products:
    st.subheader("Product Performance")

    n = st.slider("Show top N products", min_value=5, max_value=25, value=10, key="prod_n")

    col_l, col_r = st.columns(2)

    with col_l:
        top_rev = (
            sales.groupby("Description")["Revenue"]
            .sum().nlargest(n).reset_index().sort_values("Revenue")
        )
        top_rev.columns = ["Product", "Revenue"]
        fig_pr = px.bar(
            top_rev, x="Revenue", y="Product",
            orientation="h", title=f"Top {n} Products by Revenue",
            color="Revenue", color_continuous_scale="Purples",
            template=PLOTLY_TEMPLATE,
        )
        fig_pr.update_layout(
            yaxis_title="", xaxis_tickprefix="£",
            xaxis_tickformat=",.0f", coloraxis_showscale=False,
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_r:
        top_qty = (
            sales.groupby("Description")["Quantity"]
            .sum().nlargest(n).reset_index().sort_values("Quantity")
        )
        top_qty.columns = ["Product", "Units Sold"]
        fig_pq = px.bar(
            top_qty, x="Units Sold", y="Product",
            orientation="h", title=f"Top {n} Products by Units Sold",
            color="Units Sold", color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE,
        )
        fig_pq.update_layout(yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_pq, use_container_width=True)

    insight(
        "High-revenue products are not always high-volume. "
        "Compare both charts to find premium items vs. commodity movers — then price and promote accordingly."
    )

    st.subheader("Product Summary Table")
    prod_tbl = (
        sales.groupby(["StockCode", "Description"])
        .agg(
            Revenue   = ("Revenue",   "sum"),
            Units     = ("Quantity",  "sum"),
            Avg_Price = ("UnitPrice", "mean"),
            Orders    = ("InvoiceNo", "nunique"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    prod_tbl_disp = prod_tbl.copy()
    prod_tbl_disp["Revenue"]   = prod_tbl_disp["Revenue"].apply(lambda x: f"£{x:,.2f}")
    prod_tbl_disp["Avg_Price"] = prod_tbl_disp["Avg_Price"].apply(lambda x: f"£{x:,.2f}")
    prod_tbl_disp["Units"]     = prod_tbl_disp["Units"].apply(lambda x: f"{int(x):,}")
    prod_tbl_disp.columns = ["Stock Code", "Description", "Revenue", "Units Sold", "Avg Unit Price", "# Orders"]

    st.dataframe(prod_tbl_disp.head(100), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — CUSTOMERS
# ════════════════════════════════════════════════════════════════════════════

with tab_customers:
    st.subheader("Customer Analysis")

    cust_sales = sales.dropna(subset=["CustomerID"])

    if empty_guard(cust_sales, "customer analysis"):
        st.stop()

    col_l, col_r = st.columns(2)

    with col_l:
        top_cust_rev = (
            cust_sales.groupby("CustomerID")["Revenue"]
            .sum().nlargest(10).reset_index().sort_values("Revenue")
        )
        top_cust_rev.columns = ["Customer ID", "Revenue"]
        fig_cr = px.bar(
            top_cust_rev, x="Revenue", y="Customer ID",
            orientation="h", title="Top 10 Customers by Revenue",
            color="Revenue", color_continuous_scale="Purples",
            template=PLOTLY_TEMPLATE,
        )
        fig_cr.update_layout(
            yaxis_title="Customer ID",
            xaxis_tickprefix="£", xaxis_tickformat=",.0f",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cr, use_container_width=True)

    with col_r:
        top_cust_ord = (
            cust_sales.groupby("CustomerID")["InvoiceNo"]
            .nunique().nlargest(10).reset_index().sort_values("InvoiceNo")
        )
        top_cust_ord.columns = ["Customer ID", "Orders"]
        fig_co = px.bar(
            top_cust_ord, x="Orders", y="Customer ID",
            orientation="h", title="Top 10 Customers by Order Count",
            color="Orders", color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE,
        )
        fig_co.update_layout(yaxis_title="Customer ID", coloraxis_showscale=False)
        st.plotly_chart(fig_co, use_container_width=True)

    # ── RFM Segmentation ─────────────────────────────────────────────────────
    st.subheader("RFM Customer Segmentation")
    st.caption(
        "Customers scored 1–5 on Recency (days since last order), "
        "Frequency (# unique invoices), and Monetary value (£ total spend). "
        "Segments are derived from combined scores."
    )

    with st.spinner("Computing RFM scores…"):
        rfm = build_rfm(cust_sales)

    if rfm.empty:
        st.info("Not enough customer data for RFM segmentation.")
    else:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Customers"]

        col_l, col_r = st.columns(2)

        with col_l:
            fig_pie = px.pie(
                seg_counts, names="Segment", values="Customers",
                title="Customer Segment Distribution",
                color="Segment", color_discrete_map=SEG_COLORS,
                template=PLOTLY_TEMPLATE, hole=0.45,
            )
            fig_pie.update_traces(textposition="outside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_r:
            rfm_avg = (
                rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
                .mean().round(1).reset_index()
            )
            fig_rfm_grp = px.bar(
                rfm_avg.melt(id_vars="Segment", var_name="Metric", value_name="Average"),
                x="Segment", y="Average", color="Metric",
                barmode="group",
                title="Avg RFM Metrics by Segment",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[COLOR_PRIMARY, COLOR_GREEN, COLOR_AMBER],
            )
            fig_rfm_grp.update_layout(xaxis_title="", yaxis_title="Average Value")
            st.plotly_chart(fig_rfm_grp, use_container_width=True)

        # Scatter: Recency vs Monetary, bubble = Frequency
        fig_rfm_scatter = px.scatter(
            rfm, x="Recency", y="Monetary",
            size="Frequency", color="Segment",
            color_discrete_map=SEG_COLORS,
            title="RFM Scatter — Recency vs. Spend  (bubble size = Purchase Frequency)",
            template=PLOTLY_TEMPLATE,
            hover_data={"CustomerID": True, "Frequency": True, "Recency": True},
            opacity=0.75,
            size_max=30,
        )
        fig_rfm_scatter.update_layout(
            xaxis_title="Recency (days since last order)",
            yaxis_title="Monetary Value (£)",
        )
        st.plotly_chart(fig_rfm_scatter, use_container_width=True)

        champ_n    = int(seg_counts.loc[seg_counts["Segment"] == "Champions",    "Customers"].sum())
        at_risk_n  = int(seg_counts.loc[seg_counts["Segment"] == "At Risk",       "Customers"].sum())
        low_val_n  = int(seg_counts.loc[seg_counts["Segment"] == "Low Value",     "Customers"].sum())

        insight(
            f"You have **{champ_n:,} Champions** (high recency, frequency & spend) and "
            f"**{at_risk_n:,} At-Risk** customers who used to buy but haven't recently. "
            f"The **{low_val_n:,} Low-Value** group is a win-back opportunity. "
            "Target At-Risk with reactivation campaigns and Champions with loyalty rewards."
        )

        with st.expander("View full RFM table"):
            rfm_disp = rfm.copy()
            rfm_disp["Monetary"] = rfm_disp["Monetary"].apply(lambda x: f"£{x:,.2f}")
            st.dataframe(
                rfm_disp[[
                    "CustomerID", "Recency", "Frequency", "Monetary",
                    "R_Score", "F_Score", "M_Score", "RFM_Score", "Segment"
                ]],
                use_container_width=True,
                hide_index=True,
            )


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — COUNTRIES
# ════════════════════════════════════════════════════════════════════════════

with tab_countries:
    st.subheader("Geographic Analysis")

    col_l, col_r = st.columns(2)

    with col_l:
        top_rev_cntry = (
            sales.groupby("Country")["Revenue"]
            .sum().nlargest(15).reset_index().sort_values("Revenue")
        )
        top_rev_cntry.columns = ["Country", "Revenue"]
        fig_crev = px.bar(
            top_rev_cntry, x="Revenue", y="Country",
            orientation="h", title="Top 15 Countries by Revenue",
            color="Revenue", color_continuous_scale="Purples",
            template=PLOTLY_TEMPLATE,
        )
        fig_crev.update_layout(
            yaxis_title="", xaxis_tickprefix="£",
            xaxis_tickformat=",.0f", coloraxis_showscale=False,
        )
        st.plotly_chart(fig_crev, use_container_width=True)

    with col_r:
        top_ord_cntry = (
            sales.groupby("Country")["InvoiceNo"]
            .nunique().nlargest(15).reset_index().sort_values("InvoiceNo")
        )
        top_ord_cntry.columns = ["Country", "Orders"]
        fig_cord = px.bar(
            top_ord_cntry, x="Orders", y="Country",
            orientation="h", title="Top 15 Countries by Orders",
            color="Orders", color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE,
        )
        fig_cord.update_layout(yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_cord, use_container_width=True)

    st.subheader("Country Summary Table")
    cntry_tbl = (
        sales.groupby("Country")
        .agg(
            Revenue   = ("Revenue",    "sum"),
            Orders    = ("InvoiceNo",  "nunique"),
            Customers = ("CustomerID", "nunique"),
            Units     = ("Quantity",   "sum"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    cntry_tbl["Rev Share (%)"] = (cntry_tbl["Revenue"] / cntry_tbl["Revenue"].sum() * 100).round(1)
    cntry_tbl["Revenue"] = cntry_tbl["Revenue"].apply(lambda x: f"£{x:,.2f}")
    cntry_tbl["Units"]   = cntry_tbl["Units"].apply(lambda x: f"{int(x):,}")

    st.dataframe(cntry_tbl, use_container_width=True, hide_index=True)

    uk_share = cntry_tbl.loc[cntry_tbl["Country"] == "United Kingdom", "Rev Share (%)"]
    uk_pct   = float(uk_share.values[0]) if not uk_share.empty else 0

    insight(
        f"The United Kingdom accounts for ~**{uk_pct:.0f}%** of total revenue. "
        "Strong geographic concentration is a risk — expanding into Germany, France, "
        "and the BENELUX region offers the highest-probability growth corridors."
    )


# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — RETURNS / CANCELLATIONS
# ════════════════════════════════════════════════════════════════════════════

with tab_returns:
    st.subheader("Returns & Cancellations")

    if returns_filt.empty:
        st.info("No returns or cancellations found for the current filters.")
    else:
        r1, r2, r3 = st.columns(3)
        cancelled_df = returns_filt[returns_filt["IsCancelled"]]
        r1.metric("Cancelled Invoices",   fmt_number(cancelled_df["InvoiceNo"].nunique()))
        r2.metric("Units Returned",       fmt_number(abs(returns_filt["Quantity"].sum())))
        r3.metric("Revenue Impact (£)",   fmt_currency(returns_filt["Revenue"].sum()))

        st.markdown("---")

        col_l, col_r = st.columns(2)

        with col_l:
            ret_prod = (
                returns_filt.groupby("Description")["Quantity"]
                .sum().abs().nlargest(10).reset_index().sort_values("Quantity")
            )
            ret_prod.columns = ["Product", "Units Returned"]
            fig_rp = px.bar(
                ret_prod, x="Units Returned", y="Product",
                orientation="h", title="Top 10 Returned Products",
                color="Units Returned", color_continuous_scale="Reds",
                template=PLOTLY_TEMPLATE,
            )
            fig_rp.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig_rp, use_container_width=True)

        with col_r:
            ret_cntry = (
                returns_filt.groupby("Country")["Quantity"]
                .sum().abs().nlargest(10).reset_index().sort_values("Quantity")
            )
            ret_cntry.columns = ["Country", "Units Returned"]
            fig_rc = px.bar(
                ret_cntry, x="Units Returned", y="Country",
                orientation="h", title="Returns by Country",
                color="Units Returned", color_continuous_scale="Oranges",
                template=PLOTLY_TEMPLATE,
            )
            fig_rc.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig_rc, use_container_width=True)

        # Cancellations over time
        ret_time = (
            returns_filt[returns_filt["IsCancelled"]]
            .set_index("InvoiceDate")
            .resample("ME")["InvoiceNo"]
            .nunique()
            .reset_index()
        )
        ret_time.columns = ["Month", "Cancellations"]

        if not ret_time.empty:
            fig_rt = px.line(
                ret_time, x="Month", y="Cancellations",
                title="Monthly Cancellations Over Time",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[COLOR_RED],
            )
            fig_rt.update_layout(xaxis_title="", hovermode="x unified")
            st.plotly_chart(fig_rt, use_container_width=True)

        insight(
            f"Overall cancellation rate: **{kpis['cancel_rate']:.1f}%**. "
            "Products that appear in both the top sellers and top returns lists may have quality "
            "or fulfilment issues worth investigating. High return rates from specific countries "
            "may indicate shipping, description, or expectation-setting problems."
        )


# ════════════════════════════════════════════════════════════════════════════
#  TAB 7 — DATA QUALITY
# ════════════════════════════════════════════════════════════════════════════

with tab_quality:
    st.subheader("Data Quality Report")

    total = len(raw)

    checks = {
        "Total rows":                             total,
        "Missing CustomerID":                     int(raw["CustomerID"].isna().sum()),
        "Missing Description":                    int(raw["Description"].isna().sum()),
        "InvoiceDate parse errors":               int(raw["InvoiceDate"].isna().sum()),
        "Fully duplicate rows":                   int(raw.duplicated().sum()),
        "Negative Quantity (returns/adj.)":       int((raw["Quantity"] < 0).sum()),
        "Zero Unit Price":                        int((raw["UnitPrice"] == 0).sum()),
        "Cancelled Invoices (starts with C)":     int(raw["IsCancelled"].sum()),
        "Valid sales rows":                       len(sales_all),
    }

    qdf = pd.DataFrame(
        [{"Check": k, "Count": v, "% of Total": round(v / total * 100, 2)} for k, v in checks.items()]
    )

    def _status(row):
        if row["Check"] in ("Total rows", "Valid sales rows"):
            return "ℹ️ Info"
        if row["Count"] == 0:
            return "✅ Clean"
        if row["% of Total"] < 5:
            return "⚠️ Minor"
        return "🔴 Significant"

    qdf["Status"] = qdf.apply(_status, axis=1)
    st.dataframe(qdf, use_container_width=True, hide_index=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Missing Values by Column")
        miss = raw.isnull().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        miss["% Missing"] = (miss["Missing"] / total * 100).round(2)
        miss = miss[miss["Missing"] > 0]

        if miss.empty:
            st.success("No missing values detected.")
        else:
            fig_miss = px.bar(
                miss, x="Column", y="% Missing",
                title="Missing Values (%)",
                color="% Missing", color_continuous_scale="Reds",
                template=PLOTLY_TEMPLATE,
            )
            fig_miss.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_miss, use_container_width=True)

    with col_r:
        st.subheader("Row Breakdown")
        other_n = max(0, total - len(sales_all) - len(returns_all))
        dist = pd.DataFrame({
            "Category": ["Valid Sales", "Cancelled / Returned", "Other (zero price, etc.)"],
            "Count":    [len(sales_all), len(returns_all), other_n],
        })
        dist = dist[dist["Count"] > 0]
        fig_dist = px.pie(
            dist, names="Category", values="Count",
            title="Dataset Row Breakdown",
            color="Category",
            color_discrete_map={
                "Valid Sales":             COLOR_PRIMARY,
                "Cancelled / Returned":    COLOR_RED,
                "Other (zero price, etc.)": COLOR_SLATE,
            },
            template=PLOTLY_TEMPLATE,
            hole=0.4,
        )
        fig_dist.update_traces(textposition="outside", textinfo="percent+label")
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    missing_cid_pct = round(raw["CustomerID"].isna().sum() / total * 100, 1)
    insight(
        f"**{missing_cid_pct}%** of rows are missing a CustomerID. "
        "These rows are excluded from customer-level metrics and RFM segmentation. "
        "Improving checkout-to-account linkage would significantly increase analytical coverage."
    )
