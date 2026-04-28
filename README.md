# E-Commerce Analytics Dashboard

A production-ready business intelligence dashboard built with **Streamlit** and **Plotly**, analysing a real-world e-commerce dataset (~540 k transactions).

---

## What It Does

| Section | Content |
|---|---|
| **Executive Overview** | Monthly revenue trend, top-5 products & countries |
| **Sales Trends** | Daily / weekly / monthly revenue, orders, units sold, and a Year × Month heatmap |
| **Products** | Top-N products by revenue and quantity; full searchable product table |
| **Customers** | Top customers by revenue and orders; full RFM segmentation (Champions, Loyal, At Risk, New, Low Value) |
| **Countries** | Top-15 countries by revenue and orders; full country summary table |
| **Returns** | Cancelled invoices and returns by product, country, and over time |
| **Data Quality** | Missing values, duplicates, zero prices, cancelled rows, row breakdown |

Sidebar controls let you filter by **date range**, **country**, and **product keyword** at any time.

---

## Dataset

The dashboard expects a file named **`data.csv`** in the project root.  
It uses the [UCI Online Retail dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) (or any CSV with the same schema):

| Column | Type | Notes |
|---|---|---|
| `InvoiceNo` | string | Invoices starting with `C` = cancellation |
| `StockCode` | string | Product identifier |
| `Description` | string | Product name |
| `Quantity` | int | Negative = return / adjustment |
| `InvoiceDate` | datetime | Format `M/D/YYYY H:MM` |
| `UnitPrice` | float | Per-unit price in £ |
| `CustomerID` | float | Can be missing |
| `Country` | string | Customer country |

Place `data.csv` in the **same directory as `app.py`** before running.

---

## Run Locally

```bash
# 1. Clone / download the project
git clone <repo-url>
cd ecommerce_dashboard

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place data.csv in the project root, then launch
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Deploy to Streamlit Community Cloud

1. Push the project to a **public GitHub repository** (ensure `data.csv` is included).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select your repo, branch (`main`), and set the **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud automatically installs `requirements.txt`.

> **File size note:** `data.csv` is ~45 MB. GitHub's 100 MB per-file limit is not an issue, but if you use Git LFS you may need to configure it for `.csv` files.

---

## Project Structure

```
ecommerce_dashboard/
├── app.py               # Main Streamlit application
├── data.csv             # Dataset (not tracked in repo — add manually)
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .streamlit/
    └── config.toml      # Theme and server configuration
```

---

## Tech Stack

- [Streamlit](https://streamlit.io) — dashboard framework
- [Plotly Express](https://plotly.com/python/plotly-express/) — interactive charts
- [Pandas](https://pandas.pydata.org) — data manipulation
- [NumPy](https://numpy.org) — numerical helpers
