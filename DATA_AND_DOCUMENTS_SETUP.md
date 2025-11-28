# Data & Documents Setup Guide

## ✅ Current Status

### Database: ✅ VERIFIED
- **Location:** `data/northwind.sqlite`
- **Size:** 24.1 MB
- **Status:** All required tables and columns present

### Documents: ✅ VERIFIED
- **Location:** `docs/`
- **Status:** All 4 required documents present

---

## 📊 Database Verification

### Required Tables (Canonical Northwind Names)

#### ✅ Orders
- **Columns:** OrderID, CustomerID, EmployeeID, OrderDate, RequiredDate, ShippedDate, ShipVia, Freight, ShipName, ShipAddress, ShipCity, ShipRegion, ShipPostalCode, ShipCountry
- **Row Count:** 16,282 rows
- **Status:** ✅ Verified

#### ✅ "Order Details" (Note: Space in name, must be quoted)
- **Columns:** OrderID, ProductID, UnitPrice, Quantity, Discount
- **Row Count:** 609,283 rows
- **Status:** ✅ Verified
- **⚠️ Important:** This table name has a space and MUST be quoted in SQL: `"Order Details"`

#### ✅ Products
- **Columns:** ProductID, ProductName, SupplierID, CategoryID, UnitPrice, QuantityPerUnit, UnitsInStock, UnitsOnOrder, ReorderLevel, Discontinued
- **Row Count:** 77 rows
- **Status:** ✅ Verified

#### ✅ Customers
- **Columns:** CustomerID, CompanyName, ContactName, ContactTitle, Address, City, Region, PostalCode, Country, Phone, Fax
- **Row Count:** 93 rows
- **Status:** ✅ Verified

#### ✅ Categories (Optional, but present)
- **Columns:** CategoryID, CategoryName, Description, Picture
- **Status:** ✅ Verified

#### ✅ Suppliers (Optional, but present)
- **Columns:** SupplierID, CompanyName, ContactName, ContactTitle, Address, City, Region, PostalCode, Country, Phone, Fax, HomePage
- **Status:** ✅ Verified

---

## 📄 Documents Verification

### ✅ docs/kpi_definitions.md
**Content:**
- Average Order Value (AOV) formula
- Gross Margin formula
- Cost approximation rules

**Status:** ✅ Present and correct

### ✅ docs/marketing_calendar.md
**Content:**
- Summer Beverages 1997 (1997-06-01 to 1997-06-30)
- Winter Classics 1997 (1997-12-01 to 1997-12-31)

**Status:** ✅ Present and correct

### ✅ docs/product_policy.md
**Content:**
- Return policies by product category
- Return windows (3-7 days, 14 days, 30 days)

**Status:** ✅ Present and correct

### ✅ docs/catalog.md
**Content:**
- Product categories list
- Category mapping information

**Status:** ✅ Present and correct

---

## 🔧 Setup Instructions (If Starting Fresh)

### Step 1: Download Northwind Database

If the database doesn't exist, download it:

```bash
# Create data directory
mkdir -p data

# Download Northwind SQLite database
# Option 1: Using curl (Linux/Mac)
curl -o data/northwind.sqlite https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

# Option 2: Using PowerShell (Windows)
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db" -OutFile "data/northwind.sqlite"

# Option 3: Using Python
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db', 'data/northwind.sqlite')"
```

### Step 2: Verify Database

Run the verification script:

```bash
python verify_database.py
```

Expected output:
```
✅ Database found: data/northwind.sqlite
✅ All required tables and columns are present
```

### Step 3: Verify Documents

Check that all documents exist:

```bash
# List documents
ls docs/

# Should show:
# - catalog.md
# - kpi_definitions.md
# - marketing_calendar.md
# - product_policy.md
```

---

## 🧪 Quick Test Queries

### Test Database Connection

```python
from agent.tools.sqlite_tool import SQLiteTool

tool = SQLiteTool()
result = tool.execute_query('SELECT COUNT(*) as count FROM Orders')
print(result)
# Should show: {'results': [{'count': 16282}], 'columns': ['count'], 'row_count': 1, 'error': None}
```

### Test "Order Details" Table (Quoted)

```python
from agent.tools.sqlite_tool import SQLiteTool

tool = SQLiteTool()
result = tool.execute_query('SELECT COUNT(*) as count FROM "Order Details"')
print(result)
# Should show: {'results': [{'count': 609283}], 'columns': ['count'], 'row_count': 1, 'error': None}
```

### Test RAG Retrieval

```python
from agent.rag.retrieval import RAGRetrieval

rag = RAGRetrieval()
citations, chunk_ids, scores = rag.search("What is AOV?")
print(f"Found {len(citations)} citations")
```

---

## ⚠️ Important Notes

### 1. "Order Details" Table Name
- **CRITICAL:** The table name has a space: `Order Details`
- **MUST be quoted in SQL:** `"Order Details"`
- **Example:**
  ```sql
  SELECT * FROM "Order Details" WHERE OrderID = 10248;
  ```
- **NOT:** `SELECT * FROM Order Details` ❌ (will fail)

### 2. Revenue Calculation
- **Formula:** `SUM(UnitPrice * Quantity * (1 - Discount))`
- **From:** `"Order Details"` table
- **Example:**
  ```sql
  SELECT SUM(UnitPrice * Quantity * (1 - Discount)) as Revenue
  FROM "Order Details"
  WHERE OrderID = 10248;
  ```

### 3. Database Schema
- The schema is automatically introspected using `PRAGMA table_info()`
- The `SQLiteTool.get_schema_string()` method provides a formatted schema for LLM prompts
- Schema includes relationship hints (foreign keys)

---

## 📋 Checklist for HR Assessment

- [x] Database exists at `data/northwind.sqlite`
- [x] All required tables present (Orders, "Order Details", Products, Customers)
- [x] All required columns present in each table
- [x] Sample data loaded (16K+ orders, 600K+ order details)
- [x] Documents present in `docs/` directory
- [x] RAG system can retrieve from documents
- [x] SQL tool can query database
- [x] "Order Details" table properly handled (quoted)

---

## 🚀 Next Steps

1. **Run Full Test:**
   ```bash
   python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
   ```

2. **Interactive Test:**
   ```bash
   python run_agent_hybrid.py
   ```

3. **Verify Output Format:**
   - Check that answers include citations
   - Verify SQL queries are properly formatted
   - Ensure "Order Details" is quoted in SQL

---

## 📞 Troubleshooting

### Database Not Found
```bash
# Check if database exists
ls data/northwind.sqlite

# If missing, download it (see Step 1 above)
```

### "Order Details" Query Fails
- Ensure table name is quoted: `"Order Details"`
- Check SQLiteTool handles quoted names correctly
- Verify schema introspection works

### Documents Not Indexed
- First run will index documents automatically
- Check `vector_store/` directory exists
- Re-run to trigger indexing if needed

---

## ✅ Summary

**Everything is set up correctly!**

- ✅ Database: 24.1 MB, all tables and columns verified
- ✅ Documents: All 4 documents present and indexed
- ✅ System ready for HR assessment

You can proceed with testing and evaluation.

