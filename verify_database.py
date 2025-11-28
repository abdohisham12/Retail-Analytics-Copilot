"""Verify Northwind database structure"""
import sqlite3
from pathlib import Path

db_path = Path('data/northwind.sqlite')

if not db_path.exists():
    print("❌ Database not found at data/northwind.sqlite")
    print("📥 You need to download it from:")
    print("   https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db")
    exit(1)

print(f"✅ Database found: {db_path}")
print(f"   Size: {db_path.stat().st_size / 1024:.2f} KB\n")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [row[0] for row in cursor.fetchall()]

print("📊 Tables in database:")
for table in tables:
    print(f"   - {table}")

# Check required tables
required_tables = {
    "Orders": ["OrderID", "CustomerID", "EmployeeID", "OrderDate"],
    "Order Details": ["OrderID", "ProductID", "UnitPrice", "Quantity", "Discount"],
    "Products": ["ProductID", "ProductName", "SupplierID", "CategoryID", "UnitPrice"],
    "Customers": ["CustomerID", "CompanyName", "Country"],
    "Categories": [],
    "Suppliers": []
}

print("\n🔍 Verifying required tables and columns:")
all_good = True

for table_name, required_cols in required_tables.items():
    if table_name not in tables:
        print(f"   ❌ Missing table: {table_name}")
        all_good = False
        continue
    
    # Get columns for this table
    quoted_table = f'"{table_name}"' if ' ' in table_name else table_name
    try:
        cursor.execute(f"PRAGMA table_info({quoted_table})")
        columns = [row[1] for row in cursor.fetchall()]
    except:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
    
    print(f"   ✅ {table_name}: {len(columns)} columns")
    
    # Check required columns
    if required_cols:
        missing = [col for col in required_cols if col not in columns]
        if missing:
            print(f"      ⚠️  Missing columns: {missing}")
            all_good = False
        else:
            print(f"      ✅ All required columns present")

# Check sample data
print("\n📈 Sample data counts:")
for table in ["Orders", "Order Details", "Products", "Customers"]:
    if table in tables:
        quoted_table = f'"{table}"' if ' ' in table else table
        try:
            cursor.execute(f'SELECT COUNT(*) FROM {quoted_table}')
            count = cursor.fetchone()[0]
            print(f"   {table}: {count} rows")
        except Exception as e:
            print(f"   {table}: Error - {e}")

conn.close()

if all_good:
    print("\n✅ Database structure verified! All required tables and columns are present.")
else:
    print("\n⚠️  Some issues found. Please check the database.")

