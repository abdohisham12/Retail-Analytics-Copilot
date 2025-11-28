"""Verify document corpus matches specifications"""
from pathlib import Path

docs_dir = Path('docs')
required_files = {
    'marketing_calendar.md': {
        'title': '# Northwind Marketing Calendar (1997)',
        'sections': ['Summer Beverages 1997', 'Winter Classics 1997']
    },
    'kpi_definitions.md': {
        'title': '# KPI Definitions',
        'sections': ['Average Order Value (AOV)', 'Gross Margin']
    },
    'catalog.md': {
        'title': '# Catalog Snapshot',
        'keywords': ['Beverages', 'Condiments', 'Confections', 'Dairy Products']
    },
    'product_policy.md': {
        'title': '# Returns & Policy',
        'keywords': ['Perishables', 'Beverages', 'Non-perishables']
    }
}

print("📄 Document Corpus Verification\n")
print("=" * 60)

all_good = True
for filename, checks in required_files.items():
    filepath = docs_dir / filename
    
    if not filepath.exists():
        print(f"❌ Missing: {filename}")
        all_good = False
        continue
    
    content = filepath.read_text(encoding='utf-8')
    
    # Check title
    if checks['title'] not in content:
        print(f"⚠️  {filename}: Title mismatch")
        all_good = False
    else:
        print(f"✅ {filename}")
        print(f"   Title: {checks['title']}")
    
    # Check sections/keywords
    if 'sections' in checks:
        for section in checks['sections']:
            if section in content:
                print(f"   ✓ Section: {section}")
            else:
                print(f"   ✗ Missing section: {section}")
                all_good = False
    
    if 'keywords' in checks:
        found = [kw for kw in checks['keywords'] if kw in content]
        print(f"   Keywords found: {len(found)}/{len(checks['keywords'])}")
    
    print()

if all_good:
    print("=" * 60)
    print("✅ All documents verified! Document corpus is ready for RAG.")
else:
    print("=" * 60)
    print("⚠️  Some issues found. Please review the documents.")

