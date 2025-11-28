from agent.graph_hybrid import _build_fallback_sql
from agent.tools.sqlite_tool import SQLiteTool
sql = _build_fallback_sql('hybrid_top_category_qty_summer_1997', SQLiteTool(), {'question': 'During Summer Beverages 1997 ...'})
with open('debug_output.txt', 'w', encoding='utf-8') as f:
    f.write(f'sql is None? {sql is None}\n')
    f.write(str(sql))
