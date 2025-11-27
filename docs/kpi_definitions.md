# KPI Definitions

## Average Order Value (AOV)

- AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)

## Gross Margin

- GM = SUM((UnitPrice - CostOfGoods) * Quantity * (1 - Discount))

- If cost is missing, approximate CostOfGoods as 70% of UnitPrice (CostOfGoods ≈ 0.7 * UnitPrice)
- This is a standard approximation when cost data is not available in the database.

