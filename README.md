# Irish Tax Calculator & Optimizer 🇮🇪 

A robust, mathematically rigorous, stateless computation engine built in Python to accurately simulate Irish net take-home pay under **2025 and 2026** tax standards. 

This is not just a standard linear calculator—it features a built-in **SciPy optimization engine (`SLSQP`)** designed to intelligently structure and balance your financial deductions (Pensions, EIIS, Deeds of Covenant, etc.) against your marginal tax utility limits without depleting your post-tax liquidity.

## 🚀 Features

- **Accurate Core Taxation:** Calculates standard PAYE (20%/40%), Universal Social Charge (USC) banding limits, and the accurate PRSI step-tapering relief.
- **Micro-Generation & Rent:** Incorporates exact cliff-edge math logic for Rent-a-Room boundaries (€14k cap) and grid energy sellback exemptions.
- **Multidimensional Optimization Engine:** Uses the SciPy SLSQP formula to greedily identify exactly how many euros you should legally park across:
  - Pension Contributions (Mapped against Net Relevant Earnings / age bounds)
  - EIIS Investments
  - Deeds of Covenant (Mapped against Total Income bounds)
  - Cycle to Work Schemes
- **Stateless Functional Architecture:** Designed purely around `dataclasses`. Eliminates God-Object bloat by keeping configuration isolated and strictly returning clean data dictionaries without permanently mutating underlying models.
- **Smart Edge-Case Exceptions:** Implements absolute Income Tax exemptions for age 65+ including marginal recovery logic.

## 📦 Requirements

- Python 3.10+
- Scipy (`pip install scipy`)

## ⚙️ Usage Configuration

The execution pipeline has been stripped of parameter bloat and requires exactly two inputs to compute your financial payload: a `UserProfile` data class, and an `Investments` data class.

```python
from tax_calculator import IrishTaxCalculator, UserProfile, Investments

# 1. Provide your standard facts
my_profile = UserProfile(
    gross_income=65000.0,
    age=32,
    employment_type="PAYE",
    marital_status="Single",
    annual_rent_paid=8000.0, # Automatically calculates the strict 20% cap 
    tax_year=2026
)

# 2. Provide your dynamic cash decisions
my_investments = Investments(
    pension_contribution=0.0, # Let the optimizer figure this out
    voucher_allocation=1500.0, # Tax-free employer provision
    eiis_investment=0.0
)

# 3. Fire the optimizer to restructure your deductions for max efficiency
optimal_investments = IrishTaxCalculator.optimize(
    profile=my_profile, 
    base_investments=my_investments, 
    utility_weight_pension=1.2, 
    utility_weight_eiis=0.8
)
```

### 💡 Marginal Rate Simulation
You can trivially generate a full mathematical grid that evaluates exactly where your "kinks" or marginal tax cliffs appear across the board:

```python
curve = IrishTaxCalculator.print_marginal_curve(my_profile, my_investments)
```

## ⚖️ Legal Disclaimer
This underlying projection repository handles algorithmic modeling strictly dynamically. It does not constitute actual accredited Irish Revenue financial advice.
