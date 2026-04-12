# Irish Tax Calculator & Optimizer

A stateless, modular computation engine for Irish PAYE and self-employed tax under **2025 and 2026** rules. It does two things:

1. **Information engine** — tells you every tax credit and relief you're entitled to based on your life circumstances.
2. **Optimization engine** — uses SciPy SLSQP to find the optimal allocation across investment levers that maximizes your total financial utility.

## Requirements

- Python 3.10+
- scipy (`pip install scipy`)

## Architecture

The engine is built on two dataclasses (`UserProfile` and `Investments`) and a stateless calculator class (`IrishTaxCalculator`). All methods are `@staticmethod`—no mutable state, no side effects.

```
UserProfile (who you are + what you spent)
    │
    ▼
IrishTaxCalculator.calculate() ─── orchestrates the full pipeline
    │
    ├── calculate_taxable_base()
    ├── calculate_total_income()
    ├── calculate_taxable_paye_income()
    ├── calculate_net_income_tax()
    │     ├── get_tax_credits()        ◄── all reliefs aggregated here
    │     └── enforce_age_exemption()
    ├── calculate_prsi()
    ├── calculate_usc()
    └── calculate_take_home()
```

---

## What You Can Configure

### Things the optimizer controls (4 levers)

These are allocation decisions where you choose how much to spend. The optimizer finds the best split to maximize your utility. **Money stays with you** as assets or coverage.

| Lever | Tax Mechanism | Cap | What You Keep |
|---|---|---|---|
| **Pension** | Deducted at marginal rate (20-40%) | Age-based % of NRE, max €115k base | Pension fund (locked until retirement) |
| **Cycle to Work** | Pre-tax salary sacrifice | €1,500 regular / €3,000 e-bike (per 4 years) | The bicycle |
| **Travel Pass** | Pre-tax salary sacrifice | €1,830/year | Public transport pass |
| **Income Protection** | 20% tax credit on premiums | 10% of gross income | Insurance coverage |

### Things you claim relief on (expenses you already paid)

These aren't optimizer decisions — you already spent the money. Fill in the amounts and the engine calculates your tax saving.

| Expense | Relief | Cap/Rules |
|---|---|---|
| **Rent** | 20% credit | Capped at €1,000 single / €2,000 married |
| **Medical expenses** | 20% credit | GP visits, prescriptions, hospital, dental (non-routine) |
| **Nursing home fees** | 20% credit | For you or a dependent relative |
| **Health insurance (you pay)** | 20% credit | Premiums you pay yourself (not employer-paid) |
| **Tuition fees** | 20% credit | Above €3,000 disregard, max €7,000 qualifying |
| **Remote working** | 30% × (days/365) × 20% | Proportional share of utility bills |
| **Flat rate expenses** | Deducted from taxable income | Revenue-approved by occupation (e.g., nurse €733) |

### Automatic credits (yes/no life facts)

No cost. Just answer truthfully.

| Circumstance | Credit |
|---|---|
| Blind | €1,650 (single) / €2,500 (married) |
| Incapacitated child | €3,300 |
| Home carer (spouse) | €1,800 |
| Single parent child carer | €1,750 |
| Dependent relative | €245 |
| Widowed (years 0-5) | Tapered from €3,600 |
| Medical card + income < €60k | Reduced USC rates |

### Manual passthrough fields (niche)

These are set by the user if applicable. The engine factors them into the tax calculation but the optimizer does not touch them.

| Field | Tax Mechanism | Who Uses This |
|---|---|---|
| Voucher | Tax-free employer benefit, up to €1,500 | Anyone whose employer offers Small Benefit Exemption |
| EIIS | Deducted at marginal rate | Investors in qualifying Irish companies |
| Deeds of Covenant | Deducted at marginal rate, cap 5% income | Legal covenant to an individual |
| Charitable Donations | Deducted at marginal rate (self-employed only) | Self-employed donors, min €250 |

---

## Usage

```python
from tax_calculator import IrishTaxCalculator, UserProfile, Investments

profile = UserProfile(
    gross_income=65000.0,
    age=32,
    employment_type="PAYE",
    marital_status="Single",
    annual_rent_paid=12000.0,
    qualifying_health_expenses=800.0,
    nursing_home_fees=0.0,
    employee_health_insurance=1800.0,
    employer_health_premium=1200.0,
    tax_year=2026,
)

investments = Investments(
    pension_contribution=0.0,
    cycle_to_work=0.0,
    cycle_type="ebike",
    travel_pass=0.0,
    income_protection_premium=0.0,
    voucher_allocation=1500.0,
)

# Run the optimizer
optimal = IrishTaxCalculator.optimize(
    profile,
    investments,
    required_liquid_cash=30000.0,
    utility_weight_pension=1.2,
    utility_weight_cycle=0.85,
    utility_weight_travel=0.95,
    utility_weight_income_protection=0.85,
)

# Or just calculate without optimizing
result = IrishTaxCalculator.calculate(profile, investments)
```

### Optimizer Weights

Weights tell the optimizer how much you subjectively value €1 locked into each lever:

| Weight | Meaning |
|---|---|
| `> 1.0` | You value it **more** than €1 cash (e.g., pension with employer match) |
| `= 1.0` | You value it **equal** to €1 cash |
| `< 1.0` | You value it **less** than €1 cash (e.g., a bike you don't really need) |
| `= 0.0` | Never allocate to this lever |

### Marginal Rate Curve

Generate a table showing where tax bracket kinks occur:

```python
IrishTaxCalculator.print_marginal_curve(profile, investments)
```

### Liquidity Constraint

Set `required_liquid_cash` to enforce a minimum take-home floor. The optimizer will constrain its allocations so your cash never drops below that amount. If the floor is impossible (exceeds your post-tax income with zero deductions), the optimizer rejects immediately.

---

## Tax Year Coverage

The `TAX_REGISTRY` contains statutory rates for 2025 and 2026 including:

- Income tax brackets (20% / 40%)
- Standard Rate Cut-Off Points (single, married 1-income, married 2-income)
- USC bands and rates (including medical card exemptions)
- PRSI rates with step-reduction tapering
- All personal, employment, and earned income credits

Set `tax_year=2025` or `tax_year=2026` in `UserProfile`.

## Disclaimer

This is an algorithmic projection tool. It does not constitute accredited Irish Revenue financial advice. Always verify with a qualified tax advisor.
