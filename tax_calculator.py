import json
from dataclasses import dataclass, replace
from scipy.optimize import minimize

TAX_REGISTRY = {
    2025: {
        "PERSONAL_CREDIT": 2000.0,
        "EMPLOYMENT_CREDIT": 2000.0, 
        "EARNED_INCOME_CREDIT": 2000.0,
        "AGE_CREDIT_SINGLE": 245.0,
        "AGE_CREDIT_MARRIED": 490.0,
        "BLIND_CREDIT_SINGLE": 1950.0,
        "BLIND_CREDIT_MARRIED": 3900.0,
        "INCAPACITATED_CHILD_CREDIT": 3800.0,
        "DEPENDENT_RELATIVE_CREDIT": 305.0,
        "HOME_CARER_CREDIT": 1950.0,
        "SINGLE_CHILD_CARER_CREDIT": 1900.0,
        "SRCOP_SINGLE": 44000.0,
        "SRCOP_MARRIED_BASE": 53000.0,
        "SRCOP_UPLIFT_MAX": 35000.0,
        "INCOME_TAX_STD_RATE": 0.20,
        "INCOME_TAX_HIGH_RATE": 0.40,
        "PRSI_THRESHOLD": 18304.0, 
        "PRSI_EXEMPT_AGE": 66,
        "USC_EXEMPT_THRESHOLD": 13000.0,
        "USC_BAND_1_LIMIT": 12012.0,
        "USC_BAND_3_LIMIT": 70044.0,
        "USC_BAND_2_LIMIT": 27382.0,
        "PRSI_RATE": 0.041
    },
    2026: {
        "PERSONAL_CREDIT": 2000.0,
        "EMPLOYMENT_CREDIT": 2000.0, 
        "EARNED_INCOME_CREDIT": 2000.0,
        "AGE_CREDIT_SINGLE": 245.0,
        "AGE_CREDIT_MARRIED": 490.0,
        "BLIND_CREDIT_SINGLE": 1950.0,
        "BLIND_CREDIT_MARRIED": 3900.0,
        "INCAPACITATED_CHILD_CREDIT": 3800.0,
        "DEPENDENT_RELATIVE_CREDIT": 305.0,
        "HOME_CARER_CREDIT": 1950.0,
        "SINGLE_CHILD_CARER_CREDIT": 1900.0,
        "SRCOP_SINGLE": 44000.0,
        "SRCOP_MARRIED_BASE": 53000.0,
        "SRCOP_UPLIFT_MAX": 35000.0,
        "INCOME_TAX_STD_RATE": 0.20,
        "INCOME_TAX_HIGH_RATE": 0.40,
        "PRSI_THRESHOLD": 18304.0, 
        "PRSI_EXEMPT_AGE": 66,
        "USC_EXEMPT_THRESHOLD": 13000.0,
        "USC_BAND_1_LIMIT": 12012.0,
        "USC_BAND_3_LIMIT": 70044.0,
        "USC_BAND_2_LIMIT": 28700.0,
        "PRSI_RATE": 0.0435
    }
}

@dataclass
class UserProfile:
    gross_income: float
    age: int = 30
    marital_status: str = "Single"
    employment_type: str = "PAYE"
    medical_card: bool = False
    second_income: float = 0.0
    rent_a_room_income: float = 0.0
    micro_generation_income: float = 0.0
    annual_rent_paid: float = 0.0
    qualifying_health_expenses: float = 0.0
    bik: float = 0.0
    employer_health_premium: float = 0.0
    additional_tax_credits: float = 0.0
    is_blind: bool = False
    has_incapacitated_child: bool = False
    claims_home_carer: bool = False
    claims_single_child_carer: bool = False
    claims_dependent_relative: bool = False
    widowed_years_since: int = -1
    tax_year: int = 2026

@dataclass
class Investments:
    pension_contribution: float = 0.0
    voucher_allocation: float = 0.0
    cycle_to_work: float = 0.0
    cycle_type: str = "regular"
    cycle_to_work_mode: str = "annual"
    travel_pass: float = 0.0
    eiis_investment: float = 0.0
    deeds_of_covenant: float = 0.0

class IrishTaxCalculator:

    @staticmethod
    def get_srcop(profile: UserProfile, cfg: dict) -> float:
        """
        Standard Rate Cut-Off Point (SRCOP) determines how much income is taxed at 20% 
        before the 40% higher rate kicks in. Single filers get a flat SRCOP (e.g. €44,000).
        Married one-income couples get a higher base (e.g. €53,000). Married two-income 
        couples get the base plus an uplift equal to the lesser of the second income or 
        the uplift cap (e.g. €35,000), meaning the SRCOP can stretch up to €88,000.
        """
        if profile.marital_status == "Married_1_Income":
            return cfg["SRCOP_MARRIED_BASE"]
        elif profile.marital_status == "Married_2_Incomes":
            uplift = min(cfg["SRCOP_UPLIFT_MAX"], profile.second_income)
            return cfg["SRCOP_MARRIED_BASE"] + uplift
        return cfg["SRCOP_SINGLE"]

    @staticmethod
    def calculate_usc(profile: UserProfile, total_income: float, cfg: dict) -> tuple[float, float]:
        """
        Universal Social Charge is a progressive tax applied in 4 bands:
        Band 1: 0.5% on the first €12,012.
        Band 2: 2% from €12,012 up to €28,700 (2026).
        Band 3: 3% from €28,700 up to €70,044.
        Band 4: 8% on everything above €70,044.
        If total income is below €13,000, USC is fully exempt.
        Medical card holders earning under €60,000 get a reduced two-band schedule (0.5% + 2%).
        Self-employed earners above €100,000 pay an additional 3% surcharge on the excess.
        Returns (total_usc, marginal_usc_rate) so the optimizer knows the slope.
        """
        if total_income <= cfg["USC_EXEMPT_THRESHOLD"]:
            return 0.0, 0.0

        if profile.medical_card and total_income <= 60000.0:
            b1 = min(total_income, cfg["USC_BAND_1_LIMIT"]) * 0.005
            b2 = max(0, total_income - cfg["USC_BAND_1_LIMIT"]) * 0.02
            marginal = 0.02 if total_income > cfg["USC_BAND_1_LIMIT"] else 0.005
            return b1 + b2, marginal

        usc_tax = 0.0
        marginal = 0.0
        
        usc_tax += min(total_income, cfg["USC_BAND_1_LIMIT"]) * 0.005
        if total_income <= cfg["USC_BAND_1_LIMIT"]: marginal = 0.005

        if total_income > cfg["USC_BAND_1_LIMIT"]:
            taxable = min(total_income, cfg["USC_BAND_2_LIMIT"]) - cfg["USC_BAND_1_LIMIT"]
            usc_tax += taxable * 0.02
            if total_income <= cfg["USC_BAND_2_LIMIT"]: marginal = 0.02

        if total_income > cfg["USC_BAND_2_LIMIT"]:
            taxable = min(total_income, cfg["USC_BAND_3_LIMIT"]) - cfg["USC_BAND_2_LIMIT"]
            usc_tax += taxable * 0.03
            if total_income <= cfg["USC_BAND_3_LIMIT"]: marginal = 0.03

        if total_income > cfg["USC_BAND_3_LIMIT"]:
            taxable = total_income - cfg["USC_BAND_3_LIMIT"]
            usc_tax += taxable * 0.08
            marginal = 0.08
            
            if profile.employment_type == "Self-Employed" and total_income > 100000.0:
                usc_tax += (total_income - 100000.0) * 0.03
                marginal = 0.11

        return usc_tax, marginal

    @staticmethod
    def calculate_prsi(profile: UserProfile, total_income: float, cfg: dict) -> tuple[float, float]:
        """
        Pay Related Social Insurance (PRSI) is a flat-rate charge on all income above €18,304.
        Workers aged 66+ are fully exempt. Below the threshold, PRSI is zero.
        Above the threshold, gross PRSI = total_income * rate (4.35% in 2026).
        However, Revenue applies a taper relief credit worth up to €624/year (€12/week) for 
        incomes just above the threshold. The credit reduces by €1 for every €6 earned over 
        €18,304, fully exhausting at ~€22,060. This prevents a brutal cliff where earning €1 
        over the threshold would instantly trigger ~€800 in PRSI. While the taper is active, 
        the effective marginal PRSI rate is rate + 1/6 (~21%) because each additional euro 
        both incurs PRSI and erodes the credit.
        Returns (net_prsi, marginal_prsi_rate).
        """
        if profile.age >= cfg["PRSI_EXEMPT_AGE"]:
            return 0.0, 0.0
        if total_income <= cfg["PRSI_THRESHOLD"]:
            return 0.0, 0.0
            
        rate = cfg["PRSI_RATE"]
        gross_prsi = total_income * rate
        
        annual_credit = max(0.0, 624.0 - ((total_income - cfg["PRSI_THRESHOLD"]) / 6.0))
        net_prsi = max(0.0, gross_prsi - annual_credit)
        
        prsi_marginal = rate
        if annual_credit > 0.0:
            prsi_marginal = rate + (1.0 / 6.0)
            
        return net_prsi, prsi_marginal

    @staticmethod
    def _calculate_rent_credit(profile: UserProfile) -> float:
        """
        Rent Tax Credit: 20% of actual annual rent paid, capped at €1,000 for singles 
        or €2,000 for jointly assessed couples. If you pay €500/year rent, you get a 
        €100 credit. If you pay €10,000/year, you hit the cap at €1,000.
        """
        rent_credit_cap = 2000.0 if profile.marital_status in ["Married_1_Income", "Married_2_Incomes"] else 1000.0
        return min(rent_credit_cap, profile.annual_rent_paid * 0.20)

    @staticmethod
    def get_tax_credits(profile: UserProfile, cfg: dict) -> float:
        """
        Tax credits are subtracted directly from your gross income tax bill (not from 
        taxable income). Every taxpayer gets a Personal Credit (€2,000). PAYE workers get 
        an additional Employment Credit (€2,000); self-employed get an Earned Income Credit 
        instead. Married couples double the Personal Credit to €4,000. Various binary 
        life-circumstance credits (blind, incapacitated child, home carer, etc.) are stacked 
        additively. Widowed parents receive a tapered credit starting at €3,600 in year 0, 
        reducing by €360/year over 5 years. Medical expenses and employer health insurance 
        premiums generate a 20% flat credit. Rent credit is calculated separately via 
        _calculate_rent_credit(). The sum of all credits is subtracted from gross income tax 
        to produce the net income tax liability (floored at zero).
        """
        credits = cfg["PERSONAL_CREDIT"]
        
        if profile.employment_type == "PAYE":
            credits += cfg["EMPLOYMENT_CREDIT"]
        elif profile.employment_type == "Self-Employed":
            credits += cfg["EARNED_INCOME_CREDIT"]
            
        if profile.marital_status in ["Married_1_Income", "Married_2_Incomes"]:
            credits += cfg["PERSONAL_CREDIT"]
            if profile.age >= 65: credits += cfg["AGE_CREDIT_MARRIED"]
            if profile.is_blind: credits += cfg["BLIND_CREDIT_MARRIED"]
        else:
            if profile.age >= 65: credits += cfg["AGE_CREDIT_SINGLE"]
            if profile.is_blind: credits += cfg["BLIND_CREDIT_SINGLE"]

        if profile.has_incapacitated_child: credits += cfg["INCAPACITATED_CHILD_CREDIT"]
        if profile.claims_home_carer: credits += cfg["HOME_CARER_CREDIT"]
        if profile.claims_single_child_carer: credits += cfg["SINGLE_CHILD_CARER_CREDIT"]
        if profile.claims_dependent_relative: credits += cfg["DEPENDENT_RELATIVE_CREDIT"]
        
        if 0 <= profile.widowed_years_since <= 5:
            credits += max(0, 3600.0 - (profile.widowed_years_since * 360.0))
            
        credits += profile.employer_health_premium * 0.20
        credits += profile.qualifying_health_expenses * 0.20
        
        return credits + profile.additional_tax_credits + IrishTaxCalculator._calculate_rent_credit(profile)

    @staticmethod
    def get_max_pension_limit(profile: UserProfile) -> float:
        """
        Maximum tax-relieved pension contribution. The basis is Net Relevant Earnings (NRE), 
        which includes gross salary, BIK, and employer health premiums, capped at €115,000. 
        An age-dependent percentage is applied to this capped figure:
        Under 30: 15%, 30-39: 20%, 40-49: 25%, 50-54: 30%, 55-59: 35%, 60+: 40%.
        Example: a 32-year-old earning €50,000 with €1,200 employer health premium has 
        NRE = €51,200, so the limit is €51,200 * 0.20 = €10,240.
        """
        total_remuneration = profile.gross_income + profile.bik + profile.employer_health_premium
        limit_salary = min(total_remuneration, 115000.0)
        
        if profile.age < 30: pct = 0.15
        elif profile.age < 40: pct = 0.20
        elif profile.age < 50: pct = 0.25
        elif profile.age < 55: pct = 0.30
        elif profile.age < 60: pct = 0.35
        else: pct = 0.40
            
        return limit_salary * pct

    @staticmethod
    def get_max_cycle_to_work_limit(investments: Investments) -> float:
        """
        Cycle to Work scheme cap. E-bikes allow up to €3,000; regular bikes up to €1,500. 
        In annual mode the cap is spread over 4 years (€750/year for e-bikes), reflecting 
        the Revenue rule that the scheme can only be used once every 4 years.
        """
        cap = 3000.0 if investments.cycle_type == "ebike" else 1500.0
        if investments.cycle_to_work_mode == "annual":
            return cap / 4.0
        return cap

    @staticmethod
    def split_micro_generation_income(income: float) -> tuple[float, float]:
        """
        Micro-generation income (selling electricity back to the grid) is exempt up to 
        €400/year. Any amount above €400 becomes taxable. Returns (taxable, tax_free).
        Example: €600 income -> (€200 taxable, €400 tax-free).
        """
        taxable = max(0, income - 400.0)
        tax_free = min(income, 400.0)
        return taxable, tax_free

    @staticmethod
    def split_rent_a_room_income(income: float) -> tuple[float, float]:
        """
        Rent-a-Room relief is a strict cliff-edge, not a gradual threshold. If total 
        rental income from a qualifying room is at or below €14,000, the entire amount 
        is tax-free. If it exceeds €14,000 by even €1, the ENTIRE amount becomes taxable 
        (not just the excess). Returns (taxable, tax_free).
        """
        if income > 14000.0:
            return income, 0.0
        return 0.0, income
        
    @staticmethod
    def calculate_gross_income_tax(taxable_paye_income: float, srcop: float, cfg: dict) -> tuple[float, float]:
        """
        Irish income tax uses a two-bracket system. Income up to the SRCOP is taxed at 
        the standard rate (20%). Everything above the SRCOP is taxed at the higher rate 
        (40%). The marginal rate returned tells the optimizer which bracket the next euro 
        of income falls into. Returns (gross_income_tax, marginal_rate).
        """
        tax_20_bracket = min(taxable_paye_income, srcop) * cfg["INCOME_TAX_STD_RATE"]
        tax_40_bracket = max(0, taxable_paye_income - srcop) * cfg["INCOME_TAX_HIGH_RATE"]
        gross_income_tax = tax_20_bracket + tax_40_bracket
        marginal_income_tax_rate = cfg["INCOME_TAX_HIGH_RATE"] if taxable_paye_income > srcop else cfg["INCOME_TAX_STD_RATE"]
        return gross_income_tax, marginal_income_tax_rate

    @staticmethod
    def enforce_age_exemption(profile: UserProfile, total_income: float, current_net_tax: float) -> float:
        """
        Taxpayers aged 65+ are fully exempt from income tax if their total income is 
        below €18,000 (single) or €36,000 (married). If income slightly exceeds the 
        limit, marginal relief applies: the tax is capped at 40% of the amount over the 
        limit, preventing a scenario where earning €1 over the threshold triggers a full 
        tax bill. The lower of (normal tax, marginal cap) is used.
        """
        if profile.age >= 65:
            exemption_limit = 36000.0 if profile.marital_status in ["Married_1_Income", "Married_2_Incomes"] else 18000.0
            if total_income <= exemption_limit:
                return 0.0
            else:
                marginal_tax_cap = (total_income - exemption_limit) * 0.40
                return min(current_net_tax, marginal_tax_cap)
        return current_net_tax

    @staticmethod
    def calculate_taxable_base(profile: UserProfile, investments: Investments) -> float:
        """
        Starting point for all tax calculations. Takes gross salary and subtracts 
        legitimate pre-tax salary sacrifice schemes (Cycle to Work, Travel Pass). 
        The voucher is excluded here because it is an employer-side tax-free benefit, 
        not a salary deduction. The result is the income base that flows into PAYE/USC/PRSI.
        """
        return max(0, profile.gross_income - investments.cycle_to_work - investments.travel_pass)

    @staticmethod
    def calculate_total_income(taxable_base: float, profile: UserProfile) -> float:
        """
        Total income for PRSI and USC purposes. Takes the taxable base and adds Benefits 
        in Kind (company car, employer health insurance) and the taxable portions of 
        ancillary income streams (micro-generation above €400, rent-a-room above €14,000). 
        This figure is what PRSI and USC are charged against, before any pension/EIIS deductions.
        """
        total_bik = profile.bik + profile.employer_health_premium
        taxable_micro_gen, _ = IrishTaxCalculator.split_micro_generation_income(profile.micro_generation_income)
        taxable_rent_a_room, _ = IrishTaxCalculator.split_rent_a_room_income(profile.rent_a_room_income)
        return taxable_base + total_bik + taxable_micro_gen + taxable_rent_a_room

    @staticmethod
    def calculate_taxable_paye_income(total_income: float, investments: Investments) -> float:
        """
        Income subject to PAYE income tax brackets. This is total income minus 
        tax-deductible investments: pension contributions (up to age-based limit), 
        EIIS investments (tax relief at marginal rate), and Deeds of Covenant (up to 5% 
        of total income). These deductions push income down the bracket ladder, potentially 
        moving euros from the 40% band back into the 20% band.
        """
        return max(0, total_income - investments.pension_contribution - investments.eiis_investment - investments.deeds_of_covenant)

    @staticmethod
    def calculate_net_income_tax(profile: UserProfile, taxable_paye_income: float, total_income: float, cfg: dict) -> tuple[float, float, float, float]:
        """
        Orchestrates the full income tax pipeline:
        1. Calculates gross income tax by applying the 20%/40% brackets to taxable PAYE income.
        2. Sums all applicable tax credits (personal, employment, medical, rent, etc.).
        3. Subtracts credits from gross tax (floored at zero) to get net income tax.
        4. Applies the age exemption override if the taxpayer is 65+.
        Returns (gross_income_tax, net_income_tax, total_credits, marginal_income_tax_rate).
        """
        srcop = IrishTaxCalculator.get_srcop(profile, cfg)
        gross_income_tax, marginal_income_tax_rate = IrishTaxCalculator.calculate_gross_income_tax(taxable_paye_income, srcop, cfg)
        total_credits = IrishTaxCalculator.get_tax_credits(profile, cfg)
        net_income_tax = max(0, gross_income_tax - total_credits)
        net_income_tax = IrishTaxCalculator.enforce_age_exemption(profile, total_income, net_income_tax)
        return gross_income_tax, net_income_tax, total_credits, marginal_income_tax_rate

    @staticmethod
    def calculate_take_home(profile: UserProfile, investments: Investments, taxable_base: float, total_taxes: float) -> float:
        """
        Computes actual liquid cash landing in your bank account.
        Starts with taxable_base, subtracts all investment outflows (pension, EIIS, deeds) 
        and total taxes (PAYE + USC + PRSI). Then adds back ancillary income streams 
        (rent-a-room and micro-generation flow through regardless of their tax treatment). 
        Subtracts out-of-pocket health expenses (the 20% credit was already applied against 
        the tax bill, but you still spent the cash). Finally adds the employer voucher as 
        a pure tax-free top-up.
        """
        _, tax_free_micro_gen = IrishTaxCalculator.split_micro_generation_income(profile.micro_generation_income)
        taxable_micro_gen, _ = IrishTaxCalculator.split_micro_generation_income(profile.micro_generation_income)
        taxable_rent_a_room, tax_free_rent_a_room = IrishTaxCalculator.split_rent_a_room_income(profile.rent_a_room_income)

        take_home = taxable_base - investments.pension_contribution - investments.eiis_investment - investments.deeds_of_covenant - total_taxes
        take_home += tax_free_rent_a_room + tax_free_micro_gen + taxable_micro_gen + taxable_rent_a_room
        take_home -= profile.qualifying_health_expenses
        take_home += investments.voucher_allocation
        return take_home

    @staticmethod
    def calculate_effective_rate(profile: UserProfile, investments: Investments, total_taxes: float) -> float:
        """
        Effective tax rate as a percentage of total gross inflow (salary + ancillary + voucher).
        This represents the real-world tax burden: what fraction of every euro that enters 
        your financial life was consumed by the state.
        """
        total_gross_inflow = profile.gross_income + profile.rent_a_room_income + profile.micro_generation_income + investments.voucher_allocation
        return (total_taxes / total_gross_inflow) * 100 if total_gross_inflow > 0 else 0.0

    @staticmethod
    def build_result_dict(profile: UserProfile, investments: Investments, gross_income_tax: float, total_credits: float, net_income_tax: float, usc: float, prsi: float, total_taxes: float, take_home: float, effective_rate: float, marginal_overall_rate: float) -> dict:
        """Assembles the final output dictionary. Includes _raw_take_home (unrounded) for 
        the SciPy optimizer which needs continuous gradients to converge."""
        return {
            "Core Financials": {
                "Gross Compensatory Value": profile.gross_income,
                "Rent-a-Room Income": profile.rent_a_room_income,
                "Micro-generation Income": profile.micro_generation_income,
                "Voucher Allocation": investments.voucher_allocation,
                "Cycle to Work": investments.cycle_to_work,
                "Travel Pass": investments.travel_pass,
                "Pension Deduction": investments.pension_contribution,
                "EIIS Investment": investments.eiis_investment,
                "Deeds of Covenant": investments.deeds_of_covenant,
                "Out-of-Pocket Health Expenses": profile.qualifying_health_expenses,
                "Benefits In Kind (BIK)": profile.bik,
                "Employer Health Premium (BIK)": profile.employer_health_premium,
            },
            "Tax Deductions": {
                "Gross Income Tax": round(gross_income_tax, 2),
                "Tax Credits Applied": round(total_credits, 2),
                "Net Income Tax (PAYE)": round(net_income_tax, 2),
                "USC": round(usc, 2),
                "PRSI": round(prsi, 2),
                "Rent Tax Credit (20%)": round(IrishTaxCalculator._calculate_rent_credit(profile), 2),
                "Cycle to Work": round(investments.cycle_to_work, 2),
                "Travel Pass": round(investments.travel_pass, 2),
                "EIIS Deduction": round(investments.eiis_investment, 2),
                "Deeds of Covenant Deduction": round(investments.deeds_of_covenant, 2),
                "Health Expenses Relief (20%)": round(profile.qualifying_health_expenses * 0.20, 2),
                "Health Insurance Relief (20%)": round(profile.employer_health_premium * 0.20, 2)
            },
            "Summary": {
                "Total Tax Deduced": round(total_taxes, 2),
                "Take Home CASH": round(take_home, 2),
                "_raw_take_home": take_home,
                "Effective Tax Rate (%)": round(effective_rate, 2),
                "Marginal Tax Rate (%)": round(marginal_overall_rate * 100, 2)
            }
        }

    @staticmethod
    def calculate(profile: UserProfile, investments: Investments) -> dict:
        """
        Pure orchestrator. Delegates every stage of the tax computation pipeline to 
        dedicated helper functions. Contains no math itself. Sequence:
        1. calculate_taxable_base -> gross minus salary sacrifice
        2. calculate_total_income -> add BIK and taxable ancillary streams
        3. calculate_taxable_paye_income -> subtract pension/EIIS/deeds
        4. calculate_net_income_tax -> apply brackets, credits, age exemption
        5. calculate_prsi -> PRSI with taper relief
        6. calculate_usc -> USC across 4 bands
        7. calculate_take_home -> final liquid cash
        8. build_result_dict -> format output
        """
        if profile.gross_income <= 0:
            return IrishTaxCalculator._build_empty_response()

        cfg = TAX_REGISTRY[profile.tax_year]

        taxable_base = IrishTaxCalculator.calculate_taxable_base(profile, investments)
        total_income = IrishTaxCalculator.calculate_total_income(taxable_base, profile)
        taxable_paye_income = IrishTaxCalculator.calculate_taxable_paye_income(total_income, investments)

        gross_income_tax, net_income_tax, total_credits, marginal_income_tax_rate = IrishTaxCalculator.calculate_net_income_tax(profile, taxable_paye_income, total_income, cfg)
        prsi, prsi_marginal = IrishTaxCalculator.calculate_prsi(profile, total_income, cfg)
        usc, usc_marginal = IrishTaxCalculator.calculate_usc(profile, total_income, cfg)

        total_taxes = net_income_tax + prsi + usc
        take_home = IrishTaxCalculator.calculate_take_home(profile, investments, taxable_base, total_taxes)
        marginal_overall_rate = marginal_income_tax_rate + prsi_marginal + usc_marginal
        effective_rate = IrishTaxCalculator.calculate_effective_rate(profile, investments, total_taxes)

        return IrishTaxCalculator.build_result_dict(profile, investments, gross_income_tax, total_credits, net_income_tax, usc, prsi, total_taxes, take_home, effective_rate, marginal_overall_rate)

    @staticmethod
    def _build_empty_response() -> dict:
        return {
            "Core Financials": {"Gross Compensatory Value": 0.0, "Rent-a-Room Income": 0.0, "Micro-generation Income": 0.0, "Voucher Allocation": 0.0, "Cycle to Work": 0.0, "Travel Pass": 0.0, "Pension Deduction": 0.0, "EIIS Investment": 0.0, "Deeds of Covenant": 0.0, "Out-of-Pocket Health Expenses": 0.0, "Benefits In Kind (BIK)": 0.0, "Employer Health Premium (BIK)": 0.0},
            "Tax Deductions": {"Gross Income Tax": 0.0, "Tax Credits Applied": 0.0, "Net Income Tax (PAYE)": 0.0, "USC": 0.0, "PRSI": 0.0, "Rent Tax Credit (20%)": 0.0, "Cycle to Work": 0.0, "Travel Pass": 0.0, "EIIS Deduction": 0.0, "Deeds of Covenant Deduction": 0.0, "Health Expenses Relief (20%)": 0.0, "Health Insurance Relief (20%)": 0.0},
            "Summary": {"Total Tax Deduced": 0.0, "Take Home CASH": 0.0, "_raw_take_home": 0.0, "Effective Tax Rate (%)": 0.0, "Marginal Tax Rate (%)": 0.0}
        }

    @staticmethod
    def _objective_function(x, profile: UserProfile, base_investments: Investments, utility_weight_pension: float, utility_weight_cycle: float, utility_weight_travel: float, utility_weight_eiis: float, utility_weight_deeds: float) -> float:
        """
        SciPy minimizes this function. We negate total_utility so minimization = maximization.
        Total utility = raw take-home cash + weighted value of each investment.
        A utility weight of 1.2 on pension means €1 in pension is valued at €1.20 of utility 
        (reflecting the forced savings + employer match value). A weight of 0.0 means the 
        optimizer will never allocate to that lever. The voucher is excluded because it is 
        a fixed employer-side input that does not drain the employee's liquidity.
        """
        investments = replace(base_investments, 
            pension_contribution = x[0],
            cycle_to_work = x[1],
            travel_pass = x[2],
            eiis_investment = x[3],
            deeds_of_covenant = x[4]
        )
        
        result = IrishTaxCalculator.calculate(profile, investments)
        take_home_cash = result["Summary"]["_raw_take_home"]
        
        utility_pension = utility_weight_pension * investments.pension_contribution
        utility_cycle = utility_weight_cycle * investments.cycle_to_work
        utility_travel = utility_weight_travel * investments.travel_pass
        utility_eiis = utility_weight_eiis * investments.eiis_investment
        utility_deeds = utility_weight_deeds * investments.deeds_of_covenant
        
        total_utility = take_home_cash + utility_pension + utility_cycle + utility_travel + utility_eiis + utility_deeds
        return -total_utility

    @staticmethod
    def optimize(profile: UserProfile, base_investments: Investments, required_liquid_cash: float = 0.0, utility_weight_pension=1.2, utility_weight_cycle=0.85, utility_weight_travel=0.95, utility_weight_eiis=0.8, utility_weight_deeds=1.0) -> Investments:
        """
        Multidimensional optimizer using SciPy SLSQP (Sequential Least Squares Programming).
        Finds the optimal allocation across 5 investment levers that maximizes total utility 
        (take-home cash + weighted investment value) subject to two constraints:
        1. Each lever is bounded by its legal maximum (e.g. pension by age-based NRE limit).
        2. A liquidity constraint ensures take-home cash never drops below required_liquid_cash.
        Before running, it checks if required_liquid_cash is even achievable by computing 
        max_possible_cash with zero investments. If not, it rejects immediately.
        Uses _raw_take_home (unrounded) for gradient continuity.
        """
        zero_investments = replace(base_investments,
            pension_contribution = 0.0,
            cycle_to_work = 0.0,
            travel_pass = 0.0,
            eiis_investment = 0.0,
            deeds_of_covenant = 0.0
        )
        max_possible_cash = IrishTaxCalculator.calculate(profile, zero_investments)["Summary"]["_raw_take_home"]
        
        if required_liquid_cash > max_possible_cash:
            print("="*50)
            print(f"OPTIMIZATION FAILED: INSUFFICIENT GROSS FUNDS")
            print(f"You requested: €{required_liquid_cash:,.2f} minimum liquid cash.")
            print(f"Absolute max cash possible (if taxes paid & no deductions): €{max_possible_cash:,.2f}")
            print("="*50 + "\n")
            return base_investments

        max_pension = IrishTaxCalculator.get_max_pension_limit(profile)
        max_cycle = IrishTaxCalculator.get_max_cycle_to_work_limit(base_investments)
        max_travel = 1830.0  
        max_eiis = 500000.0
        max_deeds = (profile.gross_income + profile.rent_a_room_income + profile.micro_generation_income) * 0.05
        
        bounds = [
            (0.0, max_pension),
            (0.0, max_cycle),
            (0.0, max_travel),
            (0.0, max_eiis),
            (0.0, max_deeds)
        ]
        
        def liquidity_constraint(x):
            investments = replace(base_investments,
                pension_contribution = x[0],
                cycle_to_work = x[1],
                travel_pass = x[2],
                eiis_investment = x[3],
                deeds_of_covenant = x[4]
            )
            return IrishTaxCalculator.calculate(profile, investments)["Summary"]["_raw_take_home"] - required_liquid_cash

        constraints = ({'type': 'ineq', 'fun': liquidity_constraint})
        
        x0 = [10.0, 10.0, 10.0, 10.0, 10.0]
        
        res = minimize(
            lambda x: IrishTaxCalculator._objective_function(x, profile, base_investments, utility_weight_pension, utility_weight_cycle, utility_weight_travel, utility_weight_eiis, utility_weight_deeds), 
            x0,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        if res.success:
            optimal_investments = replace(base_investments,
                pension_contribution = res.x[0],
                cycle_to_work = res.x[1],
                travel_pass = res.x[2],
                eiis_investment = res.x[3],
                deeds_of_covenant = res.x[4]
            )
            final_result = IrishTaxCalculator.calculate(profile, optimal_investments)
            
            print(json.dumps(final_result, indent=4))
            print("\n" + "="*50)
            print("MULTIDIMENSIONAL UTILITY OPTIMIZATION RESULT:")
            if required_liquid_cash > 0:
                print(f"[CONSTRAINT ACTIVE] Minimum Liquidity Enforced: €{required_liquid_cash:,.2f}")
            print(f"Optimal Pension Allocation: €{round(optimal_investments.pension_contribution, 2)} (Bound: €{round(max_pension,2)})")
            print(f"Optimal Cycle to Work Allocation: €{round(optimal_investments.cycle_to_work, 2)} (Bound: €{round(max_cycle,2)})")
            print(f"Optimal Travel Pass Allocation: €{round(optimal_investments.travel_pass, 2)} (Bound: €{round(max_travel,2)})")
            print(f"Optimal EIIS Investment: €{round(optimal_investments.eiis_investment, 2)} (Bound: €{round(max_eiis,2)})")
            print(f"Optimal Deeds of Covenant: €{round(optimal_investments.deeds_of_covenant, 2)} (Bound: €{round(max_deeds,2)})")
            print(f"Manually Input Voucher Allocation: €{round(optimal_investments.voucher_allocation, 2)} (Bypassed Optimizer)")
            print(f"- Pension utility metric: {utility_weight_pension}")
            print(f"- Cycle utility metric: {utility_weight_cycle}")
            print(f"- Travel pass utility metric: {utility_weight_travel}")
            print(f"- EIIS utility metric: {utility_weight_eiis}")
            print(f"- Deeds utility metric: {utility_weight_deeds}")
            print("="*50 + "\n")
            return optimal_investments
        else:
            print("Bounded multidimensional optimization failed.", res.message)
            return base_investments

    @staticmethod
    def marginal_rate_curve(base_profile: UserProfile, base_investments: Investments, max_income: float = 200_000, step: float = 500) -> list[dict]:
        """
        Generates a discrete marginal tax rate curve by sweeping gross income from 0 to 
        max_income. At each point, it calculates the tax on income X and income X+1, then 
        derives the marginal rate as the difference. All deductions are zeroed out to 
        isolate the pure tax system behavior. Flags any point where the marginal rate 
        exceeds 80% as a USC exemption kink (caused by the cliff at €13,000 where USC 
        jumps from 0% to the full banded rate on the entire income).
        """
        curve = []
        
        investments_zeroed = replace(base_investments,
            pension_contribution = 0.0,
            voucher_allocation = 0.0,
            cycle_to_work = 0.0,
            travel_pass = 0.0
        )
        
        income_pts = [float(x) for x in range(int(step), int(max_income) + int(step), int(step))]
        
        for inc in income_pts:
            profile_current = replace(base_profile, gross_income=inc)
            res_base = IrishTaxCalculator.calculate(profile_current, investments_zeroed)
            tax_at_x = res_base["Summary"]["Total Tax Deduced"]
            eff_rate = res_base["Summary"]["Effective Tax Rate (%)"]
            
            profile_plus = replace(base_profile, gross_income=inc + 1.0)
            res_plus = IrishTaxCalculator.calculate(profile_plus, investments_zeroed)
            tax_at_x_plus_1 = res_plus["Summary"]["Total Tax Deduced"]
            
            marginal_m = (tax_at_x_plus_1 - tax_at_x) / 1.0 * 100.0
            
            row = {
                "gross_income": inc,
                "marginal_rate_pct": round(marginal_m, 2),
                "effective_rate_pct": round(eff_rate, 2)
            }
            if marginal_m > 80.0:
                row["usc_kink"] = True
                
            curve.append(row)
            
        return curve

    @staticmethod
    def print_marginal_curve(base_profile: UserProfile, base_investments: Investments, max_income: float = 120_000, step: float = 1_000):
        """Pretty-prints the marginal rate curve as a formatted table."""
        curve = IrishTaxCalculator.marginal_rate_curve(base_profile, base_investments, max_income, step)
        print(f"{'Gross Income':<15} | {'Marginal Rate %':<16} | {'Effective Rate %':<16} | {'Notes'}")
        print("-" * 75)
        for row in curve:
            notes = "USC Exemption Kink (Jump > 80%)" if row.get("usc_kink") else ""
            print(f"€{row['gross_income']:<14.2f} | {row['marginal_rate_pct']:<16.2f} | {row['effective_rate_pct']:<16.2f} | {notes}")


if __name__ == "__main__":
    
    my_profile = UserProfile(
        gross_income=49000.0,
        age=24,
        employment_type="PAYE",
        marital_status="Single",
        medical_card=False,
        annual_rent_paid=500.0,
        qualifying_health_expenses=600.0,
        bik=0.0,
        employer_health_premium=1200.0
    )
    
    my_investments = Investments(
        pension_contribution=0.0,
        voucher_allocation=0.0,
        cycle_to_work=0.0,
        travel_pass=0.0,
        cycle_type="ebike",
        cycle_to_work_mode="annual"
    )

    IrishTaxCalculator.optimize(
        my_profile, 
        my_investments, 
        required_liquid_cash=38000.0,
        utility_weight_pension=1.2, 
        utility_weight_cycle=0.85, 
        utility_weight_travel=0.0
    )
