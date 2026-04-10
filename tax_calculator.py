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
    """
    Stateless Engine. Computes Irish absolute net take-home pay via structurally pure 
    functions enforcing the 2025/2026 tax standards framework.
    """

    @staticmethod
    def get_srcop(profile: UserProfile, cfg: dict) -> float:
        if profile.marital_status == "Married_1_Income":
            return cfg["SRCOP_MARRIED_BASE"]
        elif profile.marital_status == "Married_2_Incomes":
            uplift = min(cfg["SRCOP_UPLIFT_MAX"], profile.second_income)
            return cfg["SRCOP_MARRIED_BASE"] + uplift
        return cfg["SRCOP_SINGLE"]

    @staticmethod
    def calculate_usc(profile: UserProfile, total_income: float, cfg: dict) -> tuple[float, float]:
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
        if profile.age >= cfg["PRSI_EXEMPT_AGE"]:
            return 0.0, 0.0
        if total_income <= cfg["PRSI_THRESHOLD"]:
            return 0.0, 0.0
            
        rate = cfg["PRSI_RATE"]
        gross_prsi = total_income * rate
        
        # PRSI Step Effect (Taper Relief) on lower incomes
        annual_credit = max(0.0, 624.0 - ((total_income - cfg["PRSI_THRESHOLD"]) / 6.0))
        net_prsi = max(0.0, gross_prsi - annual_credit)
        
        prsi_marginal = rate
        # If inside the taper zone, the marginal rate is actually higher because the credit drops
        if annual_credit > 0.0:
            prsi_marginal = rate + (1.0 / 6.0)
            
        return net_prsi, prsi_marginal

    @staticmethod
    def _calculate_rent_credit(profile: UserProfile) -> float:
        rent_credit_cap = 2000.0 if profile.marital_status in ["Married_1_Income", "Married_2_Incomes"] else 1000.0
        return min(rent_credit_cap, profile.annual_rent_paid * 0.20)

    @staticmethod
    def get_tax_credits(profile: UserProfile, cfg: dict) -> float:
        credits = cfg["PERSONAL_CREDIT"]
        
        if profile.employment_type == "PAYE":
            credits += cfg["EMPLOYMENT_CREDIT"]
        elif profile.employment_type == "Self-Employed":
            credits += cfg["EARNED_INCOME_CREDIT"]
            
        if profile.marital_status in ["Married_1_Income", "Married_2_Incomes"]:
            credits += cfg["PERSONAL_CREDIT"]  # Married gets €4000 personal credit
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
            credits += max(0, 3600.0 - (profile.widowed_years_since * 360.0)) # Rough 5-year taper
            
        credits += profile.employer_health_premium * 0.20
        credits += profile.qualifying_health_expenses * 0.20
        
        return credits + profile.additional_tax_credits + IrishTaxCalculator._calculate_rent_credit(profile)

    @staticmethod
    def get_max_pension_limit(profile: UserProfile) -> float:
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
        cap = 3000.0 if investments.cycle_type == "ebike" else 1500.0
        if investments.cycle_to_work_mode == "annual":
            return cap / 4.0
        return cap

    @staticmethod
    def calculate(profile: UserProfile, investments: Investments) -> dict:
        """Pure function engine for calculating the final take home layout."""
        if profile.gross_income <= 0:
            return IrishTaxCalculator._build_empty_response()
            
        cfg = TAX_REGISTRY[profile.tax_year]
        
        # Shield variables cleanly off the top (Voucher is an employer top-up, not a salary deduction)
        taxable_base = max(0, profile.gross_income - investments.cycle_to_work - investments.travel_pass)
        
        # Micro-generation relief
        taxable_micro_gen = max(0, profile.micro_generation_income - 400.0)
        tax_free_micro_gen = min(profile.micro_generation_income, 400.0)
        
        # Rent-a-room cliff-edge logic
        if profile.rent_a_room_income > 14000.0:
            taxable_rent_a_room = profile.rent_a_room_income
            tax_free_rent_a_room = 0.0
        else:
            taxable_rent_a_room = 0.0
            tax_free_rent_a_room = profile.rent_a_room_income
            
        total_bik = profile.bik + profile.employer_health_premium
        total_income_for_prsi_usc = taxable_base + total_bik + taxable_micro_gen + taxable_rent_a_room
        
        # Deeds of covenant and EIIS reduce PAYE income natively
        taxable_paye_income = max(0, total_income_for_prsi_usc - investments.pension_contribution - investments.eiis_investment - investments.deeds_of_covenant)

        srcop = IrishTaxCalculator.get_srcop(profile, cfg)
        tax_20_bracket = min(taxable_paye_income, srcop) * cfg["INCOME_TAX_STD_RATE"]
        tax_40_bracket = max(0, taxable_paye_income - srcop) * cfg["INCOME_TAX_HIGH_RATE"]
        gross_income_tax = tax_20_bracket + tax_40_bracket
        
        marginal_income_tax_rate = cfg["INCOME_TAX_HIGH_RATE"] if taxable_paye_income > srcop else cfg["INCOME_TAX_STD_RATE"]

        total_credits = IrishTaxCalculator.get_tax_credits(profile, cfg)
        net_income_tax = max(0, gross_income_tax - total_credits)

        # Age Exemption Limit
        if profile.age >= 65:
            exemption_limit = 36000.0 if profile.marital_status in ["Married_1_Income", "Married_2_Incomes"] else 18000.0
            if total_income_for_prsi_usc <= exemption_limit:
                net_income_tax = 0.0
            else:
                # Marginal relief (cap tax at 40% of the difference over limit)
                marginal_tax_cap = (total_income_for_prsi_usc - exemption_limit) * 0.40
                net_income_tax = min(net_income_tax, marginal_tax_cap)

        prsi, prsi_marginal = IrishTaxCalculator.calculate_prsi(profile, total_income_for_prsi_usc, cfg)
        usc, usc_marginal = IrishTaxCalculator.calculate_usc(profile, total_income_for_prsi_usc, cfg)

        total_taxes = net_income_tax + prsi + usc
        
        # Cash flow deductions (investments) plus bonus employer inputs
        take_home = taxable_base - investments.pension_contribution - investments.eiis_investment - investments.deeds_of_covenant - total_taxes 
        take_home += tax_free_rent_a_room + tax_free_micro_gen + taxable_micro_gen + taxable_rent_a_room - profile.qualifying_health_expenses
        take_home += investments.voucher_allocation
        
        marginal_overall_rate = marginal_income_tax_rate + prsi_marginal + usc_marginal
        
        # Effective rate against total gross inflow
        total_gross_inflow = profile.gross_income + profile.rent_a_room_income + profile.micro_generation_income + investments.voucher_allocation
        effective_rate = (total_taxes / total_gross_inflow) * 100 if total_gross_inflow > 0 else 0.0

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
    def _build_empty_response() -> dict:
        return {
            "Core Financials": {"Gross Compensatory Value": 0.0, "Rent-a-Room Income": 0.0, "Micro-generation Income": 0.0, "Voucher Allocation": 0.0, "Cycle to Work": 0.0, "Travel Pass": 0.0, "Pension Deduction": 0.0, "EIIS Investment": 0.0, "Deeds of Covenant": 0.0, "Out-of-Pocket Health Expenses": 0.0, "Benefits In Kind (BIK)": 0.0, "Employer Health Premium (BIK)": 0.0},
            "Tax Deductions": {"Gross Income Tax": 0.0, "Tax Credits Applied": 0.0, "Net Income Tax (PAYE)": 0.0, "USC": 0.0, "PRSI": 0.0, "Rent Tax Credit (20%)": 0.0, "Cycle to Work": 0.0, "Travel Pass": 0.0, "EIIS Deduction": 0.0, "Deeds of Covenant Deduction": 0.0, "Health Expenses Relief (20%)": 0.0, "Health Insurance Relief (20%)": 0.0},
            "Summary": {"Total Tax Deduced": 0.0, "Take Home CASH": 0.0, "_raw_take_home": 0.0, "Effective Tax Rate (%)": 0.0, "Marginal Tax Rate (%)": 0.0}
        }

    @staticmethod
    def _objective_function(x, profile: UserProfile, base_investments: Investments, utility_weight_pension: float, utility_weight_cycle: float, utility_weight_travel: float, utility_weight_eiis: float, utility_weight_deeds: float) -> float:
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
        
        # Voucher is omitted as it is purely additive and doesn't drain liquidity
        total_utility = take_home_cash + utility_pension + utility_cycle + utility_travel + utility_eiis + utility_deeds
        return -total_utility

    @staticmethod
    def optimize(profile: UserProfile, base_investments: Investments, utility_weight_pension=1.2, utility_weight_cycle=0.85, utility_weight_travel=0.95, utility_weight_eiis=0.8, utility_weight_deeds=1.0) -> Investments:
        max_pension = IrishTaxCalculator.get_max_pension_limit(profile)
        max_cycle = IrishTaxCalculator.get_max_cycle_to_work_limit(base_investments)
        max_travel = 1830.0  
        max_eiis = 500000.0
        max_deeds = (profile.gross_income + profile.rent_a_room_income + profile.micro_generation_income) * 0.05
        
        bounds = [
            (0.0, max_pension),    # x[0]
            (0.0, max_cycle),      # x[1]
            (0.0, max_travel),     # x[2]
            (0.0, max_eiis),       # x[3]
            (0.0, max_deeds)       # x[4]
        ]
        
        def liquidity_constraint(x):
            investments = replace(base_investments,
                pension_contribution = x[0],
                cycle_to_work = x[1],
                travel_pass = x[2],
                eiis_investment = x[3],
                deeds_of_covenant = x[4]
            )
            return IrishTaxCalculator.calculate(profile, investments)["Summary"]["_raw_take_home"]

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
        curve = []
        
        # Zero out deductions for a pure marginal curve
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
        curve = IrishTaxCalculator.marginal_rate_curve(base_profile, base_investments, max_income, step)
        print(f"{'Gross Income':<15} | {'Marginal Rate %':<16} | {'Effective Rate %':<16} | {'Notes'}")
        print("-" * 75)
        for row in curve:
            notes = "USC Exemption Kink (Jump > 80%)" if row.get("usc_kink") else ""
            print(f"€{row['gross_income']:<14.2f} | {row['marginal_rate_pct']:<16.2f} | {row['effective_rate_pct']:<16.2f} | {notes}")


if __name__ == "__main__":
    
    # Instantiate Data Classes cleanly instead of 28 ad-hoc parameters
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

    # Scipy execution relies on pure functions, protecting primary state
    IrishTaxCalculator.optimize(
        my_profile, 
        my_investments, 
        utility_weight_pension=1.2, 
        utility_weight_cycle=0.85, 
        utility_weight_travel=0.0
    )
