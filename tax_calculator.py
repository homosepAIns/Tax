import json
from scipy.optimize import minimize

class IrishTaxCalculator:
    """
    Computes Irish absolute net take-home pay, applying Income Tax (PAYE), 
    Universal Social Charge (USC), and Pay Related Social Insurance (PRSI).
    Configurable for 2025/2026 budget estimates.
    """

    def __init__(
        self,
        gross_income: float,
        pension_contribution: float = 0.0,
        voucher_allocation: float = 0.0,
        cycle_to_work: float = 0.0,
        travel_pass: float = 0.0,
        cycle_type: str = "regular",
        cycle_to_work_mode: str = "annual",
        bik: float = 0.0,
        employer_health_premium: float = 0.0,
        employment_type: str = "PAYE",
        marital_status: str = "Single",
        age: int = 30,
        medical_card: bool = False,
        additional_tax_credits: float = 0.0,
        tax_year: int = 2026,
        second_income: float = 0.0,
        annual_rent_paid: float = 0.0,
        is_blind: bool = False,
        has_incapacitated_child: bool = False,
        claims_home_carer: bool = False,
        claims_single_child_carer: bool = False,
        claims_dependent_relative: bool = False,
        widowed_years_since: int = -1,
        rent_a_room_income: float = 0.0,
        micro_generation_income: float = 0.0,
        qualifying_health_expenses: float = 0.0,
        eiis_investment: float = 0.0,
        deeds_of_covenant: float = 0.0
    ):
        """
        :param tax_year: Year for tax rules (2025 or 2026).
        :param second_income: Spouse's income, required for accurate 'Married_2_Incomes' cut-off.
        :param rent_tax_credit: Current limit is usually €1,000 for a single individual.
        """
        self.gross_income = gross_income
        self.pension_contribution = pension_contribution
        self.voucher_allocation = voucher_allocation
        self.cycle_to_work = cycle_to_work
        self.travel_pass = travel_pass
        self.cycle_type = cycle_type
        self.cycle_to_work_mode = cycle_to_work_mode
        self.bik = bik
        self.employer_health_premium = employer_health_premium
        self.employment_type = employment_type
        self.marital_status = marital_status
        self.age = age
        self.medical_card = medical_card
        self.additional_tax_credits = additional_tax_credits
        self.tax_year = tax_year
        self.second_income = second_income
        self.annual_rent_paid = annual_rent_paid
        self.is_blind = is_blind
        self.has_incapacitated_child = has_incapacitated_child
        self.claims_home_carer = claims_home_carer
        self.claims_single_child_carer = claims_single_child_carer
        self.claims_dependent_relative = claims_dependent_relative
        self.widowed_years_since = widowed_years_since
        self.rent_a_room_income = rent_a_room_income
        self.micro_generation_income = micro_generation_income
        self.qualifying_health_expenses = qualifying_health_expenses
        self.eiis_investment = eiis_investment
        self.deeds_of_covenant = deeds_of_covenant

    def _get_tax_config(self) -> dict:
        """Returns tax configurations dynamically based on the active tax_year."""
        cfg = {
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
        }
        
        if self.tax_year == 2025:
            cfg["USC_BAND_2_LIMIT"] = 27382.0
            cfg["PRSI_RATE"] = 0.041
        else: 
            cfg["USC_BAND_2_LIMIT"] = 28700.0
            cfg["PRSI_RATE"] = 0.0435
            
        return cfg

    def get_srcop(self, cfg: dict) -> float:
        if self.marital_status == "Married_1_Income":
            return cfg["SRCOP_MARRIED_BASE"]
        elif self.marital_status == "Married_2_Incomes":
            uplift = min(cfg["SRCOP_UPLIFT_MAX"], self.second_income)
            return cfg["SRCOP_MARRIED_BASE"] + uplift
        return cfg["SRCOP_SINGLE"]

    def calculate_usc(self, total_income: float, cfg: dict) -> tuple[float, float]:
        if total_income <= cfg["USC_EXEMPT_THRESHOLD"]:
            return 0.0, 0.0

        if self.medical_card and total_income <= 60000.0:
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
            
            if self.employment_type == "Self-Employed" and total_income > 100000.0:
                usc_tax += (total_income - 100000.0) * 0.03
                marginal = 0.11

        return usc_tax, marginal

    def calculate_prsi(self, total_income: float, cfg: dict) -> tuple[float, float]:
        if self.age >= cfg["PRSI_EXEMPT_AGE"]:
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

    def _calculate_rent_credit(self) -> float:
        rent_credit_cap = 2000.0 if self.marital_status in ["Married_1_Income", "Married_2_Incomes"] else 1000.0
        return min(rent_credit_cap, self.annual_rent_paid * 0.20)

    def get_tax_credits(self, cfg: dict) -> float:
        credits = cfg["PERSONAL_CREDIT"]
        
        if self.employment_type == "PAYE":
            credits += cfg["EMPLOYMENT_CREDIT"]
        elif self.employment_type == "Self-Employed":
            credits += cfg["EARNED_INCOME_CREDIT"]
            
        if self.marital_status in ["Married_1_Income", "Married_2_Incomes"]:
            credits += cfg["PERSONAL_CREDIT"]  # Married gets €4000 personal credit
            if self.age >= 65: credits += cfg["AGE_CREDIT_MARRIED"]
            if self.is_blind: credits += cfg["BLIND_CREDIT_MARRIED"]
        else:
            if self.age >= 65: credits += cfg["AGE_CREDIT_SINGLE"]
            if self.is_blind: credits += cfg["BLIND_CREDIT_SINGLE"]

        if self.has_incapacitated_child: credits += cfg["INCAPACITATED_CHILD_CREDIT"]
        if self.claims_home_carer: credits += cfg["HOME_CARER_CREDIT"]
        if self.claims_single_child_carer: credits += cfg["SINGLE_CHILD_CARER_CREDIT"]
        if self.claims_dependent_relative: credits += cfg["DEPENDENT_RELATIVE_CREDIT"]
        
        if 0 <= self.widowed_years_since <= 5:
            credits += max(0, 3600.0 - (self.widowed_years_since * 360.0)) # Rough 5-year taper
            
        credits += self.employer_health_premium * 0.20
        credits += self.qualifying_health_expenses * 0.20
        
        return credits + self.additional_tax_credits + self._calculate_rent_credit()

    def get_max_pension_limit(self) -> float:
        total_remuneration = self.gross_income + self.bik + self.employer_health_premium
        limit_salary = min(total_remuneration, 115000.0)
        
        if self.age < 30: pct = 0.15
        elif self.age < 40: pct = 0.20
        elif self.age < 50: pct = 0.25
        elif self.age < 55: pct = 0.30
        elif self.age < 60: pct = 0.35
        else: pct = 0.40
            
        return limit_salary * pct

    def get_max_cycle_to_work_limit(self) -> float:
        cap = 3000.0 if self.cycle_type == "ebike" else 1500.0
        if self.cycle_to_work_mode == "annual":
            return cap / 4.0
        return cap

    def calculate(self) -> dict:
        """Core math engine for calculating the final take home layout."""
        if self.gross_income <= 0:
            return self._build_empty_response()
            
        cfg = self._get_tax_config()
        
        # Shield variables cleanly off the top (Voucher is an employer top-up, not a salary deduction)
        taxable_base = max(0, self.gross_income - self.cycle_to_work - self.travel_pass)
        
        # Micro-generation relief
        taxable_micro_gen = max(0, self.micro_generation_income - 400.0)
        tax_free_micro_gen = min(self.micro_generation_income, 400.0)
        
        # Rent-a-room cliff-edge logic
        if self.rent_a_room_income > 14000.0:
            taxable_rent_a_room = self.rent_a_room_income
            tax_free_rent_a_room = 0.0
        else:
            taxable_rent_a_room = 0.0
            tax_free_rent_a_room = self.rent_a_room_income
            
        total_bik = self.bik + self.employer_health_premium
        total_income_for_prsi_usc = taxable_base + total_bik + taxable_micro_gen + taxable_rent_a_room
        
        # Deeds of covenant and EIIS reduce PAYE income natively
        taxable_paye_income = max(0, total_income_for_prsi_usc - self.pension_contribution - self.eiis_investment - self.deeds_of_covenant)

        srcop = self.get_srcop(cfg)
        tax_20_bracket = min(taxable_paye_income, srcop) * cfg["INCOME_TAX_STD_RATE"]
        tax_40_bracket = max(0, taxable_paye_income - srcop) * cfg["INCOME_TAX_HIGH_RATE"]
        gross_income_tax = tax_20_bracket + tax_40_bracket
        
        marginal_income_tax_rate = cfg["INCOME_TAX_HIGH_RATE"] if taxable_paye_income > srcop else cfg["INCOME_TAX_STD_RATE"]

        total_credits = self.get_tax_credits(cfg)
        net_income_tax = max(0, gross_income_tax - total_credits)

        # Age Exemption Limit
        if self.age >= 65:
            exemption_limit = 36000.0 if self.marital_status in ["Married_1_Income", "Married_2_Incomes"] else 18000.0
            if total_income_for_prsi_usc <= exemption_limit:
                net_income_tax = 0.0
            else:
                # Marginal relief (cap tax at 40% of the difference over limit)
                marginal_tax_cap = (total_income_for_prsi_usc - exemption_limit) * 0.40
                net_income_tax = min(net_income_tax, marginal_tax_cap)

        prsi, prsi_marginal = self.calculate_prsi(total_income_for_prsi_usc, cfg)
        usc, usc_marginal = self.calculate_usc(total_income_for_prsi_usc, cfg)

        total_taxes = net_income_tax + prsi + usc
        
        # Cash flow deductions (investments) plus bonus employer inputs
        take_home = taxable_base - self.pension_contribution - self.eiis_investment - self.deeds_of_covenant - total_taxes 
        take_home += tax_free_rent_a_room + tax_free_micro_gen + taxable_micro_gen + taxable_rent_a_room - self.qualifying_health_expenses
        take_home += self.voucher_allocation
        
        marginal_overall_rate = marginal_income_tax_rate + prsi_marginal + usc_marginal
        
        # Effective rate against total gross inflow
        total_gross_inflow = self.gross_income + self.rent_a_room_income + self.micro_generation_income + self.voucher_allocation
        effective_rate = (total_taxes / total_gross_inflow) * 100 if total_gross_inflow > 0 else 0.0

        return {
            "Core Financials": {
                "Gross Compensatory Value": self.gross_income,
                "Rent-a-Room Income": self.rent_a_room_income,
                "Micro-generation Income": self.micro_generation_income,
                "Voucher Allocation": self.voucher_allocation,
                "Cycle to Work": self.cycle_to_work,
                "Travel Pass": self.travel_pass,
                "Pension Deduction": self.pension_contribution,
                "EIIS Investment": self.eiis_investment,
                "Deeds of Covenant": self.deeds_of_covenant,
                "Out-of-Pocket Health Expenses": self.qualifying_health_expenses,
                "Benefits In Kind (BIK)": self.bik,
                "Employer Health Premium (BIK)": self.employer_health_premium,
            },
            "Tax Deductions": {
                "Gross Income Tax": round(gross_income_tax, 2),
                "Tax Credits Applied": round(total_credits, 2),
                "Net Income Tax (PAYE)": round(net_income_tax, 2),
                "USC": round(usc, 2),
                "PRSI": round(prsi, 2),
                "Rent Tax Credit (20%)": round(self._calculate_rent_credit(), 2),
                "Cycle to Work": round(self.cycle_to_work, 2),
                "Travel Pass": round(self.travel_pass, 2),
                "EIIS Deduction": round(self.eiis_investment, 2),
                "Deeds of Covenant Deduction": round(self.deeds_of_covenant, 2),
                "Health Expenses Relief (20%)": round(self.qualifying_health_expenses * 0.20, 2),
                "Health Insurance Relief (20%)": round(self.employer_health_premium * 0.20, 2)
            },
            "Summary": {
                "Total Tax Deduced": round(total_taxes, 2),
                "Take Home CASH": round(take_home, 2),
                "_raw_take_home": take_home,
                "Effective Tax Rate (%)": round(effective_rate, 2),
                "Marginal Tax Rate (%)": round(marginal_overall_rate * 100, 2)
            }
        }

    def _build_empty_response(self) -> dict:
        return {
            "Core Financials": {"Gross Compensatory Value": 0.0, "Rent-a-Room Income": 0.0, "Micro-generation Income": 0.0, "Voucher Allocation": 0.0, "Cycle to Work": 0.0, "Travel Pass": 0.0, "Pension Deduction": 0.0, "EIIS Investment": 0.0, "Deeds of Covenant": 0.0, "Out-of-Pocket Health Expenses": 0.0, "Benefits In Kind (BIK)": 0.0, "Employer Health Premium (BIK)": 0.0},
            "Tax Deductions": {"Gross Income Tax": 0.0, "Tax Credits Applied": 0.0, "Net Income Tax (PAYE)": 0.0, "USC": 0.0, "PRSI": 0.0, "Rent Tax Credit (20%)": 0.0, "Cycle to Work": 0.0, "Travel Pass": 0.0, "EIIS Deduction": 0.0, "Deeds of Covenant Deduction": 0.0, "Health Expenses Relief (20%)": 0.0, "Health Insurance Relief (20%)": 0.0},
            "Summary": {"Total Tax Deduced": 0.0, "Take Home CASH": 0.0, "_raw_take_home": 0.0, "Effective Tax Rate (%)": 0.0, "Marginal Tax Rate (%)": 0.0}
        }

    def _objective_function(self, x, utility_weight_pension: float, utility_weight_cycle: float, utility_weight_travel: float, utility_weight_eiis: float, utility_weight_deeds: float) -> float:
        self.pension_contribution = x[0]
        self.cycle_to_work = x[1]
        self.travel_pass = x[2]
        self.eiis_investment = x[3]
        self.deeds_of_covenant = x[4]
        
        result = self.calculate()
        take_home_cash = result["Summary"]["_raw_take_home"]
        
        utility_pension = utility_weight_pension * self.pension_contribution
        utility_cycle = utility_weight_cycle * self.cycle_to_work
        utility_travel = utility_weight_travel * self.travel_pass
        utility_eiis = utility_weight_eiis * self.eiis_investment
        utility_deeds = utility_weight_deeds * self.deeds_of_covenant
        
        # Voucher is omitted as it is purely additive and doesn't drain liquidity
        total_utility = take_home_cash + utility_pension + utility_cycle + utility_travel + utility_eiis + utility_deeds
        return -total_utility

    def optimize(self, utility_weight_pension=1.2, utility_weight_cycle=0.85, utility_weight_travel=0.95, utility_weight_eiis=0.8, utility_weight_deeds=1.0):
        max_pension = self.get_max_pension_limit()
        max_cycle = self.get_max_cycle_to_work_limit()
        max_travel = 1830.0  
        max_eiis = 500000.0
        max_deeds = (self.gross_income + self.rent_a_room_income + self.micro_generation_income) * 0.05
        
        bounds = [
            (0.0, max_pension),    # x[0]
            (0.0, max_cycle),      # x[1]
            (0.0, max_travel),     # x[2]
            (0.0, max_eiis),       # x[3]
            (0.0, max_deeds)       # x[4]
        ]
        
        def liquidity_constraint(x):
            self.pension_contribution = x[0]
            self.cycle_to_work = x[1]
            self.travel_pass = x[2]
            self.eiis_investment = x[3]
            self.deeds_of_covenant = x[4]
            return self.calculate()["Summary"]["_raw_take_home"]

        constraints = ({'type': 'ineq', 'fun': liquidity_constraint})
        
        x0 = [10.0, 10.0, 10.0, 10.0, 10.0]
        
        res = minimize(
            lambda x: self._objective_function(x, utility_weight_pension, utility_weight_cycle, utility_weight_travel, utility_weight_eiis, utility_weight_deeds), 
            x0,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        if res.success:
            self.pension_contribution = res.x[0]
            self.cycle_to_work = res.x[1]
            self.travel_pass = res.x[2]
            self.eiis_investment = res.x[3]
            self.deeds_of_covenant = res.x[4]
            
            final_result = self.calculate()
            
            print(json.dumps(final_result, indent=4))
            print("\n" + "="*50)
            print("MULTIDIMENSIONAL UTILITY OPTIMIZATION RESULT:")
            print(f"Optimal Pension Allocation: €{round(self.pension_contribution, 2)} (Bound: €{round(max_pension,2)})")
            print(f"Optimal Cycle to Work Allocation: €{round(self.cycle_to_work, 2)} (Bound: €{round(max_cycle,2)})")
            print(f"Optimal Travel Pass Allocation: €{round(self.travel_pass, 2)} (Bound: €{round(max_travel,2)})")
            print(f"Optimal EIIS Investment: €{round(self.eiis_investment, 2)} (Bound: €{round(max_eiis,2)})")
            print(f"Optimal Deeds of Covenant: €{round(self.deeds_of_covenant, 2)} (Bound: €{round(max_deeds,2)})")
            print(f"Manually Input Voucher Allocation: €{round(self.voucher_allocation, 2)} (Bypassed Optimizer)")
            print(f"- Pension utility metric: {utility_weight_pension}")
            print(f"- Cycle utility metric: {utility_weight_cycle}")
            print(f"- Travel pass utility metric: {utility_weight_travel}")
            print(f"- EIIS utility metric: {utility_weight_eiis}")
            print(f"- Deeds utility metric: {utility_weight_deeds}")
            print("="*50 + "\n")
        else:
            print("Bounded multidimensional optimization failed.", res.message)

    def marginal_rate_curve(self, max_income: float = 200_000, step: float = 500) -> list[dict]:
        curve = []
        original_gross = self.gross_income
        original_pension = self.pension_contribution
        original_voucher = self.voucher_allocation
        original_cycle = self.cycle_to_work
        original_travel = self.travel_pass
        
        self.pension_contribution = 0.0
        self.voucher_allocation = 0.0
        self.cycle_to_work = 0.0
        self.travel_pass = 0.0
        
        income_pts = [float(x) for x in range(int(step), int(max_income) + int(step), int(step))]
        
        for inc in income_pts:
            self.gross_income = inc
            res_base = self.calculate()
            tax_at_x = res_base["Summary"]["Total Tax Deduced"]
            eff_rate = res_base["Summary"]["Effective Tax Rate (%)"]
            
            self.gross_income = inc + 1.0
            res_plus = self.calculate()
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
            
        self.gross_income = original_gross
        self.pension_contribution = original_pension
        self.voucher_allocation = original_voucher
        self.cycle_to_work = original_cycle
        self.travel_pass = original_travel
        return curve

    def print_marginal_curve(self, max_income: float = 120_000, step: float = 1_000):
        curve = self.marginal_rate_curve(max_income, step)
        print(f"{'Gross Income':<15} | {'Marginal Rate %':<16} | {'Effective Rate %':<16} | {'Notes'}")
        print("-" * 75)
        for row in curve:
            notes = "USC Exemption Kink (Jump > 80%)" if row.get("usc_kink") else ""
            print(f"€{row['gross_income']:<14.2f} | {row['marginal_rate_pct']:<16.2f} | {row['effective_rate_pct']:<16.2f} | {notes}")


if __name__ == "__main__":
    
    calc = IrishTaxCalculator(
        gross_income=49000.0,
        age=24,
        pension_contribution=0.0,
        voucher_allocation=0.0,
        cycle_to_work=0.0,
        travel_pass=0.0,
        cycle_type="ebike",
        cycle_to_work_mode="annual",
        bik=0.0,
        employer_health_premium=1200.0,
        employment_type="PAYE",
        marital_status="Single",
        medical_card=False,
        annual_rent_paid=500.0,
        qualifying_health_expenses=600.0
    )

    # Calling optimization to ensure bounds behavior applies cleanly on fixed BIK
    calc.optimize(utility_weight_pension=1.2, utility_weight_cycle=0.85, utility_weight_travel=0.0)
