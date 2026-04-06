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
        cycle_type: str = "regular",
        cycle_to_work_mode: str = "annual",
        bik: float = 0.0,
        employment_type: str = "PAYE",
        marital_status: str = "Single",
        age: int = 30,
        medical_card: bool = False,
        additional_tax_credits: float = 0.0,
        tax_year: int = 2026,
        second_income: float = 0.0,
        rent_tax_credit: float = 0.0
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
        self.cycle_type = cycle_type
        self.cycle_to_work_mode = cycle_to_work_mode
        self.bik = bik
        self.employment_type = employment_type
        self.marital_status = marital_status
        self.age = age
        self.medical_card = medical_card
        self.additional_tax_credits = additional_tax_credits
        self.tax_year = tax_year
        self.second_income = second_income
        self.rent_tax_credit = rent_tax_credit

    def _get_tax_config(self) -> dict:
        """Returns tax configurations dynamically based on the active tax_year."""
        cfg = {
            "PERSONAL_CREDIT": 2000.0,
            "EMPLOYMENT_CREDIT": 2000.0, 
            "AGE_CREDIT_SINGLE": 245.0,
            
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
            cfg["PRSI_RATE"] = 0.042
            
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
        return total_income * rate, rate

    def get_tax_credits(self, cfg: dict) -> float:
        credits = cfg["PERSONAL_CREDIT"]
        credits += cfg["EMPLOYMENT_CREDIT"]
        
        if self.age >= 65:
            credits += cfg["AGE_CREDIT_SINGLE"]
            
        return credits + self.additional_tax_credits + self.rent_tax_credit

    def get_max_pension_limit(self) -> float:
        limit_salary = min(self.gross_income, 115000.0)
        
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
        
        taxable_base = max(0, self.gross_income - self.voucher_allocation - self.cycle_to_work)
        
        total_income_for_prsi_usc = taxable_base + self.bik
        taxable_paye_income = max(0, total_income_for_prsi_usc - self.pension_contribution)

        srcop = self.get_srcop(cfg)
        tax_20_bracket = min(taxable_paye_income, srcop) * cfg["INCOME_TAX_STD_RATE"]
        tax_40_bracket = max(0, taxable_paye_income - srcop) * cfg["INCOME_TAX_HIGH_RATE"]
        gross_income_tax = tax_20_bracket + tax_40_bracket
        
        marginal_income_tax_rate = cfg["INCOME_TAX_HIGH_RATE"] if taxable_paye_income > srcop else cfg["INCOME_TAX_STD_RATE"]

        total_credits = self.get_tax_credits(cfg)
        net_income_tax = max(0, gross_income_tax - total_credits)

        prsi, prsi_marginal = self.calculate_prsi(total_income_for_prsi_usc, cfg)
        usc, usc_marginal = self.calculate_usc(total_income_for_prsi_usc, cfg)

        total_taxes = net_income_tax + prsi + usc
        take_home = taxable_base - self.pension_contribution - total_taxes
        
        marginal_overall_rate = marginal_income_tax_rate + prsi_marginal + usc_marginal
        effective_rate = (total_taxes / self.gross_income) * 100

        return {
            "Core Financials": {
                "Gross Compensatory Value": self.gross_income,
                "Voucher Allocation": self.voucher_allocation,
                "Cycle to Work": self.cycle_to_work,
                "Pension Deduction": self.pension_contribution,
                "Benefits In Kind (BIK)": self.bik,
            },
            "Tax Deductions": {
                "Gross Income Tax": round(gross_income_tax, 2),
                "Tax Credits Applied": round(total_credits, 2),
                "Net Income Tax (PAYE)": round(net_income_tax, 2),
                "USC": round(usc, 2),
                "PRSI": round(prsi, 2),
                "Rent Tax Credit": round(self.rent_tax_credit, 2),
                "Cycle to Work": round(self.cycle_to_work, 2)
            },
            "Summary": {
                "Total Tax Deduced": round(total_taxes, 2),
                "Take Home CASH": round(take_home, 2),
                "Effective Tax Rate (%)": round(effective_rate, 2),
                "Marginal Tax Rate (%)": round(marginal_overall_rate * 100, 2)
            }
        }

    def _build_empty_response(self) -> dict:
        return {
            "Core Financials": {"Gross Compensatory Value": 0.0, "Voucher Allocation": 0.0, "Cycle to Work": 0.0, "Pension Deduction": 0.0, "Benefits In Kind (BIK)": 0.0},
            "Tax Deductions": {"Gross Income Tax": 0.0, "Tax Credits Applied": 0.0, "Net Income Tax (PAYE)": 0.0, "USC": 0.0, "PRSI": 0.0, "Rent Tax Credit": 0.0, "Cycle to Work": 0.0},
            "Summary": {"Total Tax Deduced": 0.0, "Take Home CASH": 0.0, "Effective Tax Rate (%)": 0.0, "Marginal Tax Rate (%)": 0.0}
        }

    def _objective_function(self, x, utility_weight_pension: float, utility_weight_voucher: float, utility_weight_cycle: float) -> float:
        self.pension_contribution = x[0]
        self.voucher_allocation = x[1]
        self.cycle_to_work = x[2]
        
        result = self.calculate()
        take_home_cash = result["Summary"]["Take Home CASH"]
        
        utility_pension = utility_weight_pension * self.pension_contribution
        utility_voucher = utility_weight_voucher * self.voucher_allocation
        utility_cycle = utility_weight_cycle * self.cycle_to_work
        
        total_utility = take_home_cash + utility_pension + utility_voucher + utility_cycle
        return -total_utility

    def optimize(self, utility_weight_pension=1.2, utility_weight_voucher=0.90, utility_weight_cycle=0.85):
        max_pension = self.get_max_pension_limit()
        max_voucher = 1000.0  
        max_cycle = self.get_max_cycle_to_work_limit()
        
        bounds = [
            (0.0, max_pension),
            (0.0, max_voucher),
            (0.0, max_cycle)
        ]
        
        x0 = [10.0, 10.0, 10.0]
        
        res = minimize(
            lambda x: self._objective_function(x, utility_weight_pension, utility_weight_voucher, utility_weight_cycle), 
            x0,
            bounds=bounds,
            method='Powell'
        )
        
        if res.success:
            self.pension_contribution = res.x[0]
            self.voucher_allocation = res.x[1]
            self.cycle_to_work = res.x[2]
            final_result = self.calculate()
            
            print(json.dumps(final_result, indent=4))
            print("\n" + "="*50)
            print("MULTIDIMENSIONAL UTILITY OPTIMIZATION RESULT:")
            print(f"Optimal Pension Allocation: €{round(self.pension_contribution, 2)} (Bound: €{round(max_pension,2)})")
            print(f"Optimal Voucher Allocation: €{round(self.voucher_allocation, 2)} (Bound: €{round(max_voucher,2)})")
            print(f"Optimal Cycle to Work Allocation: €{round(self.cycle_to_work, 2)} (Bound: €{round(max_cycle,2)})")
            print(f"- Pension utility metric: {utility_weight_pension}")
            print(f"- Voucher utility metric: {utility_weight_voucher}")
            print(f"- Cycle utility metric: {utility_weight_cycle}")
            print("="*50 + "\n")
        else:
            print("Bounded multidimensional optimization failed.")

    def marginal_rate_curve(self, max_income: float = 200_000, step: float = 500) -> list[dict]:
        curve = []
        original_gross = self.gross_income
        original_pension = self.pension_contribution
        original_voucher = self.voucher_allocation
        original_cycle = self.cycle_to_work
        
        self.pension_contribution = 0.0
        self.voucher_allocation = 0.0
        self.cycle_to_work = 0.0
        
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
        return curve

    def print_marginal_curve(self, max_income: float = 120_000, step: float = 1_000):
        curve = self.marginal_rate_curve(max_income, step)
        print(f"{'Gross Income':<15} | {'Marginal Rate %':<16} | {'Effective Rate %':<16} | {'Notes'}")
        print("-" * 75)
        for row in curve:
            notes = "USC Exemption Kink (Jump > 80%)" if row.get("usc_kink") else ""
            print(f"€{row['gross_income']:<14.2f} | {row['marginal_rate_pct']:<16.2f} | {row['effective_rate_pct']:<16.2f} | {notes}")


if __name__ == "__main__":
    
    # Instantiate the calculator matching the test conditions exactly:
    # gross_income=49000, age=24, cycle_type="ebike", cycle_to_work_mode="annual"
    calc = IrishTaxCalculator(
        gross_income=49000.0,
        age=24,
        pension_contribution=0.0,
        voucher_allocation=0.0,
        cycle_to_work=0.0,
        cycle_type="ebike",
        cycle_to_work_mode="annual",
        bik=0.0,
        employment_type="PAYE",
        marital_status="Single",
        medical_card=False,
        rent_tax_credit=1000.0
    )

    # Call the multidimensional utility optimizer to evaluate the 3D surface parameters
    calc.optimize(utility_weight_pension=1.2, utility_weight_voucher=0.90, utility_weight_cycle=0.85)

