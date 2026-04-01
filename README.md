---
title: Credit Assessment Environment
emoji: 🏦
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# Credit Assessment Environment

An RL training environment that simulates real-world Indian loan underwriting. The agent receives a loan application and must decide: approve, reject, request documents, or counter-offer — just like a bank's credit officer would.

Built on [OpenEnv](https://github.com/facebookresearch/openenv), this environment supports three loan types of increasing difficulty:

| Task | Loan Type | Difficulty | Key Challenge |
|------|-----------|------------|---------------|
| 1 | Personal Loan | Easy | Credit score, FOIR, employment checks. Trap cases: high income but credit just below 700 |
| 2 | Vehicle Loan | Medium | Adds LTV ratio and collateral. Traps: excellent profile but LTV over 85%, rich applicant with low credit |
| 3 | Home Loan | Hard | RBI tiered LTV (must compute from property/loan values), RERA compliance traps, employment threshold differs from other loans |

## Quick Start

```bash
git clone https://huggingface.co/spaces/iamnijin/credit-assessment-env
cd credit-assessment-env
uv pip install .
```

```python
from credit_assessment_env import CreditAssessmentAction, CreditAssessmentEnv
from credit_assessment_env.loan_decision import LoanDecision

# Connect to the live HuggingFace Space
env = CreditAssessmentEnv(base_url="https://iamnijin-credit-assessment-env.hf.space").sync()

with env:
    result = env.reset()
    print(result.observation.applicant_profile)

    result = env.step(CreditAssessmentAction(
        decision=LoanDecision.APPROVE,
        reasoning="Strong credit score and low FOIR"
    ))
    print(f"Reward: {result.reward}, Done: {result.done}")
```

## Underwriting Criteria

The ground truth decisions are based on actual Indian banking norms and RBI guidelines. Here's what drives each decision:

### Credit Score (CIBIL)

All three loan types require a minimum CIBIL score of **700**. This aligns with what major Indian banks (HDFC, ICICI, SBI) use as their floor for loan approval. Scores of 750+ typically get preferential rates, but 700 is the cutoff for basic eligibility.

- Personal Loan: min 700
- Vehicle Loan: min 700
- Home Loan: min 700

**References:**
- [HDFC Bank - Personal Loan Eligibility](https://www.hdfcbank.com/personal/products/loans/personal-loans/eligbilty-criteria)
- [ICICI Bank - Car Loan Credit Score](https://www.icici.bank.in/blogs/car-loan/minimum-credit-score-required-to-get-car-loan)
- [Bajaj Markets - Home Loan CIBIL Score](https://www.bajajfinservmarkets.in/cibil-score/cibil-score-for-home-loan)

### FOIR (Fixed Obligation to Income Ratio)

FOIR above **50%** triggers a rejection across all loan types. Most banks cap FOIR at 40-50% for salaried applicants. While secured loans (vehicle, home) sometimes stretch to 55-60% for high-income profiles, I've kept it at 50% across the board for cleaner training signals.

**References:**
- [TechFinServ - FOIR & Multiplier Explained](https://www.techfinserv.com/blogs/foir/)
- [MoneyControl - FOIR Impact on Loan Eligibility](https://www.moneycontrol.com/news/business/personal-finance/foir-calculation-impact-of-foir-on-personal-loan-eligibility-and-tips-to-improve-it-12970645.html)

### LTV Ratio (Loan to Value)

**Vehicle Loans:** LTV above **85%** triggers a counter-offer (reduce loan amount). SBI's published norm is 85% max on the on-road price. Most banks operate in the 80-90% range for new vehicles.

**Home Loans:** This follows the RBI's tiered structure from the Master Circular on Housing Finance:
- Loan ≤ ₹30 lakh → max LTV **90%**
- Loan ₹30-75 lakh → max LTV **80%**
- Loan > ₹75 lakh → max LTV **75%**

Breaching the applicable LTV tier triggers a counter-offer rather than outright rejection, since the applicant may still qualify with a higher down payment.

**References:**
- [RBI Master Circular on Housing Finance (LTV Norms)](https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=6161&Mode=0)
- [MagicBricks - RBI Home Loan Guidelines 2026](https://www.magicbricks.com/blog/rbi-guidelines-for-home-loans/127863.html)
- [Axis Bank - Car Loan LTV](https://axisbank.com/progress-with-us-articles/money-matters/borrow/what-is-a-good-car-loan-ltv-ratio)

### RERA Registration (Home Loans)

If the property is **not RERA registered**, it's an immediate rejection. Banks in India have collectively decided (in consultation with RBI) not to finance non-RERA projects. This is a hard compliance requirement, not a risk judgement.

**Reference:**
- [Enterslice - No Loan to Non-RERA Builders](https://enterslice.com/learning/no-loan-builders-not-registered-rera/)

### Employment Stability

- Personal Loan: minimum **1 year** at current employer
- Vehicle Loan: minimum **1 year**
- Home Loan: minimum **2 years** (longer commitment = stricter check)

Most banks (HDFC, Axis) want at least 1 year at current employer and 2 years total experience for personal loans. For home loans with 15-20 year tenures, employment stability matters more.

**References:**
- [HDFC Bank - Personal Loan Eligibility](https://www.hdfcbank.com/personal/products/loans/personal-loans/eligbilty-criteria)
- [Axis Bank - Personal Loan Eligibility](https://www.creditmantri.com/axis-bank-personal-loan-eligibility/)

### Document Completeness

Incomplete documents → `request_docs`. This is always the first check. No bank processes a loan without complete documentation.

## Reward Structure

The reward function reflects the asymmetric cost structure of real banking:

| Scenario | Reward | Rationale |
|----------|--------|-----------|
| Correct decision (matches ground truth) | **+10.0** | Ideal outcome |
| Request docs when docs are incomplete | **+3.0** | Correct procedural step, partial credit |
| Counter-offer catches high LTV (vehicle) | **+7.0** | Smart risk mitigation, near-correct |
| Reject a good applicant | **-8.0** | Lost revenue, but recoverable |
| Approve a bad applicant | **-15.0** | NPA risk — far more costly than lost revenue |
| Approve non-RERA home loan | **-20.0** | Regulatory/legal risk — worst case for the bank |
| Counter-offer without specifying amount | **-5.0** | Invalid action penalty |
| Other wrong decisions | **-2.0** | Wrong but not catastrophic |

The key asymmetry: **approving a bad loan (-15) is penalized almost twice as hard as rejecting a good one (-8)**. This mirrors banking reality where NPA losses (principal + interest + recovery costs) far exceed the opportunity cost of a declined good customer.

## Action Space

```python
class CreditAssessmentAction:
    decision: LoanDecision      # approve / reject / request_docs / counter_offer
    reasoning: str               # why the agent made this decision
    counter_offer_amount: float  # required only if decision is counter_offer
    docs_requested: str          # required only if decision is request_docs
```

## Observation Space

Each observation contains:
- `applicant_profile` — human-readable text summary for LLM reasoning
- Core fields: `credit_score`, `monthly_income`, `foir`, `employment_years`, `loan_type`, `loan_amount`, `documents_complete`
- Secured loan fields (when applicable): `collateral_value`, `ltv_ratio`
- Home loan fields (when applicable): `rera_registered`, `has_co_applicant`
- Episode fields: `task_id`, `reward`, `done`, `available_actions`

## Episode Flow

![Episode Flow](assets/episode_flow.png)

1. `reset()` → generates a loan application (good, bad, borderline, or **trap** profile)
2. Agent receives observation with applicant details as narrative text
3. Agent submits action (approve/reject/request_docs/counter_offer)
4. Environment returns reward and next state
5. Episode ends when agent makes a final decision (approve/reject) or after 3 steps

Multi-step episodes:
- **`request_docs`** → the same applicant returns with documents now complete, agent must re-evaluate
- **`counter_offer`** → the same applicant returns with the reduced loan amount, LTV recalculated

## What Makes This Hard for LLMs

The environment includes **trap profiles** designed to test whether an LLM can follow rules precisely even when the overall picture looks good:

- **Trap: Perfect profile, one hidden flaw** — credit score 690 (just below 700) with ₹2L income, 20% FOIR, 10 years employment. Everything screams "approve" except the score.
- **Trap: Co-applicant doesn't fix FOIR** — an LLM might assume having a co-applicant compensates for 52% FOIR. It doesn't — FOIR is based on the primary applicant.
- **Trap: LTV tier mismatch** — for a ₹1.2Cr home loan, LTV of 78% looks safe if you assume an 80% limit. But RBI caps loans > ₹75L at 75%. The home loan profile doesn't show pre-computed LTV — the agent must calculate it from property value and loan amount.
- **Trap: Non-RERA with everything else green** — 830 credit score, ₹2L income, 20% FOIR, but RERA = No. Must reject despite the dream financials.

## Project Structure

```
credit_assessment_env/
├── inference.py               # LLM inference entry point
├── baseline.py                # Baseline agents (Random, Rule-Based, LLM)
├── models.py                  # Action and Observation schemas
├── loan_decision.py           # LoanDecision enum
├── client.py                  # CreditAssessmentEnv client
├── openenv.yaml               # OpenEnv manifest
├── server/
│   ├── credit_assessment_env_environment.py  # Main environment (orchestrator)
│   ├── app.py                 # FastAPI server
│   ├── generators/            # Applicant generation per loan type
│   │   ├── personal_loan.py
│   │   ├── vehicle_loan.py
│   │   └── home_loan.py
│   ├── ground_truth/          # Decision rules per loan type
│   │   ├── personal_loan.py
│   │   ├── vehicle_loan.py
│   │   └── home_loan.py
│   ├── rewards/               # Reward calculation per loan type
│   │   ├── personal_loan.py
│   │   ├── vehicle_loan.py
│   │   └── home_loan.py
│   └── helpers/
│       └── profile_builder.py # Builds LLM-readable applicant profiles
```

### Adding a New Loan Type

The modular structure makes it straightforward to add new loan types (e.g., business loan, education loan, gold loan). Four steps:

1. **Generator** — create `server/generators/business_loan.py` with a `generate_business_loan()` function that returns an applicant dict. Include good/bad/borderline/trap profiles.
2. **Ground truth** — create `server/ground_truth/business_loan.py` with the underwriting rules (e.g., business vintage > 3 years, ITR filed for last 2 years, GST registration).
3. **Reward** — create `server/rewards/business_loan.py` with domain-specific reward shaping (e.g., extra penalty for approving a shell company).
4. **Register** — add the new loan type to each `__init__.py` router and add a new task entry in `TASKS` inside `credit_assessment_env_environment.py`.

No changes needed to `models.py`, `client.py`, or the Dockerfile — the observation schema uses optional fields that accommodate any loan type.

## Baseline Scores

Run from the **parent directory** (the one containing `credit_assessment_env/`):

```bash
# Random + Rule-Based only (no API key needed)
python -m credit_assessment_env.baseline --episodes 100 --seed 42

# Include LLM agent (requires OPENAI_API_KEY in environment or .env file)
OPENAI_API_KEY=sk-... python -m credit_assessment_env.baseline --llm --model gpt-4o-mini --episodes 10
```

The baseline also supports Azure OpenAI — set `OPENAI_BASE_URL` and `OPENAI_API_VERSION` in a `.env` file alongside `OPENAI_API_KEY`.

Results from `--episodes 10 --seed 42`:

| Agent | Task 1 (Personal, Easy) | Task 2 (Vehicle, Medium) | Task 3 (Home, Hard) | Overall |
|-------|------------------------|--------------------------|---------------------|---------|
| Random | 0.467 | 0.350 | 0.400 | 0.406 |
| Rule-Based | 1.000 | 1.000 | 1.000 | 1.000 |
| GPT-4o-mini | 0.900 | 0.900 | 0.700 | 0.833 |
| GPT-5 | 0.700 | 0.700 | 0.700 | 0.700 |

![Baseline Results (GPT-4o-mini)](assets/baseline_result_gpt-4o-mini.png)

The **Random agent** picks a decision at random — lower bound.

The **Rule-Based agent** follows the exact ground truth logic — upper bound at 1.0.

**GPT-4o-mini** scores 0.833 overall, handling personal and vehicle loans well (90%) but struggling with home loans (70%) where it must compute LTV from raw values and apply tiered RBI limits. **GPT-5** scores 0.700 — interestingly lower, likely due to running at temperature 1.0 (the only value GPT-5 supports) which introduces variability in rule-following.

The 17-30% gap between LLMs and the Rule-Based agent is the value of this environment: there's measurable room for improvement, and closing the gap requires precise rule adherence over pattern matching.

## Inference Script

The `inference.py` at the project root runs an LLM agent against all 3 tasks using the OpenAI client.

**Environment variables:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | Yes | — | API key / HF token |
| `LOCAL_IMAGE_NAME` | No | `credit_assessment_env-env:latest` | Local Docker image name |
| `TASK_NAME` | No | `all` | Task to run: `all`, `1`, `2`, `3`, `personal-loan`, `vehicle-loan`, `home-loan` |
| `BENCHMARK` | No | `credit-assessment` | Benchmark label used in log output |

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_..."
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

uv run python inference.py
```

The script runs 10 episodes per task (seed 42) and emits structured stdout in the required hackathon format:

```
[START] task=personal-loan env=credit-assessment model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action=approve reward=10.00 done=true error=null
[END] success=true steps=1 rewards=10.00
```

It uses the standard `openai.OpenAI` client, so any OpenAI-compatible endpoint works — HuggingFace Inference, OpenAI API, Azure OpenAI, etc.

![Inference Results](assets/inference_sample.png)

## Building & Running

```bash
# Build and run with Docker
docker build -t credit_assessment_env-env:latest .
docker run -p 7860:7860 credit_assessment_env-env:latest

# Run locally with uv (from the parent directory)
uv run uvicorn credit_assessment_env.server.app:app --host 0.0.0.0 --port 7860 --reload

# Deploy to Hugging Face Spaces
openenv push
```

## References

- [RBI Master Circular on Housing Finance](https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=6161&Mode=0)
- [RBI Circular on Consumer Credit Risk Weights (Nov 2023)](https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12567&Mode=0)
- [RERA Act and Bank Lending](https://enterslice.com/learning/no-loan-builders-not-registered-rera/)
- [FOIR Calculation and Impact](https://www.techfinserv.com/blogs/foir/)
- [CIBIL Score Requirements by Loan Type](https://paytm.com/blog/credit-score/the-ultimate-guide-to-cibil-score-requirements-for-different-loan-types-home-car-personal-2/)
- [Personal Loan Rules 2026 (RBI Guidelines)](https://pragatiinstitute.co.in/personal-loan-rules-2026/)
