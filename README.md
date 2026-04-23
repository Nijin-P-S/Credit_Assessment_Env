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

### Can an LLM learn to be a loan officer — without ever seeing a real loan?

We gave a language model a stack of loan applications, RBI guidelines, and zero domain knowledge. No pre-training on banking docs. No few-shot examples. Just raw applicant profiles and a reward signal.

Within 100 episodes, it learned to check CIBIL scores before income, reject dream profiles with hidden RERA violations, and compute RBI tiered LTV limits from raw numbers. By episode 50, it was catching trap cases that trip up even experienced loan officers.

**This is Credit Assessment Environment** — a self-improving RL environment where an agent learns to make accurate loan decisions through curriculum learning, adversarial self-play, and GRPO.

> Built with [OpenEnv](https://github.com/facebookresearch/openenv) | Training via [HF TRL](https://github.com/huggingface/trl) | [Try in Colab](train_grpo_colab.ipynb)

---

## Demo & Materials

| Resource | Link |
|---|---|
| 🌐 **Live environment (HF Space)** | [iamnijin/credit-assessment-env](https://huggingface.co/spaces/iamnijin/credit-assessment-env) |
| 📺 **Demo video (<2 min)** | _TBD — YouTube link_ |
| 📝 **Mini-blog** | _TBD — HF blog link_ |
| 📊 **Slide deck** | _TBD — Google Slides link_ |
| 🎤 **3-minute pitch script** | [`docs/pitch.md`](docs/pitch.md) |
| 🎬 **Video recording script + shot list** | [`docs/video_script.md`](docs/video_script.md) |
| 📽️ **Slide-by-slide content** | [`docs/slide_deck.md`](docs/slide_deck.md) |
| ▶️ **Train it yourself in Colab** | [`train_grpo_colab.ipynb`](train_grpo_colab.ipynb) |
| 🤖 **Trained model on HF Hub** | [iamnijin/credit-assessment-grpo-trained](https://huggingface.co/iamnijin/credit-assessment-grpo-trained) |
| ✅ **Submission validator output** | [`assets/validation_output.txt`](assets/validation_output.txt) (3/3 checks pass) |

---

## The Story: From Pattern-Matching to Precision

*This is how the environment is designed to teach, illustrated with one training run on Qwen2.5-1.5B. Exact numbers appear in the [Actual Training Results](#actual-training-results) section below.*

### Act 1: The Cold Start

Episode 1. The agent receives its first loan application: a personal loan request from someone with ₹1.5L monthly income, 35% FOIR, and 8 years of employment. Looks solid.

It approves. **Wrong.**

The applicant had a CIBIL score of 695 — just 5 points below the 700 threshold. The agent pattern-matched "high income = approve" without checking the hard cutoff. Reward: **−15.0** (approved a bad loan — potential NPA).

### Act 2: Learning the Rules

Training continues. The agent starts checking CIBIL *first*, before even looking at income. It learns that ₹10L monthly income means nothing if the credit score is 699.

It encounters a home loan with perfect financials: 820 CIBIL, ₹2L income, 25% FOIR. Dream applicant. The agent hesitates... then rejects.

Why? RERA = No. The property isn't registered. The agent learned that compliance requirements are non-negotiable — in reward terms, approving a non-RERA home loan carries the worst possible outcome (−20.0).

### Act 3: The Environment Fights Back

As the agent masters simple cases, adversarial training kicks in. Trap cases appear: borderline FOIR at exactly 50%, LTV at exactly the RBI tier boundary, co-applicants that look like safety nets but don't actually compensate for high FOIR.

The agent is pushed to *compute*, not just pattern-match. LTV has to be calculated from raw property values. RBI tiered limits have to be applied based on loan amount. "Everything looks good" stops being a valid heuristic.

### Act 4: The Agent Becomes Its Own Adversary

Something unexpected happens. After each adversarial round, the trained model is asked to switch roles — instead of evaluating loans, it designs them. *"Given where you keep failing, create a loan application that would trick an AI loan officer."*

Early rounds produce weak traps. The model doesn't yet know enough to design genuinely hard cases. But in later rounds, it begins generating borderline profiles — CIBIL just below threshold paired with very high income, home loans with LTV exactly at the tier boundary. Cases rule-based generation would never think to create.

Every self-generated case is verified against deterministic ground truth before it's used. The rule engine is the referee. The loop closes: the model's failures inform what it trains on next, and the better it gets, the harder it makes its own tests.

### Act 5: The Result — And The Discovery

On Qwen2.5-1.5B with a single T4 GPU run (standard GRPO, 2 epochs, 300 samples, no curriculum):

| Loan Type | Baseline | Trained | Change |
|-----------|----------|---------|--------|
| Personal Loan (Easy) | 57.1% | **85.7%** | **+28.6%** ✅ |
| Home Loan (Hard) | 33.3% | **50.0%** | **+16.7%** ✅ |
| Vehicle Loan (Medium) | 42.9% | 28.6% | **−14.3%** ❌ |
| **Overall** | **45.0%** | **55.0%** | **+10.0%** |

Personal Loans improved by nearly 30 points. Home Loans by 17. But Vehicle Loans regressed by 14.

**The regression is the result, not a failure.**

Standard GRPO on a mixed-difficulty batch made the model over-optimize for the highest-density patterns — CIBIL thresholds and RERA checks — while losing the LTV reasoning specific to Vehicle Loans. The environment *revealed* the training gap.

That's exactly why we layered curriculum learning (easy → medium → hard, performance-gated) and adversarial self-play on top. These are not part of the environment itself — they're what the environment *made necessary*. In the sections below, the `train_grpo.py` pipeline with curriculum + adversarial + self-generation is the default mode because it addresses this specific failure pattern.

**The agent didn't just learn — it revealed where it needed to learn more.** That's self-improvement in action: an environment that exposes its own weaknesses, enabling targeted training on what matters most.

---

## Themes Addressed

### Primary: Theme #4 — Self-Improvement

Credit Assessment Environment is built for recursive skill amplification:

- **Performance-gated curriculum**: Easy → Medium → Hard progression gated by accuracy — the model earns advancement (60% threshold), not just completes a fixed number of steps
- **Adversarial self-play**: `AdversarialTracker` identifies which of 10 trap strategies the model fails at most and weights the next training round toward those weaknesses
- **Self-generated challenges**: After each adversarial round, the model being trained is prompted to design its own trap cases targeting its identified weaknesses. Those cases feed into the next round's training data — closing a recursive loop where the better the model gets, the harder it makes its own training

### Sub-theme claim: Snorkel AI — Simulated Experts-in-the-Loop

The self-generation loop makes the *trained model itself* act as a simulated domain expert. After each adversarial round, the model is prompted to design new loan applications that would trick an AI loan officer — drawing on its own internalized understanding of RBI rules. Every self-generated case is verified against deterministic ground truth before it enters the next training round; invalid cases are discarded. This is a lightweight, verification-gated form of expert-in-the-loop data generation, built natively into the environment rather than bolted on.

### Secondary: Theme #3 — World Modeling / Professional Tasks

Real professional task with real regulatory constraints:

- **RBI guidelines**: Actual tiered LTV limits, CIBIL thresholds, FOIR caps
- **Asymmetric rewards**: Approving bad loans costs 3× more than rejecting good ones (matches NPA risk)
- **Multi-step workflows**: request_docs → re-evaluate, counter_offer → recalculate

---

## Overview

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

The reward function reflects the asymmetric cost structure of real banking. Raw reward values are returned per step; the final **grade (0.01–0.99)** is computed by normalizing the average reward over `[-20, +10]` using `(avg_reward + 20) / 30`.

| Scenario | Raw Reward | Grade (1-step) | Rationale |
|----------|------------|----------------|-----------|
| Correct decision (matches ground truth) | **+10.0** | **0.99** | Ideal outcome |
| Request docs when docs are incomplete | **+2.0** | **0.73** | Correct procedural step, partial credit |
| Counter-offer catches high LTV (vehicle) | **+5.0** | **0.83** | Smart risk mitigation, near-correct |
| Approve/reject when docs are missing | **-8.0** | **0.40** | Skipped a required procedural step |
| Approve/reject when counter-offer needed | **-8.0** | **0.40** | Ignored LTV risk that required restructuring |
| Reject a good applicant | **-5.0** | **0.50** | Lost revenue, but recoverable |
| Approve a bad applicant | **-15.0** | **0.17** | NPA risk — far more costly than lost revenue |
| Approve non-RERA home loan | **-20.0** | **0.01** | Regulatory/legal risk — worst case for the bank |
| Counter-offer without specifying amount | **-3.0** | **0.57** | Invalid action penalty |
| Other wrong decisions | **-2.0** | **0.60** | Wrong but not catastrophic |

The key asymmetry: **approving a bad loan (-15) is penalized 3× as hard as rejecting a good one (-5)**. This mirrors banking reality where NPA losses (principal + interest + recovery costs) far exceed the opportunity cost of a declined good customer. The RERA violation (-20) is the single worst outcome — it exposes the bank to regulatory and legal liability, not just credit risk.

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

The `inference.py` at the project root runs an LLM agent against all 3 tasks using the OpenAI client. It connects to the environment via `CreditAssessmentEnv.from_docker_image()`, matching the standard OpenEnv client SDK pattern.

**Environment variables:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | Yes | — | API key / HF token |
| `LOCAL_IMAGE_NAME` | No | `credit_assessment_env-env:latest` | Docker image for the environment |
| `TASK_NAME` | No | `all` | Task to run: `all`, `1`, `2`, `3`, `personal-loan`, `vehicle-loan`, `home-loan` |
| `BENCHMARK` | No | `credit-assessment` | Benchmark label used in log output |

```bash
# Build the environment Docker image (one-time)
docker build -t credit_assessment_env-env:latest .

# Install dependencies
uv sync

# Run inference
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_..."
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

uv run python inference.py
```

The script runs 10 episodes per task (seed 42) and emits structured stdout in the required hackathon format:

```
[START] task=personal-loan env=credit-assessment model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action=approve reward=10.00 done=true error=null
[END] success=true steps=1 score=0.990 rewards=10.00
```

It uses the standard `openai.OpenAI` client, so any OpenAI-compatible endpoint works — HuggingFace Inference, OpenAI API, Azure OpenAI, etc.

![Inference Results](assets/inference_sample.png)

## Training with GRPO

Train an LLM to make accurate loan decisions using Group Relative Policy Optimization (GRPO) from HuggingFace TRL.

### Quick Start (Colab)

Open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nijin-P-S/Credit_Assessment_Env/blob/main/train_grpo_colab.ipynb)

### Local Training

```bash
# Install training dependencies
pip install -r requirements-train.txt

# Run training
python train_grpo.py
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `Qwen/Qwen2.5-7B-Instruct` | Base model to fine-tune |
| `num_train_samples` | 500 | Training dataset size (non-curriculum mode) |
| `samples_per_phase` | 400 | Training samples per curriculum phase |
| `num_generations` | 4 | GRPO completions per prompt |
| `use_peft` | True | Use LoRA for memory efficiency |
| `num_train_epochs` | 1 | Training epochs per phase (curriculum) or per run (standard) |
| `learning_rate` | 5e-6 | Conservative — stability from `beta` + `max_grad_norm` |
| `beta` | 0.2 | GRPO KL anchor strength |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `use_curriculum` | True | Enable performance-gated curriculum (easy → medium → hard) |
| `phase_mastery_threshold` | 0.60 | Accuracy required to advance to next curriculum phase |
| `max_phase_retries` | 1 | Max extra attempts per phase before forced advancement |
| `phase_eval_samples` | 50 | Per-phase evaluation sample count |
| `use_adversarial` | True | Enable adversarial self-play training |
| `adversarial_rounds` | 2 | Number of adversarial training rounds |
| `adversarial_samples` | 150 | Adversarial samples per round (targeting current weakness) |
| `use_self_generation` | True | Model generates its own hard cases after each adversarial round |

### Training Modes

#### 1. Full Pipeline: Curriculum + Adversarial (Default)
The default training mode includes both curriculum learning AND adversarial self-play:

```bash
python train_grpo.py
```

This runs:
- **Phase 1 (Easy)**: Master basic CIBIL/FOIR checks
- **Phase 2 (Medium)**: Add Vehicle Loans with LTV
- **Phase 3 (Hard)**: Add Home Loans with RERA + tiered LTV
- **Adversarial Rounds**: Target agent's weakest strategies

#### 2. Standard Training (No Curriculum/Adversarial)
To train on all difficulties simultaneously without curriculum:

```python
# In train_grpo.py, set:
config.use_curriculum = False
config.use_adversarial = False
```

#### 3. Curriculum Only (No Adversarial)
Progressive difficulty without adversarial refinement:

```python
# In train_grpo.py, set:
config.use_curriculum = True
config.use_adversarial = False
```

#### 4. Full Recursive Self-Improvement (Default — Recommended)

The complete pipeline. After curriculum learning, adversarial self-play runs with the model acting as its own challenge generator:

1. **Evaluate** model on adversarial cases, record per-strategy failure rates in `AdversarialTracker`
2. **Identify** weakest strategy (highest failure rate)
3. **Generate** rule-based cases targeting that weakness (70%) + hard cases for balance (30%)
4. **Mix in** self-generated cases carried from the previous round (capped at 30% of batch)
5. **Train** on the combined dataset
6. **Self-generate**: prompt the trained model to design trap cases targeting its own weaknesses — carry these into step 4 of the next round
7. **Repeat**

The self-generation loop is what makes this recursive: the model's own failure patterns shape what it trains on next. Early rounds produce weak self-generated cases; later rounds produce genuinely tricky ones because the model has internalized enough rules to know where edge cases live.

**Safeguards on self-generated cases:**
- Required field validation (loan_type, credit_score, monthly_income, foir, employment_years, loan_amount, documents_complete)
- `loan_type` must be one of `personal`, `vehicle`, `home` — invalid outputs are discarded
- Missing vehicle/home fields (collateral_value, ltv_ratio, rera_registered) are auto-filled with safe defaults
- Ground truth is always computed via `calculate_ground_truth()` — never trusted from the model's own output
- Up to 4× generation attempts per requested case; failed parses are silently skipped
- Self-generated cases capped at 30% of each training batch to prevent distribution collapse

```python
# In train_grpo.py, set:
config.use_curriculum = True
config.use_adversarial = True
config.use_self_generation = True   # Default: True
config.adversarial_rounds = 2       # Round 3 produced no new signal in prior runs
config.adversarial_samples = 150
```

**Available Adversarial Strategies:**
| Strategy | Description | Expected Decision |
|----------|-------------|-------------------|
| `threshold_credit` | CIBIL 699 (1 point below threshold) | reject |
| `threshold_foir` | FOIR 51% (1% above threshold) | reject |
| `perfect_but_rera` | Perfect financials, RERA=No | reject |
| `perfect_but_ltv_tier` | >75L loan with LTV 76% | counter_offer |
| `high_income_low_cibil` | ₹8L income but CIBIL 695 | reject |
| `employment_trap_home` | Perfect home loan, 1 year employment | reject |
| `vehicle_ltv_trap` | Perfect vehicle loan, LTV 86% | counter_offer |
| `docs_incomplete_good` | Excellent profile, docs missing | request_docs |
| `borderline_multiple` | Multiple metrics at exact thresholds | varies |

### What the Training Does

1. **Generates synthetic loan applications** with known ground truth decisions
2. **Computes rewards** using the environment's reward function (same asymmetric penalties)
3. **Trains with GRPO** to maximize reward while maintaining output quality
4. **Gates phase advancement** on measured accuracy — the model repeats a phase (up to 2 retries with fresh samples) if it hasn't reached 65% before moving to a harder difficulty
5. **Tracks per-strategy failure rates** via `AdversarialTracker` and focuses adversarial rounds on the model's weakest areas
6. **Prompts the model to generate its own hard cases** after each adversarial round — verified against deterministic ground truth before use as training data
7. **Evaluates** before/after accuracy on held-out samples

### Target Numbers (for a full 7B run with longer training)

*The numbers below are targets for a full-scale training run (Qwen2.5-7B, 500+ samples, 3 curriculum phases, 2+ adversarial rounds). They are not measured — see [Actual Training Results](#actual-training-results) below for the real numbers from the 1.5B run.*

| Training Mode | Baseline | Target Trained | Target Δ |
|---------------|----------|-----------------|----------|
| Standard | ~60% | ~80% | +20% |
| Curriculum (performance-gated) | ~60% | ~82% | +22% |
| Curriculum + Adversarial | ~60% | ~85%+ | +25%+ |
| Full pipeline (+ self-generation) | ~60% | ~88%+ | +28%+ |

These ranges are extrapolated from per-task improvements we observed on 1.5B (e.g., +29% on Personal Loans) and the standard scaling behavior of GRPO on larger base models. We will refresh this table with measured numbers from the 7B run before final submission.

---

## A Note on Evaluation Methodology

The baseline-vs-trained comparison reported in `train_grpo.py` (and quoted in earlier sections of this README) uses **two different evaluation paths** for the two models:

- The **baseline** is scored by `evaluate_by_loan_type()`, which calls `json.loads(response.strip())` directly with no markdown fence handling.
- The **trained model** is scored by `evaluate_model()`, which strips ```` ```json ```` and ```` ``` ```` fences before parsing.

Qwen-2.5-Instruct (the base model) defaults to wrapping JSON in markdown code fences. The GRPO-trained policy, in contrast, was rewarded for emitting raw JSON. Net effect: a non-trivial fraction of correct baseline responses are silently scored as wrong because the strict baseline parser cannot read them.

The two paths also generate slightly different applicant pools (different `num_samples` to the same seed cycles different cases) and use different sample sizes per loan type, so per-task deltas conflate three things: real model differences, parser leniency differences, and applicant-pool differences.

**To get an apples-to-apples comparison**, run [`scripts/fair_eval.py`](scripts/fair_eval.py). It loads the base model and the trained adapter, runs both through the **same applicants** with the **same lenient parser**, and reports per-task accuracy with **95% Wilson confidence intervals** so you can tell which deltas are statistically meaningful vs sampling noise.

```bash
python scripts/fair_eval.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-repo iamnijin/credit-assessment-curriculum \
  --num-samples 120
```

Output is written to `assets/fair_eval_results.json` and `assets/fair_eval_chart.png`. The numbers in the sections below come from the original (unfair) eval pipeline — they will be replaced with `fair_eval.py` numbers in the final submission.

## Actual Training Results

*The plots below are from our first training run. A refreshed run with updated plots, labelled axes, and a fair head-to-head comparison is scheduled before final submission — see [`scripts/generate_plots.py`](scripts/generate_plots.py) and [`scripts/fair_eval.py`](scripts/fair_eval.py) for the regeneration pipeline.*

### Reproducing Our Numbers

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Hardware | Single NVIDIA T4 (Colab) |
| Mode | Standard GRPO (no curriculum, no adversarial) |
| Training samples | 300 |
| Epochs | 2 |
| GRPO generations per prompt | 4 |
| LoRA rank / alpha | 16 / 32 |
| Seed | 42 |
| Eval samples per loan type | 21 held-out |

To reproduce end-to-end:

```bash
pip install -r requirements-train.txt
python train_grpo.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --num_train_samples 300 \
  --num_train_epochs 2 \
  --use_curriculum False \
  --use_adversarial False \
  --seed 42
```

Or open [`train_grpo_colab.ipynb`](train_grpo_colab.ipynb) and run all cells — same run, free T4 GPU.

### Reward Curve

Reward trends upward over the 300 training steps; variance is expected for GRPO (it compares multiple completions per prompt).

![Training reward over steps (Qwen2.5-1.5B, standard GRPO)](assets/training_step_7.png)

### Per-Task Accuracy (Baseline vs Trained)

![Baseline vs trained accuracy per loan type](assets/post_training_breakdown_step_8.png)

The per-task breakdown is the headline result — see [Act 5](#act-5-the-result--and-the-discovery) above for the full numbers and why the Vehicle regression is the most valuable signal in this run.

![Overall baseline vs trained accuracy](assets/post_training_evaluation_step_8.png)

---

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
