# Slide Deck — Credit Assessment Environment

**Target length:** 10 slides (cover + 8 content + close)
**Intended use:** 3-minute pitch (Round 1) + Q&A fallback
**Tool:** Google Slides / Keynote / PowerPoint. 16:9 aspect ratio.
**Visual style:** Dark background, monospaced fonts for code/numbers, 1–2 accent colors (pink + cream match the HF Space emoji).

Each slide below has: **title**, **on-slide content**, **what you say** (speaker notes), and **visual cue**.

Rule of thumb: **no more than 25 words on any slide**. The slide is the stage, not the script.

---

## Slide 1 — Cover (~15s)

**Title:** `Credit Assessment Environment`
**Subtitle:** `Can an LLM learn to be a loan officer — without seeing a real loan?`
**Bottom-right corner:** OpenEnv logo, HF logo, your name + team, hackathon name
**Visual cue:** Single hero image — the `personal_loan_hero.png` asset, or a minimal vector illustration of a loan document with a magnifying glass.

**Speaker notes:**
> "Today I'll show you an environment where an LLM is handed 10 RBI loan underwriting rules and a reward signal, and learns to become a credit officer through self-play."

---

## Slide 2 — The problem (~20s)

**Title:** `The gap LLMs hit on rule-following tasks`

**Body (bulleted):**
- ✅ LLMs are great at **pattern-matching** ("high income, looks approvable")
- ❌ LLMs are bad at **precise rule adherence** ("CIBIL 699 ≠ CIBIL 700")
- Real loan underwriting punishes the first and demands the second
- RBI rules have **tiers, exceptions, and hard cutoffs** you can't vibe-check

**Visual cue:** Split-panel diagram. Left panel: an "approve" green checkmark next to a glowing profile card. Right panel: the same card with "CIBIL 699" pulsing red, overlaid with a rejection stamp.

**Speaker notes:**
> "Pattern-matching gets you 60% on a loan dataset. The remaining 40% is rule-following — tiered RBI limits, RERA checks, FOIR boundaries exactly at 50 percent. That's the gap."

---

## Slide 3 — The environment (~25s)

**Title:** `An OpenEnv environment with 3 escalating loan types`

**Body (table):**

| Task | Loan Type | Key challenge |
|---|---|---|
| 1 · Easy | Personal | CIBIL, FOIR, employment |
| 2 · Medium | Vehicle | + LTV ratio, collateral |
| 3 · Hard | Home | + **RBI tiered LTV**, RERA compliance |

**Bottom note:** `Actions: approve · reject · request_docs · counter_offer`

**Visual cue:** Show one real applicant profile card (rendered from `build_profile_text`) next to the 4 action buttons.

**Speaker notes:**
> "Three tasks of escalating difficulty. Four actions. Multi-step episodes where the applicant responds — request docs and they come back with papers; counter-offer and the loan amount is recalculated."

---

## Slide 4 — The reward (~20s)

**Title:** `An asymmetric reward that encodes real NPA economics`

**Body (number-prominent):**
- ✅ Correct decision → `+10`
- ❌ Reject a good applicant → `−5` (lost revenue)
- ❌ Approve a bad loan → `−15` (NPA risk, 3× worse)
- 🚨 **Approve a non-RERA home loan → `−20`** (regulatory liability)

**Bottom note:** Not gameable — rejecting everything tops out at ~−5 average.

**Visual cue:** A horizontal bar chart with 4 bars colored green/yellow/orange/red.

**Speaker notes:**
> "The reward structure matches actual banking economics. Approving a bad loan costs three times more than rejecting a good one. RERA breach is worst — that's regulatory, not just credit risk."

---

## Slide 5 — The innovation: self-improving environment (~30s) — ⭐ MAIN POINT

**Title:** `The environment improves its own training data`

**Body (numbered loop diagram):**
1. Model trains on adversarial cases
2. `AdversarialTracker` finds the weakest strategy
3. Next batch is biased toward that weakness
4. **Model is prompted to design its own trap cases**
5. Cases are verified by the rule engine, then fed back to step 1

**Visual cue:** A circular loop diagram with the 5 steps as arrows. Step 4 (self-generation) highlighted in the accent color.

**Speaker notes:**
> "This is the Theme-4 self-improvement angle. After every adversarial round we ask the model to *design* traps targeting its own weaknesses. Rule engine verifies the cases before they're used as training data. The better the model gets, the harder it makes its own training."

---

## Slide 6 — Training results (~25s)

**Title:** `Measured improvement on Qwen2.5-1.5B`

**Body (two charts side-by-side):**
- **Left:** reward curve — x=step, y=normalized reward, clear upward trend
- **Right:** per-task accuracy — baseline vs trained, grouped bars per loan type

**Bottom note:** `+29% on Personal Loans · +17% on Home Loans · −14% on Vehicle Loans (the interesting one)`

**Visual cue:** Real plots from `assets/reward_curve.png` and `assets/per_task_accuracy.png` (regenerate tomorrow).

**Speaker notes:**
> "Standard GRPO lifts Personal Loans by nearly 30 points. Home Loans improve 17. But Vehicle Loans regress — and that regression is *exactly* why we built curriculum learning on top. The environment revealed its own training gap."

---

## Slide 7 — The discovery (~25s)

**Title:** `Why Vehicle Loans regressed — and why that's valuable`

**Body (pull quote, large):**

> **"An agent that exploits the reward without solving the task should not get high scores."**
> — OpenEnv Hackathon judge guide

**Below quote (smaller):**
- Standard training mixes all difficulties — model over-fits easy rules
- Curriculum (easy → medium → hard) + performance-gated phase advancement
- Adversarial self-play targets the specific trap patterns the model missed

**Visual cue:** The pull quote centered. Below it, a minimal diagram of the 3-phase curriculum arrow, with a second arrow showing adversarial rounds.

**Speaker notes:**
> "The Vehicle Loan regression was the discovery. A flat reward signal makes the model over-optimize. Curriculum learning plus adversarial self-play are how we recover — and they're built *on top of* the environment, not baked into it."

---

## Slide 8 — What's shipped (~20s)

**Title:** `Fully compliant with the hackathon's minimum requirements`

**Body (checklist):**
- ✅ OpenEnv 0.2+ (`Environment` base class, Gym-style API, `openenv.yaml`)
- ✅ Live Hugging Face Space (`iamnijin/credit-assessment-env`)
- ✅ HF TRL `GRPOTrainer` training script + **Colab notebook**
- ✅ 63 regression tests on reward + ground-truth + adversarial strategies
- ✅ Real training plots (reward curve, per-task, adversarial rounds)
- ✅ `./validate-submission.sh` passes 3/3

**Visual cue:** Screenshot of the validator green output (`assets/validation_output.txt`).

**Speaker notes:**
> "Everything the hackathon asks for is shipped. OpenEnv compliance, live Space, Colab, tests, plots, validator passing. Links in the README."

---

## Slide 9 — Sub-theme & extensibility (~15s)

**Title:** `Targeting Snorkel AI — Simulated Experts-in-the-Loop`

**Body:**
- The trained model **acts as its own simulated expert** — designs new trap cases each round
- Extensible to new loan types (business, education, gold) in **4 files**: generator, ground truth, reward, router

**Visual cue:** A tree diagram with `credit_assessment_env/` expanding into `generators/`, `ground_truth/`, `rewards/` — each already has `personal_loan.py`, `vehicle_loan.py`, `home_loan.py`, with a greyed-out `business_loan.py` placeholder.

**Speaker notes:**
> "We're also claiming the Snorkel sub-theme — the self-generation loop *is* a simulated expert. And the modular design means new loan types plug in as four files, no core changes."

---

## Slide 10 — Close / QR (~15s)

**Title:** `Try it. Train it. Extend it.`

**Body (three QR codes or URLs):**
- 🌐 `huggingface.co/spaces/iamnijin/credit-assessment-env`
- ▶️ Colab: `[badge link from README]`
- 📺 YouTube demo: `[your video URL]`

**Visual cue:** 3 QR codes, equally sized. Your name + email / GitHub handle in the footer.

**Speaker notes:**
> "Environment's live, training's in Colab, the full pitch including architecture and trap examples is in the README. Happy to take questions."

---

## Q&A fallback slides (hidden / appendix)

Keep these hidden in the appendix of the deck and show them only if asked.

### A1 — "Is the self-generation loop verified to actually improve results?"

> Cases are verified against deterministic `calculate_ground_truth` before being added to training. Self-generated cases are capped at 30% of each adversarial batch to prevent distribution collapse. The training script logs self-generated case count per round — see `train_grpo.py:1151`.

### A2 — "How do you prevent reward hacking?"

> Three safeguards:
> 1. Rejecting everything tops out at −5 average reward (worse than correct behavior).
> 2. Approving everything tops out at −15 / −20 (catastrophic).
> 3. Invalid JSON or missing fields costs −0.5 of the normalized reward — the model can't hack by emitting garbage.

### A3 — "Why is the model sometimes worse on Vehicle Loans?"

> Standard training without curriculum over-optimizes for the highest-density patterns in the training set. Personal Loans dominate the easy bucket, Home Loans force LTV computation, but Vehicle Loans sit in the "medium uncanny valley." Our solution: curriculum + adversarial self-play specifically targeted at that bucket. Vehicle Loan accuracy is expected to recover in the 7B / longer training runs.

### A4 — "Could this fit another theme?"

> Primary: Theme 4 (Self-Improvement). Secondary: Theme 3.1 (World Modeling / Professional Tasks — real RBI regulatory constraints). Sub-theme: Snorkel AI (Simulated Experts-in-the-Loop).

### A5 — "How big is the environment? Is it reusable?"

> ~750 LOC of environment code. Four files to add a new loan type. Unit test suite protects refactors. Docker image under 2 GB.

---

## Design brief for whoever makes the actual deck

- **Font:** Inter or SF Pro for body, JetBrains Mono / Fira Code for numbers and code. Size: title 44pt, body 24pt minimum (judges may view on a projector).
- **Color palette:** Background `#0F0F12` (near-black). Text `#F4E9D8` (warm cream). Accent `#FF6F91` (pink — matches the HF Space emoji frontmatter `colorTo: pink`). Secondary accent `#8AB6D6` (muted blue for data).
- **Charts:** Light backgrounds on dark cards. Always label both axes. Caption under each chart in italic.
- **Transitions:** None. Cuts only. Fancy transitions eat time.
- **Logos:** OpenEnv + Hugging Face on the cover and closing slide. Meta's OpenEnv repo: https://github.com/facebookresearch/openenv
