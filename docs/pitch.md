# 3-Minute Pitch Script — Credit Assessment Environment

**Format:** 3 minutes pitch + 2 minutes Q&A = 5 minutes total.
**Timing aid:** Right margin shows target *cumulative* time. Aim to finish each section by its mark. If you're >5 seconds over at any checkpoint, trim the next section on the fly.

---

## Pitch opener (20 seconds) — ends at **0:20**

> "Can a language model learn to be a loan officer — without ever seeing a real loan?
>
> That's the environment we built. An LLM gets handed an Indian loan application, a reward signal, and ten RBI underwriting rules. Its job: approve, reject, request documents, or counter-offer."

**Delivery notes:** Slow on "without ever seeing a real loan." Hold eye contact after the question. The judges don't know where this is going — the hook buys you their attention for the next 60 seconds.

---

## The problem & the environment (45 seconds) — ends at **1:05**

> "LLMs are great at pattern-matching. A high-income applicant with low FOIR *looks* approvable — and the model happily approves.
>
> But real loan underwriting punishes vibes. An applicant with CIBIL 699 is one point below the hard cutoff. A perfect home loan profile with no RERA registration is an instant reject — that's a regulatory issue, not a credit issue.
>
> Our environment captures three loan types of escalating difficulty. Personal loans on easy — credit score, FOIR, employment. Vehicle loans add loan-to-value ratio. Home loans add tiered LTV under RBI rules, RERA compliance, and an employment threshold that differs from the other two.
>
> The agent gets observations in natural language, so an LLM can reason over them. It emits JSON actions. Multi-step episodes let it request documents and re-evaluate, or counter-offer and see the loan recalculated."

**Delivery notes:** "Vibes" is an audience wake-up word. Use it. When you say "tiered LTV," flash your hands to suggest hierarchy — it makes the regulation sound real.

---

## The innovation — self-improving loop (45 seconds) — ends at **1:50** ⭐ MAIN POINT

> "Here's what makes this interesting for Theme 4 — self-improvement.
>
> After each adversarial training round, we ask the model being trained to *design its own trap cases* — cases it knows are tricky, targeting the strategies it just failed at.
>
> Early rounds produce weak traps. The model doesn't yet know enough to be mean. But by round three it's generating borderline profiles — CIBIL 699 with ₹8 lakh income, home loans with LTV exactly at a tier boundary — cases rule-based generation would never think to create.
>
> Every self-generated case is verified against deterministic ground truth before it's used. The rule engine is the referee. The better the model gets, the harder it makes its own training. That's the recursive loop."

**Delivery notes:** This is the slide where you slow down. *"Early rounds produce weak traps"* — pause for half a beat. The surprise is that the model improves the environment, not the other way around.

---

## The results & the discovery (50 seconds) — ends at **2:40**

> "We trained Qwen 2.5 1.5B on a T4 using GRPO. Personal Loan accuracy went from 57% to 86% — a 29 point improvement. Home Loan accuracy went up 17 points.
>
> Vehicle Loans? They got *worse* by 14 points.
>
> And that's the most interesting result. A flat reward signal made the model over-optimize for Personal Loans while forgetting Vehicle patterns. The environment *revealed its own training gap*.
>
> That's exactly why we layered curriculum learning — easy, medium, hard, gated on measured accuracy — and adversarial self-play on top. They're not part of the environment; they're what the environment *made necessary*.
>
> Everything's open. The environment is live on Hugging Face Spaces. The training script runs in Colab. Sixty-three regression tests guard the reward logic. The OpenEnv validator passes three-for-three."

**Delivery notes:** Lean into the regression. Most pitches hide bad numbers. Judges remember the one that led with them. *"The environment revealed its own training gap"* — that sentence is doing the heavy lifting. Land it.

---

## Close (20 seconds) — ends at **3:00**

> "Primary theme: Self-Improvement. Secondary: Professional Tasks with real regulatory constraints. Sub-theme claim: Snorkel — the model acts as a simulated expert proposing harder cases each round.
>
> Links are in the README. Happy to take questions."

**Delivery notes:** If you're behind time by >15s, skip "Snorkel — the model acts as..." and just say "Snorkel sub-theme." If you're ahead, don't fill. Early finish is better than overshoot.

---

## Timing card (print or memorize)

| Section | Cumulative | Word count |
|---|---|---|
| Opener | 0:20 | 48 |
| Problem + environment | 1:05 | 136 |
| Self-improvement loop | 1:50 | 108 |
| Results + discovery | 2:40 | 132 |
| Close | 3:00 | 44 |
| **Total** | **3:00** | **~468 words** |

468 words ÷ 180 seconds = **156 wpm** — right in the "natural but confident" speaking range.

---

## Q&A Preparation (2 minutes)

Judges will likely probe one of these. Have a 30-second answer ready for each.

### Q: "How do you prove the self-generation loop actually works, not just that it runs?"

> "Three signals. One — we track per-strategy failure rates across rounds in `AdversarialTracker`; if the target strategy's failure rate drops after training on it, the loop is working. Two — we verify every self-generated case against deterministic ground truth, so invalid cases are discarded before training. Three — we cap self-generated cases at 30% of each batch to prevent distribution collapse. Actually publishing the per-round accuracy chart is on our roadmap for tomorrow's retrain."

### Q: "Couldn't the model just output 'reject' for everything and get a decent reward?"

> "Rejecting everything tops out at minus five per step because good applicants exist. Approving everything is catastrophic — minus fifteen for a bad loan, minus twenty for a non-RERA home. And invalid JSON costs half a point of normalized reward. The asymmetry is the anti-hacking mechanism."

### Q: "Your overall +10% isn't huge. What's actually happening?"

> "Two things. One — the per-type numbers tell the real story: +29% on Personal Loans is a big shift. Two — on 1.5B with 300 samples, we're compute-limited. The environment is sized for 7B training with longer runs; our 1.5B result is a proof of signal. The vehicle regression also argues *against* claiming we're just overfitting — if we were, vehicle would be up too."

### Q: "Why not use a standard loan dataset like LendingClub or Kaggle?"

> "Two reasons. One — those datasets are approve/default labels, not underwriting decisions. There's no 'request_docs' action, no counter-offer, no RBI compliance check. Two — real datasets bake in historical bank bias. We wanted a ground truth tied to *published rules*, not to outcomes shaped by selection bias."

### Q: "What's stopping someone from just prompting GPT-4 and skipping training?"

> "GPT-4o-mini scores 0.83 on our benchmark — see the baseline table. Rule-based tops out at 1.0. The 17 percent gap between a strong prompted LLM and the deterministic rule engine is where this environment is useful as a training target — it specifically rewards closing that gap."

### Q: "How extensible is this to new loan types or jurisdictions?"

> "Four files: a generator that produces applicants, a ground truth function, a reward schedule, and a router registration. A business loan or education loan plugs in with zero changes to the core. Different jurisdiction — swap RBI rules for FCA or Fed rules in `ground_truth/*.py`."

### Q: "What about privacy? Real loan data?"

> "All applicants are synthetic — generated with `random` from parameter ranges anchored to public RBI guidelines and published bank eligibility pages. No real customer data, no PII, no data collection."

---

## If the pitch goes sideways

**If you blank:** Fall back to the one-sentence elevator version:

> "An OpenEnv environment where an LLM learns Indian loan underwriting through curriculum learning and adversarial self-play, with a self-generation loop where the model designs its own trap cases each round."

**If the mic cuts / projector fails:** Hold up your laptop, navigate to the HF Space, hit reset, show a live observation. Narrate in 45 seconds: here's an applicant, here are the four actions, here's the reward formula. That's a complete pitch by itself.

**If a judge is hostile or interrupts:** "Great question — can I finish this one point first?" is a complete sentence. Don't concede time until the 3 minutes are done.
