# Teaching a Language Model to Be a Loan Officer

*A story about RBI rulebooks, CIBIL 699, the −20 reward that ate my first agent, and what happens when you stop treating reasoning as a dataset and start treating it as a job.*

---

## The moment that started it

A few months ago I was poking at Qwen2.5-7B with a synthetic loan profile I'd hand-written to break it. CIBIL score: 699. Income: high. FOIR: comfortable. Employment: stable. Documents: complete. Everything green except that one number, sitting one point under the cutoff every Indian bank uses.

The model approved it.

I tried again with the score nudged to 700. Approved. 701. Approved. 698. *Approved again.*

It wasn't reading the threshold. It was pattern-matching on the overall vibe of the application — high income, clean record, plausible borrower — and ignoring the rule that, in real life, gets a junior analyst a phone call from compliance. A human loan officer with one year of training would not make that mistake. A 7B parameter model with no banking background obviously did, and I started wondering what it would take to teach it the way you'd teach a person: by giving it cases, letting it decide, and letting consequences shape the next decision.

That's how `Credit_Assessment_Env` started.

---

## Why this is a real problem, not just a toy one

India's banking system carries roughly ₹4 lakh crore in non-performing assets at any given time. A meaningful share of that is loans that should never have been written — borderline applications waved through because the underwriter had a quota to hit, or a bad-but-confident-looking borrower, or a builder whose RERA registration had quietly lapsed. Every approved loan that defaults isn't just a number on a balance sheet; it's a provisioning hit, a regulatory filing, and (if the loan is large enough) a phone call between the bank and the RBI.

The rules that prevent these mistakes are not secret. They're sitting in the [RBI Master Circular on Housing Finance](https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=6161&Mode=0), in HDFC's eligibility pages, in every NBFC's training manual. CIBIL ≥ 700. FOIR ≤ 50%. Vehicle LTV ≤ 85%. Home LTV tiered: 90% under ₹30L, 80% from ₹30–75L, 75% above ₹75L. Two years' employment for a home loan, one year for the others. RERA registration mandatory or you don't lend. The rules are unglamorous and there are not many of them.

What's hard isn't memorising them. What's hard is *applying them under noise*: an applicant whose income looks great until you realise the FOIR is computed off the primary holder; a ₹1.2Cr home loan with an LTV that looks "safe" at 78% but is actually above the tier-3 cap; a perfect profile with one missing document where the right answer isn't approve or reject, it's "request docs." Real underwriting is a small, well-defined rulebook applied to a large, messy stream of borderline cases. Which is exactly the shape of problem RL is good at — provided you can build the simulator.

So that's what I built.

---

## Why an environment, and not just a dataset

I could have generated 50,000 (profile, decision) pairs and fine-tuned a model on them. People do this all the time. It would have worked, sort of. The model would learn to map inputs to outputs and would probably hit 90%+ on the easy half of the distribution.

But supervised data has a problem: it doesn't teach the *consequence* of being wrong. A model trained on labels treats "approve a CIBIL 699 applicant" and "approve a non-RERA home loan" as roughly the same kind of mistake — they both add the same amount to a cross-entropy loss. In the real world they aren't the same mistake at all. Approving a borderline borrower is a calculated risk that occasionally goes bad. Approving a non-RERA home loan is a regulatory liability that ends careers.

An environment lets you encode that. The reward function in this env reflects the asymmetric cost of underwriting:

- Correct decision: **+10**
- Reject a good applicant (lost revenue, recoverable): **−5**
- Approve a bad applicant (NPA risk): **−15**
- **Approve a non-RERA home loan: −20**
- Counter-offer that catches a high LTV (risk mitigation done right): **+5**
- Request documents when documents are missing (correct procedural step): **+2**

Three times worse to approve than to reject. Four times worse to miss a RERA breach than to be a bit too conservative. The agent doesn't have to be told these things; the gradient teaches them. And it can't be gamed — rejecting everything caps you around −5 average (you lose money on every good case), approving everything blows up at −20 the first time a non-RERA case shows up. The only profit-maximising strategy is to actually read the file and apply the rules.

That asymmetry is the whole reason this is an RL problem and not a classification one.

---

## Designing the environment

The env exposes three tasks of escalating difficulty, all served from a FastAPI box that speaks the [OpenEnv](https://github.com/facebookresearch/openenv) protocol:

| Task | Loan Type | Difficulty | What the agent has to reason about |
|------|-----------|------------|-------------------------------------|
| 1 | Personal | Easy | CIBIL ≥ 700, FOIR ≤ 50%, 1y employment, docs complete |
| 2 | Vehicle | Medium | All of the above + LTV ≤ 85% (counter-offer if exceeded) |
| 3 | Home | Hard | All of the above + RBI tiered LTV + RERA + 2y employment |

Each `reset()` produces a fresh applicant. The applicant's profile is rendered as **narrative text** — the way an actual loan file reads — plus structured fields for the agent that wants them. The action space is four things a real loan officer can do: `approve`, `reject`, `request_docs`, `counter_offer`. The last two trigger multi-step episodes: ask for documents, the applicant comes back with them, you decide again. Counter-offer at a smaller amount, the LTV recomputes, you decide again.

The interesting part of the design isn't the rules. It's the **trap profiles**. There are ten of them, hand-built to target the exact failure modes that RBI rules create:

- **Threshold credit (CIBIL 699)** — the case I started this post with. One point below the cutoff, everything else perfect. Pattern-matching screams approve. The rules say reject.
- **Perfect-but-RERA** — 830 CIBIL, ₹2L income, 20% FOIR, RERA = No. The −20 reward case. The single worst thing you can do in the environment.
- **LTV tier mismatch** — a home loan above ₹75L (e.g. a ~₹95L loan against a ₹1.2Cr property) at LTV 78%. Looks safe under an 80% cap; RBI caps loans above ₹75L at 75%, so it's a counter-offer.
- **Vehicle LTV trap** — perfect profile, LTV 86%. One point over the 85% cap. Counter-offer, not approve.
- **Co-applicant mirage** — a co-applicant is on the file, but FOIR is computed off the primary borrower. Adding a name doesn't reduce the income test.

These aren't randomly hard cases. They're each designed to break a specific intuition the model would otherwise rely on. If your agent gets them right, it's because it's reading the rules, not vibing the application.

---

## What actually happened during training

The first training run failed in an interesting way and I want to talk about it because it's the moment the rest of the project's design fell into place.

I started with vanilla GRPO on a mixed-difficulty batch — Personal, Vehicle, Home all in the same buffer — using a smaller Qwen2.5-1.5B for speed. Overall accuracy went up by about 10 points. I was pleased. Then I broke it down per task and saw that **Vehicle accuracy had regressed by 14 points**.

What had happened: the high-density CIBIL/FOIR/RERA patterns from Personal and Home loans dominated the gradient. The model got better at those, then "forgot" the LTV nuance that Vehicle loans depend on. Catastrophic forgetting in miniature, exactly the failure mode curriculum learning was invented to prevent.

So I rebuilt the pipeline as three things in series:

1. **SFT warmup** — 600 supervised examples across all three loan types, 2 epochs. Not to teach the rules; to anchor the output format (chain-of-thought reasoning followed by JSON) so GRPO could spend its compute on judgment instead of syntax.
2. **Per-task curriculum with replay buffer** — Personal first, then Vehicle, then Home, 400 samples per phase, with a 20% replay slice from earlier phases mixed into later ones. Each phase has a 60% mastery threshold; you don't advance until you clear it.
3. **One adversarial round** — 50 GRPO steps trained exclusively on the 10 trap profiles, starting from the curriculum adapter, with a low LR (5e-7) and a strong KL anchor (β=0.4) to the curriculum reference so the model couldn't drift while it patched its weak spots.

The curriculum cleared every mastery gate on the first try — Personal 100%, Vehicle 98%, Home 92% on the held-out per-phase slices. The replay buffer did its job: Vehicle didn't regress when Home was introduced. The adversarial round on top moved Home Loan accuracy from 87.5% to 90% on the held-out n=120 slice (one extra correct out of 40, on the hardest task), with **zero regression anywhere else**.

The headline numbers, on a fair head-to-head where the same applicant pool and the same lenient JSON parser are used for both models:

| Loan Type | Baseline | Trained | Δ |
|---|---|---|---|
| Personal | 95.0% | 100% | +5.0pp (ceiling) |
| Vehicle  | 62.5% | **92.5%** | **+30.0pp** (CIs don't overlap) |
| Home     | 85.0% | 90.0% | +5.0pp |
| **Overall (n=120)** | **80.8%** | **94.2%** | **+13.3pp** (CIs don't overlap) |

Vehicle is the win that mattered. The base model didn't really know what to do with LTV — it could read the number but couldn't translate "86% LTV with everything else perfect" into "counter-offer at a lower amount." The trained model can. The +30 points on that task isn't a sampling artifact; the Wilson 95% confidence intervals on baseline and trained literally do not overlap.

What's also worth saying: the trained model **strictly does not regress** on any task. That property — never make the model worse at anything it was already good at — is what the curriculum + replay design was specifically built to preserve, and it held.

---

## A few things I noticed along the way

A handful of small observations that didn't fit neatly into the README but are worth writing down:

- **The lenient JSON parser is doing real work.** Qwen-Instruct base models default to wrapping their answer in ` ```json ` fences. The trained model emits raw JSON. A strict parser would silently mark the base model's *correct* answers as wrong and inflate the training delta by several points. Using the same lenient parser for both sides is one of the single most important fairness levers in the eval, and almost nobody talks about it.

- **Reasoning before JSON helps.** I tried "JSON-only, no thinking" prompts during the SFT phase. They hit a ceiling around 88%. The chain-of-thought-then-JSON format pushes past 94% with the same training budget. The reasoning isn't decorative; it's the place where the model walks through "FOIR is 0.42, that's under 0.50, fine; LTV is 0.86, that's above 0.85, so this is counter-offer not approve" before committing.

- **The adversarial round didn't help the easy tasks at all.** Personal stayed at 100%, Vehicle stayed at 92.5%. The whole gain landed on Home Loan, which is the task with the densest rule interaction (RERA + tiered LTV + employment + everything else). That's a useful signal for where future adversarial rounds should focus.

- **A 60% mastery gate sounds low until you watch a model fail it.** Vehicle Loans at the start of phase 2 were sitting around 55%. Without the gate, the next phase would have started from a weaker policy and Home Loan training would have been compromised. The gate is doing real safety work even when the average run sails through it.

---

## Where this goes next

The whole point of structuring the env the way I did was to make extension easy. Adding a new loan type — business, education, gold, agricultural — is genuinely a four-file change:

1. `server/generators/<loan>_loan.py` — applicant generator (good / bad / borderline / trap profiles)
2. `server/ground_truth/<loan>_loan.py` — the underwriting rules
3. `server/rewards/<loan>_loan.py` — the reward shaping (what counts as a catastrophic miss for *this* loan type)
4. Register it in `__init__.py` and add a task entry in `credit_assessment_env_environment.py`

Nothing in `models.py`, `client.py`, or the Dockerfile has to change. The action space is already general (`approve` / `reject` / `request_docs` / `counter_offer` cover almost any unsecured or secured product). The narrative profile builder is already loan-type-agnostic. The training pipeline doesn't care how many phases you have — adding a fourth phase for business loans is a config line.

Concretely, the things I'd love to see built on top of this:

- **Business loans** — DSCR (Debt Service Coverage Ratio), GST return consistency, working-capital vs term-loan distinction, MSME priority sector tagging. The rule density is comparable to home loans and the trap profiles practically design themselves (consistent revenue but seasonal cash flow, profitable on paper but high promoter withdrawals, etc.).
- **Education loans** — collateral requirements that depend on the loan amount (uncollateralised below ₹4L, partial above ₹7.5L), course recognition checks, co-applicant employment as the primary income test. A genuinely different reasoning pattern.
- **Gold loans** — purity verification, LTV against gold rate (currently capped at 75% per RBI), short tenure economics. Simpler rule surface, interesting because the "collateral value" itself is a market-priced variable.
- **Agricultural loans** — the priority sector lending requirements, KCC (Kisan Credit Card) eligibility, land record checks. Heavier on procedural correctness than judgment.
- **Frontier model benchmarks** — `scripts/fair_eval.py` is set up to run any LLM through the same n=120 slice. Plugging in GPT-4o, Claude, and Gemini with the matched chain-of-thought prompt would give the first apples-to-apples view of where a 7B specialist sits relative to frontier generalists on a domain like this. My priors: the gap is much smaller than people assume, and on Vehicle Loans the trained 7B might be ahead.
- **More adversarial rounds.** Two rounds are now shipped (see the Postscript below for what each one moved). The `AdversarialTracker` in `train_utils.py` already drives this loop — re-weighting each round toward whatever trap class the current policy is failing most. After 2 rounds, two trap classes still resist (`borderline_multiple` at 0/5, residual `perfect_but_ltv_tier` at 60%); a 3rd or 4th round, possibly with a different reward shaping or a longer KL schedule, is the natural next experiment.
- **Self-generated challenges** — the most interesting frontier. Prompt the trained model to *design* new trap cases ("write a profile that should be a counter_offer but looks like an approve"), verify each generated case against the deterministic ground truth, then feed the validated ones back into training. The env is set up for this — ground truth is a pure function of the applicant dict — but I haven't run it yet.
- **A real partner bank.** This is the obvious one. Everything in the env is synthetic, but the rules are real and the reward asymmetry is real. Running the trained model against a slice of historical (anonymised) loan applications from an actual NBFC and comparing its decisions to the ones a human officer made would be the first grown-up evaluation. I have not done this yet. If you work at one and want to talk, my email is in the repo.

---

## Postscript: a second adversarial round, run live on HF Jobs

For the hackathon onsite, I re-ran the full pipeline end-to-end on a Hugging Face Job (single L40S, ~5 hours, ~$11 of credits) and bumped the adversarial training from 1 round to 2 — partly to test the "AdversarialTracker re-targets weakness each round" claim with actual data, and partly because the full transcript would be reviewable by judges in real time.

What happened was the cleanest demonstration of the self-improvement loop the project has produced so far.

A note on small-sample noise before the numbers: each adversarial round's per-trap-class eval draws a *fresh* random sample of 5 cases per trap class (50 total). Across two snapshots of the same trap class on the same model you can easily see 0/5 in one and 2/5 in another — that's binomial sampling at n=5, not a model change. So when I quote per-trap numbers below I'm citing what the live diagnostic printed at the moment, and treating them as directional rather than precise.

After the curriculum finished, the tracker probed the 10 trap profiles and identified `perfect_but_ltv_tier` as the worst — **0/5 on that probe**. Round 1 trained against it (150 targeted samples + 111 replay). When the post-round diagnostic ran, two things happened worth separating:

- The *targeted* trap was still at 0/5. Round 1's training reward saturated at 0.84 with std≈0, which means the gradient signal flattened — the model overfit its own training distribution and didn't generalise to a fresh n=5 sample of `perfect_but_ltv_tier` cases. Honest read: the round didn't move the needle on what it was *trying* to fix.
- A *different* trap, `perfect_but_rera`, jumped from 4/5 → 5/5. Surprise transfer: the LTV-tier training nudged the model's general home-loan reasoning enough that an unrelated failure mode quietly fixed itself.

Round 2 looked at the new state, saw `perfect_but_ltv_tier` was *still* the worst trap (0/5 on the new probe), and re-targeted it — this time with 195 targeted samples + 111 replay + 45 self-generated cases from Round 1. Reward variance held at std≈0.20 instead of 0, which meant the gradient was actually informative, and the post-round diagnostic came back at **2/5 on the targeted trap, vs 0/5 going in**. Small n, but the *direction* finally moved. (The aggregate adversarial-eval accuracy across all 10 trap classes also climbed from **80% after Round 1 to 84% after Round 2**, recorded in [`training_log.json`](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/blob/main/run-20260425-105001/training_log.json).)

The honest scorecard after 2 rounds: 7 of 10 trap classes consistently mastered, 2 hard (`perfect_but_ltv_tier` and `borderline_multiple` still bouncing around 0–40% across snapshots), 1 lifted via transfer (`perfect_but_rera`). And the general loan tasks held — Personal 100%, Vehicle 90%, Home 87% on the script's internal eval — while the proper fair-eval at n=120 against the cold base model came in at **95.0% overall vs 81.7% baseline (+13.3pp, Wilson 95% CIs do not overlap)**. The +13.3pp delta matches the Colab number to the decimal point.

Two things this independent rerun proves that the original Colab run alone couldn't:

1. **The +13.3pp gain reproduces.** Different hardware (L40S vs A100), different training-time random seeds, one extra adversarial round — and the same statistically significant overall delta on the same fair-eval slice. That's much stronger evidence than a single run.
2. **The AdversarialTracker actually drives self-improvement between rounds.** Round 1's transfer to `perfect_but_rera` and Round 2's targeted lift on `perfect_but_ltv_tier` are concrete, traceable instances of the loop doing its job. Not all rounds will move every targeted trap (R1 didn't), but the *system as a whole* keeps the model improving where it has the most to learn, without regressing where it was already good.

The full transcript, the per-step reward curves, the per-strategy adversarial accuracy across both rounds, and the fair-eval JSON are committed to the [run-20260425-105001 folder](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/tree/main/run-20260425-105001) for anyone who wants to audit. The trained adapter is at [`iamnijin/credit-assessment-onsite-adversarial`](https://huggingface.co/iamnijin/credit-assessment-onsite-adversarial).

If you've read this far, the only thing I'd ask you to take from the postscript is this: when adversarial RL works, it doesn't always look like the round you targeted got better. Sometimes the round you targeted plateaus and a *different* failure mode quietly fixes itself, and you only catch it because you measured all 10 strategies, not just the one you intended to train on. Whatever you build next, measure more than the thing you optimised for.

---

## What I think this project is actually about

Most of the discussion about LLMs in finance is about chatbots — "ask your bank a question, get an answer." That's fine. But it's not where the value is. The value is in the long, unglamorous tail of decisions that real banks make tens of thousands of times a day, where the rules are clear, the cost of error is asymmetric, and a competent intermediate analyst is the bottleneck.

A 7B parameter model can be that competent intermediate analyst, on a single task, after about four hours of training on a single A100 — or roughly five hours and ~$11 of credits on a single L40S via Hugging Face Jobs, which is what the live re-run for the hackathon used. Not because the model is magic, but because the *environment* is honest about what the job actually is: read the file, apply the rules, accept that approving the wrong loan costs more than rejecting the right one, and don't approve a non-RERA home loan, ever.

The environment is public ([HF Space](https://huggingface.co/spaces/iamnijin/credit-assessment-env)). Two trained adapters are public — the original Colab one ([`iamnijin/credit-assessment-adversarial`](https://huggingface.co/iamnijin/credit-assessment-adversarial), 1 adversarial round, 94.2% on n=120) and the onsite HF Jobs reproduction ([`iamnijin/credit-assessment-onsite-adversarial`](https://huggingface.co/iamnijin/credit-assessment-onsite-adversarial), 2 adversarial rounds, 95.0% on the same n=120, with the full pipeline transcript and plots committed to a [public dataset run folder](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/tree/main/run-20260425-105001) for verification). The Colab notebook will reproduce the whole thing on a free T4 in an afternoon. If any of this is useful to you — for a different domain, a different rulebook, a different country's regulations — fork it and tell me what you build.

The interesting work in the next few years isn't going to be bigger models. It's going to be better environments.
