# Demo Video Script — Credit Assessment Environment

**Target length:** 1:45–2:00 (hard cap 2:00 per hackathon rules)
**Format:** screen-capture + voiceover. No talking head needed.
**Tools:** QuickTime / OBS / Loom. Export as 1080p MP4, upload to YouTube as **Unlisted**.

Every line below has a **word count** calibrated to ~155 words/minute (normal speech ≈150–160 wpm). Total script ≈290–300 words = 1:50–1:55.

---

## Shot list — what's on screen

| # | Duration | On-screen |
|---|---|---|
| A | 0:00–0:08 | README hero paragraph (*"Can an LLM learn to be a loan officer..."*) zoomed in |
| B | 0:08–0:22 | Split screen: applicant profile (`applicant_profile` text) on left, trap profile call-out on right |
| C | 0:22–0:38 | Reward asymmetry table (+10 / -5 / -15 / -20) highlighted with mouse |
| D | 0:38–0:55 | Zoom into `server/ground_truth/home_loan.py` RBI tiered LTV logic + RERA check |
| E | 0:55–1:15 | Scroll through the 10 adversarial strategies list + `AdversarialTracker.get_weakness()` |
| F | 1:15–1:35 | `assets/reward_curve.png` + `assets/per_task_accuracy.png` side-by-side (use tomorrow's retrained plots) |
| G | 1:35–1:55 | HF Space URL being typed into browser, then live `/reset` response showing |

---

## Voiceover script (deliver at 150–160 wpm)

### [Shot A] 0:00–0:08 — The hook (18 words)

> "Can a language model learn to be a loan officer — without ever seeing a real loan? That's what this environment asks."

### [Shot B] 0:08–0:22 — The problem (32 words)

> "The agent receives an Indian loan application. Its job: approve, reject, request documents, or counter-offer. Some cases are obvious. Others are traps — perfect financials hiding one disqualifying flaw an LLM would miss."

### [Shot C] 0:22–0:38 — The reward signal (38 words)

> "The reward is asymmetric, like real banking. Correct decisions pay +10. Wrongly rejecting a good borrower costs five. Approving a bad loan costs fifteen. Approving a non-RERA home loan — a regulatory violation — costs twenty."

### [Shot D] 0:38–0:55 — Why it's hard (42 words)

> "The rules are grounded in actual RBI guidelines — tiered loan-to-value limits based on loan size, RERA property registration, CIBIL thresholds at exactly seven hundred. To succeed, the model must compute LTV from raw property values, not pattern-match."

### [Shot E] 0:55–1:15 — The self-improvement loop (50 words)

> "After every adversarial round, we ask the model to design its own trap cases targeting its weakest skills. Those cases — verified against deterministic ground truth — feed into the next round. The better the model gets, the harder it makes its own training."

### [Shot F] 1:15–1:35 — Results (48 words)

> "Standard GRPO training on Qwen 2.5 lifts Personal Loan accuracy from fifty-seven to eighty-six percent. Home Loans improve seventeen points. But Vehicle Loans regress — and that regression is exactly why we built curriculum learning and adversarial self-play on top of the environment."

### [Shot G] 1:35–1:55 — The call to action (44 words)

> "Everything is open. The environment is live on Hugging Face Spaces. The training notebook runs in Colab. The reward logic is OpenEnv-compliant and inspectable. Clone it, train on it, or extend it to a new loan type in four files."

---

## Recording checklist

Before you hit record:

- [ ] README scrolled to `## Demo & Materials` section in one tab
- [ ] `server/ground_truth/home_loan.py` open in another tab, LTV tier block visible
- [ ] `train_utils.py` scrolled to `ADVERSARIAL_STRATEGIES` (line ~530)
- [ ] `assets/` folder open in finder to drag in training plots for shot F
- [ ] Browser tab with `https://iamnijin-credit-assessment-env.hf.space` ready
- [ ] Terminal with `curl -X POST https://iamnijin-credit-assessment-env.hf.space/reset` typed but not executed
- [ ] Notifications OFF (macOS: Focus → Do Not Disturb)
- [ ] Mic check: speak a test phrase, confirm levels
- [ ] Close Slack, Mail, Messages — especially if doing screen capture

## Recording tips

- Do **three full takes** back-to-back. Use the best one. Don't try to get it perfect in one.
- If you fluff a line, pause 2 seconds and restart that line — you can cut the flub in iMovie / QuickTime with zero audio glitch.
- Cursor should move **deliberately and slowly** during shots D and E — recording will compress motion; fast cursor movement looks jittery.
- For shot G, the live curl output is the closer. Make sure the terminal font is 18pt+ so it's legible on mobile viewers.

## Post-recording

1. Trim leading/trailing silence.
2. Export 1080p H.264 MP4.
3. Upload to YouTube, set **Unlisted** (not Private — judges must open via URL without being invited).
4. Title: `Credit Assessment Environment — OpenEnv Hackathon Submission`.
5. Description: one-paragraph summary + links to HF Space, Colab, GitHub repo.
6. Grab the URL and paste it into `README.md` `## Demo & Materials` table (replacing the `_TBD_` placeholder).
