# Thesis V8 Final-Submission Review

Scope checked: included LaTeX source, current `thesis.pdf`, references/cross-references, included result tables, and selected external source spot checks for claim support. I did not edit the thesis source in this pass.

## Highest-Priority Issues

1. `thesis.pdf` is stale relative to the source.
   - `thesis.pdf` timestamp: Aug 4 19:06:14 2026.
   - Several source files are newer, including `chapters/abstract.tex` at Aug 4 19:06:24 2026.
   - The PDF abstract text does not match `chapters/abstract.tex:1`.
   - Local TeX tools (`pdflatex`, `latexmk`, `bibtex`) are not installed in this environment, so I could not rebuild here.

2. The behavior-cloning claim is technically inconsistent.
   - `chapters/methods.tex:202-206` describes partially observable policies learning from full-state rewards, which is not behavior cloning in the usual supervised imitation sense.
   - `Appendices/methodology.tex:77` says the student policy imitates the full-state teacher, which is closer to behavior cloning but conflicts with the methods text.
   - Fix by clarifying whether the KF-on policy is:
     - PPO trained with EKF observations and true-state rewards, or
     - a supervised student trained to imitate full-state teacher actions.

3. Player/reward notation is swapped or mislabeled in the methods chapter.
   - `chapters/environment.tex:7-10` defines Player 1 as defender and Player 2 as attacker.
   - `chapters/methods.tex:58` says the first EKF observation is for the attacker, but `o_{1,EKF}` uses agent 1 as the true self-state, which should be defender under the earlier convention.
   - `chapters/methods.tex:74` similarly calls `o_{2,EKF}` the defender observation, but it appears to be attacker.
   - `chapters/methods.tex:102` calls `d_1` attacker-to-center and `d_2` defender-to-center, but the equations use `p_1` and `p_2`; this conflicts with `Appendices/methodology.tex:23-24`.
   - `chapters/methods.tex:128-139` has the defender reward but label `eq:attacker_step_reward`; `chapters/methods.tex:176-187` has attacker reward but label `eq:defender_step_reward`.
   - `chapters/methods.tex:189-197` introduces attacker terminal rewards but still uses `$r_1$`; likely should be `$r_2$`.

4. Monte Carlo run-count totals are ambiguous.
   - `chapters/testing.tex:235` and `chapters/conclusion.tex:5` state 57,600 / 172,800 / 345,600 run totals.
   - In the tables, `Total n` is per matchup column. Under that interpretation:
     - one dynamics + one observability + both arenas + four matchups = 57,600;
     - three cases + one dynamics + one observability = 172,800;
     - three cases + two observability settings + one dynamics = 345,600;
     - three cases + two observability settings + two dynamics = 691,200.
   - Because the surrounding text says evaluations span both HCW and elliptic dynamics, define exactly which axes each total includes.

5. Inference-time claims conflict with included tables.
   - `chapters/testing.tex:255-257`, `chapters/testing.tex:261`, `chapters/introduction.tex:31`, and `chapters/conclusion.tex:16` report full-state `0.062--0.145 ms` and KF-on up to `3.91 ms`.
   - Included table values show full-state medians up to at least `0.1549 ms` in an IQR and KF-on medians up to `11.6603 ms` for one elliptic 20 m case: `Figures/Merged_Thesis/Training_Policy_0.1u_1vmax_0.05_icVmax/merged_thesis_tables.tex:241`.
   - Either qualify the reported values as a subset or update them from the included appendix tables.

## Typos And Grammar To Fix

- `chapters/environment.tex:139`: "Credits of the image of the  go..." is broken. Suggested: "Image credit: NASA, Reid Wiseman, Artemis II \cite{apod2026helloworld}."
- `chapters/literature_review.tex:12`: "researchers have found problems in which have been useful..." is ungrammatical.
- `chapters/methods.tex:90`: double space in "measured  via".
- `chapters/methods.tex:394`: missing space in "model.The".
- `chapters/methods.tex:498`: "adversaries action" should be "adversary's action"; also the sentence after "Therefore" needs tightening.
- `chapters/testing.tex:297`: "The trained policies  translates..." should be "The trained policies translate..." and "it was" should be "they were".
- `chapters/testing.tex:311`: `CFB-QP` should be `CBF-QP`.
- `chapters/testing.tex:315`: "terms currently present the reward function" should be "terms currently present in the reward function".
- `chapters/testing.tex:342`: "Some ideas for this include." should end with a colon.
- `chapters/testing.tex:350`: "This would allow to further test..." should be "This would allow further testing of..." or "This would allow us to further test...".
- `chapters/testing.tex:358`: "allow us to be a showcase how these RL can be used" needs rewriting.
- `chapters/testing.tex:369`: "debris fields that actively shifting" should be "debris fields that are actively shifting".
- `structural/body.tex:152`: "Masters of Science" should be "Master of Science" unless UT's required vita style says otherwise.

## Claims To Soften Or Support

- `chapters/testing.tex:5`: "statistically significant results" is too strong unless you define a hypothesis test. Safer: "statistically precise empirical estimates" or "low-standard-error estimates."
- `chapters/testing.tex:259`: "prove the competency" should be softened to "support the competence/performance" or "provide evidence for".
- `chapters/testing.tex:247` and `chapters/conclusion.tex:11`: "adapt to unmodeled dynamics" should be limited to "transfer to the evaluated elliptic LTV model under the tested actuation and arena conditions."
- `chapters/testing.tex:331`: "minimal performance loss" is not true for the 100 m KF-on cases; qualify it as applying mainly to the 20 m arena or to selected aggregate comparisons.
- `chapters/literature_review.tex:43`: "realistic constraints one would see in a real-world autonomous vehicle" overstates the assumptions. The thesis explicitly assumes perfect attitude pointing, fully actuated translation, no actuator lag, and measured opponent control, so use "more realistic constraints than the full-state baseline" or similar.
- `chapters/conclusion.tex:21`: "without needing to resort or fallback to classical optimization methods" is too broad, especially because the architecture uses a CBF-QP controller. Suggested: "without solving an online pursuit-evasion optimization at every step."

## Structural / Formatting Notes

- The abstract source is a single very long line. It works, but it makes line-based review hard; consider wrapping it to normal paragraph width.
- The main technical chapters use many contractions (`we'll`, `we're`, `don't`, `it's`). Fine for preface/acknowledgments, but for the technical body, expand them for final thesis tone.
- `Select_Sections/results.tex` figure captions reference tables that include both KF-off and KF-on within one table; the wording is understandable but slightly confusing because the "KF On" figure points to `...kf-off-vs-on` labels.
- `Appendices/methodology.tex:69` mentions a residual actor/prior blend, while `chapters/methods.tex:768` says residual networks were skipped. Reconcile or remove the stale appendix phrase.

## Cross-Reference / Citation Check

- Included manuscript graph: no undefined `\ref`s, no undefined citations, and no duplicate labels among included files.
- Non-included template/demo files do contain unresolved demo references, but they are not part of the compile path.
- Included manuscript graph contains no `N/A` table placeholders.

## External Spot Checks

- The APOD image credit supports "NASA, Reid Wiseman, Artemis II" for the 2026 April 4 "Hello World" image.
- The Drucker/Shaferman arXiv citation is current as arXiv:2606.17594, submitted June 16, 2026 and revised June 25, 2026.
