# Hearsay × Keyguard: one defense for a live call

[Keyguard](https://github.com/LordKarV/keyboard-acoustic-shield) (teammate's HackGT project) reads keystrokes from the
sound of a keyboard leaking into call audio, then defeats that attack with an adversarial shield that keeps speech
intact (PESQ, STOI and Whisper must hear no change). Hearsay decides whether a voice is a real human or synthetic.
Both defend the same channel, **the audio of a live call**, against the two ways it gets abused:

| threat on a call | who stops it |
|---|---|
| the voice asking you for something is a clone | Hearsay (inbound audio) |
| the password you type while talking leaks through your mic | Keyguard (outbound audio) |

The scam that needs both: a cloned voice of your manager asks you to "just type the reset code while we're on the
call". The 2024 Hong Kong deepfake video-conference fraud (≈ US$25M) is the documented version of the first half.

## Fusion ideas, ranked

1. **CallGuard: one audio layer, both directions.** Outbound mic → Keyguard shield. Inbound far-end audio → Hearsay,
   scored on a rolling 4 s window every 2 s, with a "voice authenticity" light in Keyguard's dashboard.
   Hook points (from Keyguard's code): `keyguard/realtime.py` already runs a 16 kHz `sd.Stream` with 20 ms blocks,
   and a ring buffer + off-thread scorer fits in its callback. `keyguard/server.py` (FastAPI) takes a
   `POST /api/authenticity` route or a `voice_authenticity` field in `/api/demo`. Both projects are 16 kHz mono.
   Demo: a consenting teammate's voice cloned with our open-source sim (`docs/synthetic_speech.md`) asks for a code;
   the Hearsay light goes red; the judge types; Keyguard's attacker reads the keys; the shield blinds it.
2. **Authenticity-preserving shield.** Keyguard's min-max has three "the speech must stay intact" critics
   (PESQ/STOI, Whisper, a human). Add Hearsay as a fourth: the shield may not make a real voice look synthetic.
   A shield that gets its own user flagged as a deepfake by the other side's detector fails the user, and the
   challenge brief prices that error (a real voice flagged) at 4×. E2 result: Keyguard's DSP shield pushes a classic
   detector's false alarms from 1 % to 17 %. Our submitted detector barely moves (0.7 % → 1.3 %, within noise). The
   constraint matters because you don't choose the detector on the other end of the call.
3. **Robustness certificate for Hearsay.** Typing while talking is the most common real-call nuisance. E1 measures
   Hearsay's false alarms against keystroke noise, using Keyguard's recorded press banks (≈ 5.7k presses over ~10 keyboard/mic
   domains; E1 draws from 5 of them). If it hurts, those banks become a training augmentation (not done: no training in this phase).
4. **Red-team the detector (stretch, needs GPU).** Keyguard's `optimize_perturbation` finds bounded (−18 dB)
   perturbations against a differentiable attacker. Point it at Hearsay's R4ft: can an inaudible perturbation carry a
   deepfake past the detector? Then harden. The same weapon is a shield in one project and an attack on the other.

## Experiments (inference only; `scripts/make_keyguard_sets.py`, `scripts/bench_score.py`)

600 clips from `test_internal_testlike` (300 real, 300 fake, seeded). The models never selected anything on this set.
- **E1 typing while talking:** Keyguard presses from 5 keyboard domains at 3 presses/s, scaled like Keyguard's
  `synth.mix` (press power vs whole-clip speech power) to 20 / 10 / 5 / 0 dB.
- **E2 shield side effects:** Keyguard's DSP shield (`Shield`, dashboard settings: inpainting + residue
  randomization + 6 decoys) on clean speech (it auto-detects onsets) and on the 10 dB typing mix (true key timestamps,
  as Keyguard runs it on the victim's side).
- Keyguard's neural separator and adversarial shield are not tested: their weights are not in the repo.

Numbers: `reports/generalization.md` §2.
