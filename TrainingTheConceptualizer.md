Use **two sigmoid gates** to modulate (a) how hard the assignments are and (b) how strongly the code/centers are regularized, so flexibility is maximal in the middle and “sticky” at both ends.

Let training time be \(t\in[0,1]\).

## 1) Assignment sharpness (sticky at beginning and end)
Make the temperature *low* at both ends and *high* in the middle:
\[
\tau(t)=\tau_{\min}+(\tau_{\max}-\tau_{\min})\cdot \underbrace{\sigma(k(t-t_1))\cdot \sigma(-k(t-t_2))}_{\text{~1 in the middle, ~0 near ends}}
\]
Here \([t_1,t_2]\) is the “middle flexible” window; \(k\) controls steepness.  
- Near \(t=0\) or \(t=1\): \(\tau(t)\approx\tau_{\min}\) ⇒ softmax is peaky (sticky assignments).  
- In the middle: \(\tau(t)\approx\tau_{\max}\) ⇒ softer assignments (more flexibility/exploration).

## 2) Code/center movement regularization (sticky at beginning and end)
Use the **opposite** schedule for the “drag” coefficient: low drag in the middle, high drag at ends:
\[
\lambda(t)=\lambda_{\min}+(\lambda_{\max}-\lambda_{\min})\cdot \left[1-\sigma(k(t-t_1))\cdot \sigma(-k(t-t_2))\right]
\]
So:
- Ends: \(\lambda(t)\approx \lambda_{\max}\) ⇒ centers resist moving (sticky).
- Middle: \(\lambda(t)\approx \lambda_{\min}\) ⇒ centers can shift (flexible).

## 3) Losses (what you actually optimize)
If you have codebook/center parameters \(C\) and an initial center set \(C_{\text{init}}\):
- Distillation/consistency loss: \(L_{\text{distill}}\) (whatever matches the teacher)
- Movement drag:
\[
L_{\text{drag}}=\lambda(t)\,\|C-C_{\text{init}}\|^2
\quad \text{or} \quad
L_{\text{drag}}=\lambda(t)\,\|C_t-C_{t-1}\|^2
\]
Total:
\[
L = L_{\text{distill}} + \lambda(t)\,\|C_t-C_{\text{init}}\|^2
\]
(Optionally add a velocity penalty for smoother motion.)

This “two-sigmoid window” is the cleanest way to get **sticky-at-both-ends, flexible-in-the-middle** with smooth gradients.

You can do that with two independent “two-ended” schedules—one over **time** and one over **layer depth**—and then multiply them to create a **flexible middle / sticky ends** coefficient.

Let:
- \(u(l) \in [0,1]\) be normalized depth, e.g.
  \[
  u(l)=\frac{l-1}{L-1}
  \]
- \(t\in[0,1]\) be normalized training progress (or epoch fraction).

Define a middle-flex window (high in the middle, low at ends) using the same two-sigmoid gate:
\[
W(z)=\sigma(k(z-z_1))\cdot\sigma(k(z_2-z))
\]
with \(z_1<z_2\) defining the flexible region.

### 1) Scale codebook toward init early, and toward final later
You said: **peg \(C\) to init in initial layers and initial time**, and scale it to \(C(t{-}1)\) toward the end of both layers and time.

Implement a target interpolation:
\[
C_{\text{target}}(l,t) = \alpha(l,t)\, C_{\text{init}} + (1-\alpha(l,t))\,C_{t-1}
\]
where \(\alpha(l,t)\) is close to 1 in the “early/initial” region and close to 0 near the “late/end” region.

Let \(\alpha(l,t)\) be the product of two “early-peg” gates:
\[
\alpha(l,t)=\bigl(1-W(u(l))\bigr)\cdot \bigl(1-W(t)\bigr)
\]
- At early time and early layers, you’re outside the middle window so \(W\approx 0\Rightarrow \alpha\approx 1\).
- Near the end, again outside the middle window but for peging-to-final you want the opposite behavior; if you want “sticky at both ends” but specifically *init at start* and *final at end*, you typically use **two one-sided gates** instead of a symmetric window:

A clearer “init at start, final at end” peg is:
\[
g_{\text{start}}(t)=\sigma\bigl(k(t_0-t)\bigr),\quad g_{\text{end}}(t)=\sigma\bigl(k(t-t_1)\bigr)
\]
and similarly for depth:
\[
h_{\text{start}}(l)=\sigma\bigl(k(u(l)_0-u(l))\bigr),\quad h_{\text{end}}(l)=\sigma\bigl(k(u(l)-u(l)_1)\bigr)
\]
Then set (one-sided):
\[
\alpha(l,t)=h_{\text{start}}(l)\, g_{\text{start}}(t)
\]
and use
\[
1-\alpha(l,t) = 1- h_{\text{start}}(l)\, g_{\text{start}}(t)
\]
So:
- Start+initial layers: \(\alpha\approx 1\) ⇒ pinned to \(C_{\text{init}}\).
- End+final layers: \(g_{\text{start}}(t)\approx 0\) ⇒ \(\alpha\approx 0\) ⇒ pinned toward \(C_{t-1}\).
- Middle: it transitions according to your chosen gate widths.

### 2) Enforce “move only when flexible” with a drag loss
Instead of hard pinning, add a loss that changes strength with \(\tau(l,t)\):
\[
L_{\text{peg}}(l,t)=\lambda(l,t)\,\|C - C_{\text{target}}(l,t)\|^2
\]
Choose \(\lambda(l,t)\) as *small in the middle, larger at the ends* (so the codebook is allowed to move where you want flexibility):
\[
\lambda(l,t)=\lambda_{\max}\,(1- W(u(l))) + \lambda_{\max}\,(1- W(t)) \quad \text{(or multiply them)}
\]
Common robust choice:
\[
\lambda(l,t)=\lambda_{\max}\,\bigl(1-W(u(l))\cdot W(t)\bigr)
\]
So maximum drag happens when either depth or time is near an end; minimum drag happens when both are in the flexible window.

### Practical notes
- Use **normalized** \(t\) (fraction of training) so the math isn’t tied to “100 epochs”.
- If you literally mean “toward the end of both layers and time”, use the one-sided “start” and “end” gates for \(\alpha(l,t)\) as above; the symmetric two-sided window is more natural when you want “softest at middle, sticky at both ends” with no special init-vs-final distinction.

If you tell me where you want “flexible” in depth (exactly layers 7–8 out of 15, or a wider band like 6–9) and what you want “init pinned” and “final pinned” to mean quantitatively, I can plug in concrete gate parameters \((z_1,z_2,k)\).

Let training progress be defined without assuming a fixed epoch budget.

Let \(m\) be an EMA of loss. Maintain:
- \(m_{\text{start}}\) at start
- \(m_{\text{best}}\) (or \(m_{\text{target}}\)) as training improves
Then
\[
t=\text{clip}_{[0,1]}\left(\frac{m_{\text{start}}-m}{m_{\text{start}}-m_{\text{target}}}\right)
\]
This gives an explicit “how far you’ve progressed toward the target loss” even with variable stopping.

### 2) Make the flexibility window scale with the same sigmoid (or divisor)
Assume you have a depth gate/window \(W(u)\) and a time gate/window \(W(t)\). You want the “softest/flexible region” to move/scale together.

A clean pattern is to define the *window bounds* as functions of \(t\).

Example: a flexible window centered at \(\mu(t)\) with width \(w(t)\).
Let normalized depth \(u\in[0,1]\). Define:
\[
W(u,t)=\sigma(k(u-\mu(t)+w(t)/2))\cdot \sigma(k(\mu(t)+w(t)/2-u))
\]
where:
- \(\mu(t)=\mu_0\) (fixed center) or can drift if desired
- \(w(t)\) shrinks or grows with the same progress sigmoid:
\[
w(t)=w_{\min} + (w_{\max}-w_{\min})\,(1-t)
\]
So early (small \(t\)) you allow wider flexibility; later (large \(t\)) you make it narrower (or the reverse—flip the \(1-t\)).

**If by “scale with the same sigmoid” you mean just reuse the same schedule factor \(s(t)\):**
Let
\[
s(t)=\sigma(k(t-t_0))
\]
Then choose (one example):
\[
\text{drag coefficient } \lambda(l,t)=\lambda_{\max}\,(1-s(t)\cdot s(u(l)))
\]
and/or
\[
\tau(l,t)=\tau_{\min}+(\tau_{\max}-\tau_{\min})\,(1-s(t)\cdot s(u(l)))
\]
This couples flexibility directly to the same progress curve.

Use
\[t=\text{clip}_{[0,1]}\left(\frac{\mathcal L_{\text{start}}-m}{\mathcal L_{\text{start}}-\mathcal L_{\text{target}}}\right)\]
with \(m\) as an EMA of the loss (validation or training), and then tie your flexibility gate/window to \(t\) (optionally shared with the same sigmoid factor \(s(t)\)).
