# Vector Notation Checklist for `MS_Thesis_Final_V2.pdf`

Page numbers below now match the original extracted PDF page numbers again.

For copy-paste consistency, I used:
- Latin vectors: `\vec{\mathbf{x}}`
- Greek vectors: `\vec{\boldsymbol{\rho}}`

Important: every `Copy/paste` snippet in this report is a math expression, so it must be placed inside math mode in LaTeX.

Use one of these forms:

```latex
$\dot{\vec{\mathbf{s}}} = F\vec{\mathbf{s}} + G\vec{\mathbf{a}}$
```

or

```latex
\[
\dot{\vec{\mathbf{s}}} = F\vec{\mathbf{s}} + G\vec{\mathbf{a}}
\]
```

If you paste `\dot{...}` or `\vec{...}` directly into normal paragraph text without `$...$` or `\[...\]`, LaTeX will throw the `allowed only in math mode` error.

If you want, you can also define shortcuts in your preamble:

```latex
\newcommand{\vect}[1]{\vec{\mathbf{#1}}}
\newcommand{\gvect}[1]{\vec{\boldsymbol{#1}}}
```

Then `\vect{x}` gives `\vec{\mathbf{x}}` and `\gvect{\rho}` gives `\vec{\boldsymbol{\rho}}`.

## Main Checklist

### Page 23
- As written: `a = (a_x, a_y, a_z)^\top`
- Copy/paste: `\vec{\mathbf{a}} = (a_x, a_y, a_z)^\top`

- As written: `s := [x\ y\ z\ \dot{x}\ \dot{y}\ \dot{z}]^\top`
- Copy/paste: `\vec{\mathbf{s}} := [x\ y\ z\ \dot{x}\ \dot{y}\ \dot{z}]^\top`

- As written: `\dot{s} = F s + G a`
- Copy/paste: `\dot{\vec{\mathbf{s}}} = F\vec{\mathbf{s}} + G\vec{\mathbf{a}}`

### Page 24
- As written: `s_{k+1} = A_d s_k + B_d u_k`
- Copy/paste: `\vec{\mathbf{s}}_{k+1} = A_d\vec{\mathbf{s}}_k + B_d\vec{\mathbf{u}}_k`

### Page 25
- As written: `r_{c,p}`
- Copy/paste: `\vec{\mathbf{r}}_{c,p}`

- As written: `v_{c,p}`
- Copy/paste: `\vec{\mathbf{v}}_{c,p}`

- As written: `r_{c,k} = Qr_{c,p}`
- Copy/paste: `\vec{\mathbf{r}}_{c,k} = Q\vec{\mathbf{r}}_{c,p}`

- As written: `v_{c,k} = Qv_{c,p}`
- Copy/paste: `\vec{\mathbf{v}}_{c,k} = Q\vec{\mathbf{v}}_{c,p}`

- As written: `\hat{R}_k`
- Copy/paste: `\hat{\vec{\mathbf{R}}}_k`

- As written: `\hat{N}_k`
- Copy/paste: `\hat{\vec{\mathbf{N}}}_k`

- As written: `\hat{T}_k`
- Copy/paste: `\hat{\vec{\mathbf{T}}}_k`

- As written: `C_{RTN \to I,k} = [\hat{R}_k\ \hat{T}_k\ \hat{N}_k]`
- Copy/paste: `C_{RTN \to I,k} = [\hat{\vec{\mathbf{R}}}_k\ \hat{\vec{\mathbf{T}}}_k\ \hat{\vec{\mathbf{N}}}_k]`

### Page 26
- As written: `\omega_k`
- Copy/paste: `\vec{\boldsymbol{\omega}}_k`

- As written: `\rho`
- Copy/paste: `\vec{\boldsymbol{\rho}}`

- As written: `\dot{\rho}`
- Copy/paste: `\dot{\vec{\boldsymbol{\rho}}}`

- As written: `s := [\rho_R\ \rho_T\ \rho_N\ \dot{\rho}_R\ \dot{\rho}_T\ \dot{\rho}_N]^\top`
- Copy/paste: `\vec{\mathbf{s}} := [\rho_R\ \rho_T\ \rho_N\ \dot{\rho}_R\ \dot{\rho}_T\ \dot{\rho}_N]^\top`

- As written: `u_k = (u_R, u_T, u_N)^\top`
- Copy/paste: `\vec{\mathbf{u}}_k = (u_R, u_T, u_N)^\top`

- As written: `r_{d,k} = r_{c,k} + C_{RTN \to I,k}\rho_k`
- Copy/paste: `\vec{\mathbf{r}}_{d,k} = \vec{\mathbf{r}}_{c,k} + C_{RTN \to I,k}\vec{\boldsymbol{\rho}}_k`

- As written: `v_{d,k} = v_{c,k} + C_{RTN \to I,k}(\dot{\rho}_k + \omega_k \times \rho_k)`
- Copy/paste: `\vec{\mathbf{v}}_{d,k} = \vec{\mathbf{v}}_{c,k} + C_{RTN \to I,k}(\dot{\vec{\boldsymbol{\rho}}}_k + \vec{\boldsymbol{\omega}}_k \times \vec{\boldsymbol{\rho}}_k)`

- As written: `\dot{r}_d = v_d`
- Copy/paste: `\dot{\vec{\mathbf{r}}}_d = \vec{\mathbf{v}}_d`

- As written: `\dot{v}_d = -\mu r_d / \|r_d\|^3 + C_{RTN \to I,k}u_k`
- Copy/paste: `\dot{\vec{\mathbf{v}}}_d = -\frac{\mu \vec{\mathbf{r}}_d}{\|\vec{\mathbf{r}}_d\|^3} + C_{RTN \to I,k}\vec{\mathbf{u}}_k`

- As written: `\rho_{k+1}`
- Copy/paste: `\vec{\boldsymbol{\rho}}_{k+1}`

- As written: `\dot{\rho}_{k+1}`
- Copy/paste: `\dot{\vec{\boldsymbol{\rho}}}_{k+1}`

- As written: `s_{k+1} = f_k(s_k, u_k)`
- Copy/paste: `\vec{\mathbf{s}}_{k+1} = f_k(\vec{\mathbf{s}}_k, \vec{\mathbf{u}}_k)`

### Page 27
- As written: `s_k = 0`
- Copy/paste: `\vec{\mathbf{s}}_k = 0`

- As written: `u_k = 0`
- Copy/paste: `\vec{\mathbf{u}}_k = 0`

- As written: `s_{k+1} = A_k s_k + B_k u_k`
- Copy/paste: `\vec{\mathbf{s}}_{k+1} = A_k\vec{\mathbf{s}}_k + B_k\vec{\mathbf{u}}_k`

### Page 29
- As written: `\hat{x}^{D \to A}_0`
- Copy/paste: `\hat{\vec{\mathbf{x}}}^{D \to A}_0`

- As written: `\hat{x}^{A \to D}_0`
- Copy/paste: `\hat{\vec{\mathbf{x}}}^{A \to D}_0`

- As written: `d^{i \to j}_t = p^j_t - p^i_t`
- Copy/paste: `\vec{\mathbf{d}}^{i \to j}_t = \vec{\mathbf{p}}^j_t - \vec{\mathbf{p}}^i_t`

- As written: `b^{i \to j}_t = R^i_{wb} d^{i \to j}_t / \|d^{i \to j}_t\|^2`
- Copy/paste: `\vec{\mathbf{b}}^{i \to j}_t = \frac{R^i_{wb}\vec{\mathbf{d}}^{i \to j}_t}{\|\vec{\mathbf{d}}^{i \to j}_t\|^2}`

- As written: `u^D_t`
- Copy/paste: `\vec{\mathbf{u}}^D_t`

- As written: `u^A_t`
- Copy/paste: `\vec{\mathbf{u}}^A_t`

- As written: `\tilde{u}^j_t`
- Copy/paste: `\tilde{\vec{\mathbf{u}}}^j_t`

- As written: `z^{i \to j}_t`
- Copy/paste: `\vec{\mathbf{z}}^{i \to j}_t`

- As written: `y^{i \to j}_t = z^{i \to j}_t - \hat{z}^{i \to j}_{t|t-1}`
- Copy/paste: `\vec{\mathbf{y}}^{i \to j}_t = \vec{\mathbf{z}}^{i \to j}_t - \hat{\vec{\mathbf{z}}}^{i \to j}_{t|t-1}`

### Page 32
- As written: `u^{nom}_t`
- Copy/paste: `\vec{\mathbf{u}}^{nom}_t`

- As written: `p_t`
- Copy/paste: `\vec{\mathbf{p}}_t`

- As written: `v_t`
- Copy/paste: `\vec{\mathbf{v}}_t`

- As written: `f_v(p_t, v_t)`
- Copy/paste: `\vec{\mathbf{f}}_v(\vec{\mathbf{p}}_t, \vec{\mathbf{v}}_t)`

- As written: `a_t = 2v_t`
- Copy/paste: `\vec{\mathbf{a}}_t = 2\vec{\mathbf{v}}_t`

- As written: `u^\star_t`
- Copy/paste: `\vec{\mathbf{u}}^\star_t`

- As written: `u`
- Copy/paste: `\vec{\mathbf{u}}`

- As written: `u_{min}`
- Copy/paste: `\vec{\mathbf{u}}_{min}`

- As written: `u(\lambda) = clip(u^{nom}_t - \lambda a_t, -u_{max}, u_{max})`
- Copy/paste: `\vec{\mathbf{u}}(\lambda) = clip(\vec{\mathbf{u}}^{nom}_t - \lambda \vec{\mathbf{a}}_t, -u_{max}, u_{max})`

### Page 33
- As written: `x_i = [p_i\ v_i]^\top`
- Copy/paste: `\vec{\mathbf{x}}_i = [\vec{\mathbf{p}}_i\ \vec{\mathbf{v}}_i]^\top`

- As written: `x_i^+ = A_d x_i + B_d u_i`
- Copy/paste: `\vec{\mathbf{x}}_i^+ = A_d\vec{\mathbf{x}}_i + B_d\vec{\mathbf{u}}_i`

- As written: `p_i`
- Copy/paste: `\vec{\mathbf{p}}_i`

- As written: `v_i`
- Copy/paste: `\vec{\mathbf{v}}_i`

- As written: `u_i`
- Copy/paste: `\vec{\mathbf{u}}_i`

- As written: `s = [x_1^\top,\ x_2^\top]^\top`
- Copy/paste: `\vec{\mathbf{s}} = [\vec{\mathbf{x}}_1^\top,\ \vec{\mathbf{x}}_2^\top]^\top`

- As written: `o = [(p_1-c), (p_2-c), (p_2-p_1), v_1, v_2]`
- Copy/paste: `\vec{\mathbf{o}} = [(\vec{\mathbf{p}}_1-\vec{\mathbf{c}}), (\vec{\mathbf{p}}_2-\vec{\mathbf{c}}), (\vec{\mathbf{p}}_2-\vec{\mathbf{p}}_1), \vec{\mathbf{v}}_1, \vec{\mathbf{v}}_2]`

- As written: `o_{1,EKF} = [(p_1-c), (\tilde{p}_{2,1}-c), (\tilde{p}_{2,1}-p_1), v_1, \tilde{v}_{2,1}]`
- Copy/paste: `\vec{\mathbf{o}}_{1,EKF} = [(\vec{\mathbf{p}}_1-\vec{\mathbf{c}}), (\tilde{\vec{\mathbf{p}}}_{2,1}-\vec{\mathbf{c}}), (\tilde{\vec{\mathbf{p}}}_{2,1}-\vec{\mathbf{p}}_1), \vec{\mathbf{v}}_1, \tilde{\vec{\mathbf{v}}}_{2,1}]`

- As written: `o_{2,EKF} = [(\tilde{p}_{1,2}-c), (p_2-c), (p_2-\tilde{p}_{1,2}), \tilde{v}_{1,2}, v_2]`
- Copy/paste: `\vec{\mathbf{o}}_{2,EKF} = [(\tilde{\vec{\mathbf{p}}}_{1,2}-\vec{\mathbf{c}}), (\vec{\mathbf{p}}_2-\vec{\mathbf{c}}), (\vec{\mathbf{p}}_2-\tilde{\vec{\mathbf{p}}}_{1,2}), \tilde{\vec{\mathbf{v}}}_{1,2}, \vec{\mathbf{v}}_2]`

### Page 34
- As written: `d_1 = \|p_1-c\|^2 / R^2`
- Copy/paste: `d_1 = \frac{\|\vec{\mathbf{p}}_1-\vec{\mathbf{c}}\|^2}{R^2}`

- As written: `d_2 = \|p_2-c\|^2 / R^2`
- Copy/paste: `d_2 = \frac{\|\vec{\mathbf{p}}_2-\vec{\mathbf{c}}\|^2}{R^2}`

- As written: `d_{dock} = max(0, \|p_2-p_1\| - r_{collision}) / R`
- Copy/paste: `d_{dock} = \frac{\max(0, \|\vec{\mathbf{p}}_2-\vec{\mathbf{p}}_1\| - r_{collision})}{R}`

### Page 39
- As written: `o^{r,(n)}_t`
- Copy/paste: `\vec{\mathbf{o}}^{r,(n)}_t`

- As written: `o^{\bar{r},(n)}_t`
- Copy/paste: `\vec{\mathbf{o}}^{\bar{r},(n)}_t`

- As written: `a^{r,(n)}_t`
- Copy/paste: `\vec{\mathbf{a}}^{r,(n)}_t`

- As written: `a^{\bar{r},(n)}_t`
- Copy/paste: `\vec{\mathbf{a}}^{\bar{r},(n)}_t`

- As written: `x^{(n)}_t`
- Copy/paste: `\vec{\mathbf{x}}^{(n)}_t`

- As written: `x^{(n)}_{t+1}`
- Copy/paste: `\vec{\mathbf{x}}^{(n)}_{t+1}`

### Page 40
- As written: `\hat{x}_t`
- Copy/paste: `\hat{\vec{\mathbf{x}}}_t`

- As written: `y_t`
- Copy/paste: `\vec{\mathbf{y}}_t`

- As written: `x_t`
- Copy/paste: `\vec{\mathbf{x}}_t`

- As written: `o^r_t`
- Copy/paste: `\vec{\mathbf{o}}^r_t`

- As written: `o^{\bar{r}}_t`
- Copy/paste: `\vec{\mathbf{o}}^{\bar{r}}_t`

- As written: `a^r_t`
- Copy/paste: `\vec{\mathbf{a}}^r_t`

- As written: `a^{\bar{r}}_t`
- Copy/paste: `\vec{\mathbf{a}}^{\bar{r}}_t`

- As written: `x_{t+1}`
- Copy/paste: `\vec{\mathbf{x}}_{t+1}`

- As written: `\hat{x}_{t+1}`
- Copy/paste: `\hat{\vec{\mathbf{x}}}_{t+1}`

- As written: `o`
- Copy/paste: `\vec{\mathbf{o}}`

- As written: `\mu(o)`
- Copy/paste: `\vec{\boldsymbol{\mu}}(\vec{\mathbf{o}})`

- As written: `u_{raw}`
- Copy/paste: `\vec{\mathbf{u}}_{raw}`

- As written: `a = u_{max}\tanh(u_{raw})`
- Copy/paste: `\vec{\mathbf{a}} = u_{max}\tanh(\vec{\mathbf{u}}_{raw})`

### Page 41
- As written: `o_t`
- Copy/paste: `\vec{\mathbf{o}}_t`

- As written: `a_t`
- Copy/paste: `\vec{\mathbf{a}}_t`

- As written: `\pi(\cdot \mid o_t)`
- Copy/paste: `\pi(\cdot \mid \vec{\mathbf{o}}_t)`

### Page 43
- As written: `u_{center}`
- Copy/paste: `\vec{\mathbf{u}}_{center}`

- As written: `p_2`
- Copy/paste: `\vec{\mathbf{p}}_2`

- As written: `c`
- Copy/paste: `\vec{\mathbf{c}}`

- As written: `E_2 := P A_d x_2 - c`
- Copy/paste: `\vec{\mathbf{E}}_2 := P A_d \vec{\mathbf{x}}_2 - \vec{\mathbf{c}}`

- As written: `x_2 = [p_2\ v_2]^\top`
- Copy/paste: `\vec{\mathbf{x}}_2 = [\vec{\mathbf{p}}_2\ \vec{\mathbf{v}}_2]^\top`

- As written: `u_{center} = -K E_2`
- Copy/paste: `\vec{\mathbf{u}}_{center} = -K \vec{\mathbf{E}}_2`

- As written: `u_{repulse}`
- Copy/paste: `\vec{\mathbf{u}}_{repulse}`

- As written: `r := p_2 - p_1`
- Copy/paste: `\vec{\mathbf{r}} := \vec{\mathbf{p}}_2 - \vec{\mathbf{p}}_1`

### Page 44
- As written: `u_{repulse} = g r / (max\{r_{min}^2, \|r\|^2\} + \epsilon)^{3/2}`
- Copy/paste: `\vec{\mathbf{u}}_{repulse} = \frac{g\vec{\mathbf{r}}}{(max\{r_{min}^2, \|\vec{\mathbf{r}}\|^2\} + \epsilon)^{3/2}}`

- As written: `v_2`
- Copy/paste: `\vec{\mathbf{v}}_2`

- As written: `u_2 = sat_{u_{max}}(w_c u_{center} + w_r u_{repulse} - w_d v_2)`
- Copy/paste: `\vec{\mathbf{u}}_2 = sat_{u_{max}}(w_c\vec{\mathbf{u}}_{center} + w_r\vec{\mathbf{u}}_{repulse} - w_d\vec{\mathbf{v}}_2)`

- As written: `u_{raw,t} = \mu_\theta(o_t)`
- Copy/paste: `\vec{\mathbf{u}}_{raw,t} = \vec{\boldsymbol{\mu}}_\theta(\vec{\mathbf{o}}_t)`

- As written: `a`
- Copy/paste: `\vec{\mathbf{a}}`

- As written: `u_{cmd,t}`
- Copy/paste: `\vec{\mathbf{u}}_{cmd,t}`

### Page 45
- As written: `x_t`
- Copy/paste: `\vec{\mathbf{x}}_t`

- As written: `u^{app}_t`
- Copy/paste: `\vec{\mathbf{u}}^{app}_t`

- As written: `x_{t+1}`
- Copy/paste: `\vec{\mathbf{x}}_{t+1}`

- As written: `o_t = [(p_1-c), (p^{obs}_2-c), (p^{obs}_2-p_1), v_1, v^{obs}_2]`
- Copy/paste: `\vec{\mathbf{o}}_t = [(\vec{\mathbf{p}}_1-\vec{\mathbf{c}}), (\vec{\mathbf{p}}^{obs}_2-\vec{\mathbf{c}}), (\vec{\mathbf{p}}^{obs}_2-\vec{\mathbf{p}}_1), \vec{\mathbf{v}}_1, \vec{\mathbf{v}}^{obs}_2]`

- As written: `u_{raw,t}`
- Copy/paste: `\vec{\mathbf{u}}_{raw,t}`

- As written: `u_{cmd,t}`
- Copy/paste: `\vec{\mathbf{u}}_{cmd,t}`

- As written: `\tilde{u}_t`
- Copy/paste: `\tilde{\vec{\mathbf{u}}}_t`

- As written: `u^{app}_t`
- Copy/paste: `\vec{\mathbf{u}}^{app}_t`

- As written: `x_{t+1} = D(x_t, u^{app}_t)`
- Copy/paste: `\vec{\mathbf{x}}_{t+1} = D(\vec{\mathbf{x}}_t, \vec{\mathbf{u}}^{app}_t)`

### Page 75
- As written: `c \in \mathbb{R}^D`
- Copy/paste: `\vec{\mathbf{c}} \in \mathbb{R}^D`

- As written: `p_i, v_i \in \mathbb{R}^D`
- Copy/paste: `\vec{\mathbf{p}}_i, \vec{\mathbf{v}}_i \in \mathbb{R}^D`

- As written: `x_i = [p_i\ v_i]^\top \in \mathbb{R}^{2D}`
- Copy/paste: `\vec{\mathbf{x}}_i = [\vec{\mathbf{p}}_i\ \vec{\mathbf{v}}_i]^\top \in \mathbb{R}^{2D}`

- As written: `u_i \in \mathbb{R}^D`
- Copy/paste: `\vec{\mathbf{u}}_i \in \mathbb{R}^D`

- As written: `s = [x_1^\top,\ x_2^\top]^\top \in \mathbb{R}^{4D}`
- Copy/paste: `\vec{\mathbf{s}} = [\vec{\mathbf{x}}_1^\top,\ \vec{\mathbf{x}}_2^\top]^\top \in \mathbb{R}^{4D}`

- As written: `o_{full} \in \mathbb{R}^{5D}`
- Copy/paste: `\vec{\mathbf{o}}_{full} \in \mathbb{R}^{5D}`

- As written: `\hat{x}_{j|i}, \Sigma_{j|i}`
- Copy/paste: `\hat{\vec{\mathbf{x}}}_{j|i}, \Sigma_{j|i}`

- As written: `o^{(i)}_{EKF} \in \mathbb{R}^{5D}`
- Copy/paste: `\vec{\mathbf{o}}^{(i)}_{EKF} \in \mathbb{R}^{5D}`

- As written: `d_1 = \|p_1-c\|^2 / R^2`
- Copy/paste: `d_1 = \frac{\|\vec{\mathbf{p}}_1-\vec{\mathbf{c}}\|^2}{R^2}`

- As written: `d_2 = \|p_2-c\|^2 / R^2`
- Copy/paste: `d_2 = \frac{\|\vec{\mathbf{p}}_2-\vec{\mathbf{c}}\|^2}{R^2}`

- As written: `r_{rel}^2 = \|p_2-p_1\|^2 / R^2`
- Copy/paste: `r_{rel}^2 = \frac{\|\vec{\mathbf{p}}_2-\vec{\mathbf{p}}_1\|^2}{R^2}`

- As written: `d_{dock} = max\{0,\|p_2-p_1\|-r_{cap}\}/R`
- Copy/paste: `d_{dock} = \frac{\max\{0,\|\vec{\mathbf{p}}_2-\vec{\mathbf{p}}_1\|-r_{cap}\}}{R}`

### Page 76
- As written: `u_{nom}, u_{cmd}, u_{app}`
- Copy/paste: `\vec{\mathbf{u}}_{nom}, \vec{\mathbf{u}}_{cmd}, \vec{\mathbf{u}}_{app}`

- As written: `u_{center} = -K(P_{pos}A_d x_2 - c)`
- Copy/paste: `\vec{\mathbf{u}}_{center} = -K(P_{pos}A_d\vec{\mathbf{x}}_2 - \vec{\mathbf{c}})`

- As written: `u_{repulse} = g (p_2-p_1) / (max\{r_{min}^2, \|p_2-p_1\|^2\} + \epsilon_{rep})^{3/2}`
- Copy/paste: `\vec{\mathbf{u}}_{repulse} = \frac{g(\vec{\mathbf{p}}_2-\vec{\mathbf{p}}_1)}{(max\{r_{min}^2, \|\vec{\mathbf{p}}_2-\vec{\mathbf{p}}_1\|^2\} + \epsilon_{rep})^{3/2}}`

- As written: `\mu_\theta(\cdot), \sigma_\theta(\cdot)`
- Copy/paste: `\vec{\boldsymbol{\mu}}_\theta(\cdot), \vec{\boldsymbol{\sigma}}_\theta(\cdot)`

### Page 77
- As written: `u_{cmd} = u_{max}\tanh(u_{raw})`
- Copy/paste: `\vec{\mathbf{u}}_{cmd} = u_{max}\tanh(\vec{\mathbf{u}}_{raw})`

### Page 80
- As written: `\|v_0\| \le v_{max}`
- Copy/paste: `\|\vec{\mathbf{v}}_0\| \le v_{max}`

## Notes

- I updated the page labels back to the original PDF page numbering.
- I only bolded-and-arrowed quantities that are acting like vectors.
- I did not bold scalar components like `x`, `y`, `z`, `a_x`, `u_R`, etc.
- I also did not convert matrices like `A_d`, `B_d`, `F`, `G`, `Q`, `K`, or covariance matrices unless they were clearly vectors.
