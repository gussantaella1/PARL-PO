# Missing And Incorrect Vector Notation in Current `MS_Thesis_Final_V2.pdf`

This report reflects the current state of the PDF and lists only the remaining vector-notation issues I could still see from the latest extraction pass.

## Summary

Most of the earlier vector issues now look fixed in the PDF, especially in:
- Pages `25–29`: HCW and Elliptical LTV dynamics
- Page `32`: EKF algorithm
- Pages `43–44`: PPO / KF-On algorithms
- Pages `77–79`: glossary and implementation terminology are now mostly corrected

## Remaining Issues

### Page 44
Policy development section is mostly corrected, but one plain observation symbol still appears in prose:

- Current text:
  - `The critic V_\theta(\vec{\mathbf{o}}) then maps the full observation o to a scalar value.`
- Suggested correction:
  - `The critic V_\theta(\vec{\mathbf{o}}) then maps the full observation \vec{\mathbf{o}} to a scalar value.`

### Page 45
PPO objective section appears to have one remaining plain observation term:

- Current text:
  - `Letting R_t = A_t + V(o_t) for a minibatch B, ...`
- Suggested correction:
  - `Letting R_t = A_t + V(\vec{\mathbf{o}}_t) for a minibatch B, ...`

The rest of that page looks much better now. In particular, these appear corrected already:
- `V(\vec{\mathbf{o}}_{t+1})`
- `V(\vec{\mathbf{o}}_t)`
- `\pi(\cdot \mid \vec{\mathbf{o}}_t)`
- `\pi_\theta(\vec{\mathbf{a}}_t \mid \vec{\mathbf{o}}_t)`
- `\pi_{\mathrm{old}}(\vec{\mathbf{a}}_t \mid \vec{\mathbf{o}}_t)`

## Minor Consistency Notes

These are not clear missing-vector errors, but they are worth one last consistency check in source if you want the notation fully polished:

- Page `44`:
  - make sure the final sentence uses `\vec{\mathbf{o}}` consistently both times
- Pages `77–79`:
  - glossary notation now largely shows vector markers, but it would be worth verifying the source still uses your preferred convention consistently:
    - Latin vectors as `\vec{\mathbf{x}}`
    - Greek vector parameters as `\vec{\boldsymbol{\mu}}`, `\vec{\boldsymbol{\sigma}}`

## Bottom Line

At this point, the remaining vector issues visible in the PDF are very small:
- one plain `o` on page `44`
- one plain `o_t` on page `45`
