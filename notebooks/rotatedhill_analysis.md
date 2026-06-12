My honest interpretation: **the pipeline may be fitting the data, but these plots do not look like it has cleanly discovered the rotated Hill structure yet.** They show some physically reasonable signals, but also some red flags.

For rotated Hill, the clean theoretical form is

[
f^2(\sigma)=\sigma:\mathbb{A}_{\text{rot}}:\sigma
]

so in your invariant file this should correspond mainly to

[
I_{11}=\sigma:\mathbb{A}:\sigma .
]

Your code indeed defines (I_{11}) exactly as a quadratic contraction with the fourth-order tensor (A), while (I_{12}) and (I_{13}) are higher-order variants involving (\sigma^2).  Also, the file groups (I_{11},I_{12},I_{13}) as fourth-order-structure invariants, while (I_1,I_2,I_3) are basic stress invariants. 

So ideally, for rotated Hill, I would expect the encoder plot to show something like:

[
I_{11} \gg \text{others}
]

or, if `homogenize=True`, the homogenized version of (I_{11}), because your code treats (I_{11}) as degree 2 and applies a signed root when homogenization is enabled. 

But your encoder plot shows that **(I_1,I_2,I_3,I_4)** are the strongest, while **(I_{11})** is only moderately used. That is not what I would expect for a clean rotated Hill discovery.

The biggest red flag is the eigenvalue plot of the learned (A). For a Hill-type quadratic yield surface, the effective quadratic matrix should be **positive semi-definite**, not indefinite. Since Hill is pressure-insensitive, you usually expect one near-zero hydrostatic mode and the remaining modes non-negative. But your plot has several negative eigenvalues. That means the learned (A) is not, by itself, a valid convex Hill quadratic form.

So I would interpret the learned (A) like this:

[
\text{learned } A \neq \mathbb{A}_{\text{rot-Hill}}
]

At least not in a physically clean sense.

This does **not necessarily mean the code is broken**. It may simply mean the model is using the invariants as a flexible feature basis, not as a symbolic/physical decomposition. Since the final model probably has an encoder and MLP after the invariants, the network can combine (I_1,\dots,I_{13}) in many non-unique ways. Then the learned (A) is just “some useful tensor feature,” not necessarily the actual rotated Hill tensor.

The suspicious parts are:

1. **High importance of (I_1)**
   Classical Hill is hydrostatic-pressure independent. Since (I_1=\mathrm{tr}(\sigma)), strong use of (I_1) can mean the model learned spurious hydrostatic dependence.

2. **High importance of (I_2,I_3)**
   These are isotropic invariants. They can help represent deviatoric information together with (I_1), but rotated Hill anisotropy should mainly need structural information, especially (I_{11}).

3. **Non-negligible use of (I_{12},I_{13})**
   Rotated Hill is quadratic. (I_{12}) and (I_{13}) are higher-degree features, so they should not be necessary for the exact equation.

4. **Negative eigenvalues of (A)**
   This is the strongest warning. A true Hill yield tensor should not produce negative quadratic energy directions.

5. **Encoder L1 strength is scale-dependent**
   If the invariants are not standardized properly, the L1 column strengths are not reliable as “physical importance.” A feature can look important just because of its numerical scale.

I would run these checks:

[
f(\sigma + pI) \stackrel{?}{=} f(\sigma)
]

for random (p). If this fails, the model learned pressure sensitivity, which is wrong for Hill.

Then train a restricted model:

[
\hat{q}(\sigma)=I_{11}
]

with target

[
q=f^2
]

instead of (f). No MLP, no (I_1)-(I_{10}), no (I_{12},I_{13}). If the pipeline is correct, rotated Hill should be recoverable almost exactly through (I_{11}).

Also, when you plot eigenvalues of the 6×6 Voigt map, make sure your Voigt/Mandel convention is correct. Because your stress vector order is

[
[\sigma_{11},\sigma_{22},\sigma_{33},\sigma_{23},\sigma_{13},\sigma_{12}],
]

and the tensor form duplicates shear entries in the symmetric (3\times3) tensor. Your file explicitly maps the vector to a symmetric tensor with both (\sigma_{23}) and (\sigma_{32}), both (\sigma_{13}) and (\sigma_{31}), etc.  If the 6×6 matrix was built without the correct shear factors, the eigenvalue plot may be misleading.

So my conclusion is:

[
\boxed{
\text{The model may fit the data, but the learned invariants/A are not yet physically interpretable as rotated Hill.}
}
]

The main issue is probably **non-identifiability + lack of PSD/convexity constraint on (A)**, rather than a simple bug. For a cleaner pipeline, I would constrain (A) to be positive semi-definite, enforce hydrostatic invariance, standardize the invariant features, and first verify that (I_{11}) alone can fit the squared rotated Hill target.
