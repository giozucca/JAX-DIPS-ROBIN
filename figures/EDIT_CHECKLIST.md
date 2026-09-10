# JAX-DIPS-Robin — edit checklist

Work top to bottom. Each item gives PREVIOUS (what is there now), NEW (what to write),
and WHY. Large replacements live in `figures/*.tex` and are quoted here in summary only.

===============================================================================
FRONTMATTER
===============================================================================

-------------------------------------------------------------------------------
ITEM 1 — Abstract
-------------------------------------------------------------------------------
PREVIOUS: begins "We present a scalable strategy for development of mesh-free hybrid
neuro-symbolic partial differential equation solvers..." and ends with the JAX-DIPS
open-sourcing note. Mentions "elliptic problems with jump conditions across irregular
interfaces". Never mentions Robin.

NEW: replace the whole `\begin{abstract}...\end{abstract}` with the contents of
     figures/abstract.tex

WHY: The abstract describes a different paper. It also omits your two actual
contributions: (a) the Robin extension, (b) the optimizer-epsilon failure mode. An
abstract that does not state the contribution is the single most costly thing to leave
stale, because it is what reviewers read first.

-------------------------------------------------------------------------------
ITEM 2 — Keywords
-------------------------------------------------------------------------------
PREVIOUS:
    Level-set method \sep
    Free boundary problems \sep
    Surrogate models \sep
    Robin boundary conditions \sep
    Differentiable programming \sep
    Neural networks

NEW:
    Level-set method \sep
    Elliptic interface problems \sep
    Robin boundary conditions \sep
    Surrogate models \sep
    Differentiable programming \sep
    Neural networks

WHY: "Free boundary problems" implies a moving interface. Nothing moves in this work —
the level set is static. "Elliptic interface problems" is the accurate category.

===============================================================================
SECTION 1 — "Results from JAX-DIPS on Robin boundary condition..."
===============================================================================

-------------------------------------------------------------------------------
ITEM 3 — Move the whole section
-------------------------------------------------------------------------------
PREVIOUS: this section sits before `\section{Introduction}`.

NEW: move it, unchanged, to become the first subsections of
     `\section{Numerical results}\label{sec:results}`.

WHY: Results before the Introduction reads as scaffolding. It also means a reader meets
your figures before the problem has been stated, so they cannot interpret them.

-------------------------------------------------------------------------------
ITEM 4 — Duplicate figure labels (6 occurrences)
-------------------------------------------------------------------------------
PREVIOUS: every one of the six figures ends with
    \label{fig:placeholder}

NEW: in order of appearance,
    \label{fig:sphere-conv}
    \label{fig:sphere-fields}
    \label{fig:star-conv}
    \label{fig:star-fields}
    \label{fig:star3-conv}
    \label{fig:star3-fields}

WHY: LaTeX emits "multiply-defined labels" and every `\ref` to them resolves to the
last one. You cannot cite these figures from the text until this is fixed.

-------------------------------------------------------------------------------
ITEM 5 — Wrong filename
-------------------------------------------------------------------------------
PREVIOUS:
    \includegraphics[width=0.5\linewidth]{star_convergence_2.png}

NEW:
    \includegraphics[width=0.5\linewidth]{star_convergence.png}

WHY: The generated file has no `_2` suffix. This is a compile error, not a warning.

-------------------------------------------------------------------------------
ITEM 6 — Figure captions are too thin
-------------------------------------------------------------------------------
PREVIOUS (convergence figures, ×3):
    Different accuracy measures, RMSE and $L^{\infty}$ across 4 different resolutions,
    Sphere geometry

NEW:
    Convergence of the solution error under grid refinement for the sphere geometry
    with a Robin boundary condition. RMSE and $L^{\infty}$ are measured in
    $\Omega^{-}$ against the exact solution on a $128^3$ evaluation grid. The dashed
    line indicates a first-order slope; only its gradient is meaningful.

PREVIOUS (field figures, ×3):
    Illustration of numerical solution and absolute error on a cross section of the
    domain across resolutions

NEW:
    Solution $U$ and error $U - U_{exact}$ on a plane through the interior of the
    sphere geometry, at four training resolutions. The exterior ($\varphi > 0$) is
    masked. Both columns use a single colour scale shared across all resolutions and
    all three geometries. $L^{\infty}$ is measured over the full three-dimensional
    interior.

WHY: A caption should let a reader interpret the figure without hunting through the
text. Three facts are load-bearing and currently absent: errors are measured only in
$\Omega^{-}$, the colour scales are shared (otherwise the panels are not comparable),
and the reference line's offset is arbitrary.

-------------------------------------------------------------------------------
ITEM 7 — Add the domain description
-------------------------------------------------------------------------------
PREVIOUS: nothing. The three geometries are used without stating their domains.

NEW: insert figures/domain_paragraph.tex at the head of the results, before the first
     figure. It gives $L = 2.0, 3.6, 4.2$, derives each from the object extent plus
     one cell of stencil clearance, and includes a domain table.

WHY: Without it, a reader assumes one common $L$, computes $\Delta x = L/(N-1)$
themselves, and concludes your star3 errors are ~3x worse than the sphere's for
geometric reasons. Most of that gap is that star3's cells are 2.1x larger.

-------------------------------------------------------------------------------
ITEM 8 — Add the summary figure
-------------------------------------------------------------------------------
NEW: add figures/summary/fields_Nx64.png as a final figure of the section:

    \begin{figure}[H]
      \centering
      \includegraphics[width=0.55\textwidth]{fields_Nx64.png}
      \caption{Solution and error at $N_x = 64$ for all three geometries, on a plane
               through the interior; the exterior ($\varphi > 0$) is masked. Both
               columns share a single colour scale. $L^{\infty}$ is measured over the
               full three-dimensional interior and matches Tables~\ref{tab:conv-sphere}
               --\ref{tab:conv-star3}.}
      \label{fig:fields-nx64}
    \end{figure}

WHY: The convergence plots give the rate; this shows the error has no structure —
smooth, global, no pile-up at the interface. That last point matters because a
first-order Taylor projection in cut cells is exactly where a referee expects boundary
error to dominate. Yours is 1.05–1.36x near-boundary versus bulk, i.e. it does not.

===============================================================================
SECTION 2 — Introduction
===============================================================================

-------------------------------------------------------------------------------
ITEM 9 — Problem statement
-------------------------------------------------------------------------------
PREVIOUS: already replaced in your draft, marked [TODO: REVIEW CHANGES].

NEW: keep it; delete the two `\textbf{[TODO...]}` marker lines.
     Source: figures/problem_statement.tex

WHY: Nothing further needed. The markers will render in the PDF if left in.

-------------------------------------------------------------------------------
ITEM 10 — SUBSECTION 2.2 "Literature on relevant finite discretization methods"
             final paragraph.  FIND IT BY SEARCHING FOR:
                 "In this work we bootstrap the level-set based"
             NOTE: item 11 is a DIFFERENT paragraph in a DIFFERENT subsection that also
             begins "In this work we". Do not confuse them. This one is about which
             numerical method you build on; item 11 is the roadmap paragraph.
-------------------------------------------------------------------------------
PREVIOUS:
    In this work we bootstrap the level-set based finite volume method on Cartesian
    grids proposed by Bochkov & Gibou (2020). This method is based on the idea of
    Taylor expansions in the normal direction and employing one-sided least-square
    interpolations for imposing jump conditions. In particular, this method offers
    second order accurate numerical solutions with first order accurate gradients in
    the $L^\infty$-norm.

NEW: replace with figures/lit_finite_discretization.tex, which keeps the attribution,
     states that the method is second order FOR JUMP CONDITIONS, then gives the Robin
     projection
         u_Gamma = (mu*u_ijk + |delta|*g) / (mu + alpha*|delta|)
     and explains that it is first order in delta and is the sole route by which the
     boundary condition enters, hence caps the scheme at first order.

WHY: Two things are wrong as written — "jump conditions" is not what you impose, and
"second order" is not what you achieve. But simply swapping the words invites the
question "why first order now?", so the replacement derives it. This also protects the
attribution: their method IS second order; the order drop is a property of the Robin
projection, not a weakening of their scheme.

NOTE: the file also contains a commented-out VERSION B that adds a gradient-order
claim. Do not use it until you have extracted gradient errors from your run logs:
    grep -h 'errors in grad u in Omega_minus' <logs>
    grep -h 'normal grad u on interface minus' <logs>

-------------------------------------------------------------------------------
ITEM 11 — SUBSECTION 2.3 "Literature on solving PDEs with neural networks"
             final paragraph — i.e. the LAST paragraph of the whole Introduction.
             FIND IT BY SEARCHING FOR:
                 "In this work we propose a novel algorithm"
             This is a small word-level edit, NOT a file replacement. The file
             figures/lit_finite_discretization.tex belongs to item 10, not here.
-------------------------------------------------------------------------------
PREVIOUS (the paragraph in full):
    In this work we propose a novel algorithm for solving PDEs based on deep neural
    networks by lifting any existing mesh-based finite discretization method off of its
    underlying grid and extend it into a mesh-free method that can be applied to high
    dimensional problems on unstructured random points in an embarrassingly parallel
    fashion. In Section~\ref{sec:nbm} we present the neural bootstrapping method, next
    we apply it to an advanced finite volume discretization scheme for elliptic problems
    with jump conditions across irregular geometries in Section~\ref{sec:jaxdips}. We
    show numerical results of the proposed framework on interfacial PDE problems in
    Section~\ref{sec:results} and conclude with Section~\ref{sec:conclusion}.

CHANGE ONLY THE SECOND AND THIRD SENTENCES. The first sentence is still accurate.

OLD SECOND/THIRD SENTENCES:
    ...next we apply it to an advanced finite volume discretization scheme for elliptic
    problems with jump conditions across irregular geometries in
    Section~\ref{sec:jaxdips}. We show numerical results of the proposed framework on
    interfacial PDE problems in Section~\ref{sec:results} and conclude with
    Section~\ref{sec:conclusion}.

NEW SECOND/THIRD SENTENCES:
    In Section~\ref{sec:nbm} we present the neural bootstrapping method, next we apply
    it to an advanced finite volume discretization scheme for elliptic problems with
    Robin boundary conditions on irregular interfaces in Section~\ref{sec:jaxdips}. We
    show numerical results for three geometries in Section~\ref{sec:results}, including
    a failure mode of the optimizer that arises from preconditioning the residual and
    which limits attainable accuracy at fine resolutions, and conclude with
    Section~\ref{sec:conclusion}.

WHY: Signposts the epsilon result so it is not a surprise in Section 3. Reviewers
dislike contributions that appear without warning.

===============================================================================
SECTION 3 — Neural Bootstrapping Method
===============================================================================

-------------------------------------------------------------------------------
ITEM 12 — "Neural network approximators for the solution", TODO block 1
-------------------------------------------------------------------------------
PREVIOUS:
    The solutions of interfacial PDE problems are discontinuous, with jumps appearing
    not only in the solution but also in the solution gradient. [...] we define two
    separate neural networks [...] We use SIREN neural networks [...] with sin
    activation function [...] Weights and biases are initialized from a truncated
    normal distribution with zero mean and unit variance.

NEW: replace with figures/network_representation.tex

WHY: Four separate factual errors. (a) Nothing is discontinuous across Gamma in a Robin
problem. (b) There is one network; every `u_p` reference in the Robin assembly is
commented out. (c) The activation is tanh, not sin — `activation_m: "jnp.tanh"`.
(d) The initialization is TruncatedNormal(stddev=0.1), not unit variance, and biases
are zero, not sampled. The replacement also corrects the domain from
`R^3 ∩ Omega^-` to `R^3`, because `u_m_at_point` pins phi = -1 and so evaluates the
network outside Omega^- at stencil neighbours.

-------------------------------------------------------------------------------
ITEM 13 — "Neural network approximators for the solution", TODO block 2 (the loss)
-------------------------------------------------------------------------------
PREVIOUS:
    We define the loss function by the mean-squared-error (MSE) of the residual of the
    discretized partial differential equation with jump conditions derived in
    Section~\ref{sec:approach1} that is evaluated on the grid points:
        L(u) = ||A u_theta(x_ijk in Omega) - b||_2^2

NEW: replace with figures/loss_passage.tex, whose equation is
        L(theta) = (1/|I|) sum_{ijk in I} [ ((A u_theta)_ijk - b_ijk) / d_ijk ]^2
     with I = { ijk : phi(x_ijk) <= 0 }.

WHY: Three corrections. (a) "with jump conditions" — there are none. (b) `x_ijk in
Omega` should be `Omega^-`; exterior points generate no equation, which is what
`train_omega_minus_only: true` does and it was on for every published run. (c) The
`1/d_ijk` preconditioner is missing, and it is the term your epsilon result depends on
— without it in the paper, Section 3.3.2 has nothing to refer back to.

===============================================================================
SECTION 4 — JAX-DIPS
===============================================================================

-------------------------------------------------------------------------------
ITEM 14 — Section opening paragraph
-------------------------------------------------------------------------------
PREVIOUS:
    Below we present and compare two possible approaches for treating the jump
    conditions in the interfacial PDE solver: (i) regression-based extrapolation, and
    (ii) neural extrapolation. The main difference between the two approaches is that
    approach (i) only requires first order AD [...] Approach (i) offers better
    computational properties [...]

NEW:
    Below we present the treatment of the Robin boundary condition in the interfacial
    PDE solver. Because the condition is one-sided, no extrapolation across the
    interface is required and no auxiliary degrees of freedom are introduced: the
    unknown interface value is eliminated analytically by a Taylor projection, and the
    resulting terms enter the cell's own equation. Optimization therefore requires only
    first order automatic differentiation.

WHY: Approach II is being deleted (item 15), so a paragraph promising two approaches
leaves the section incoherent. The replacement also makes a positive point: the Robin
setting is simpler than the jump-condition one, needing neither the regression
machinery nor second-order AD.

-------------------------------------------------------------------------------
ITEM 15 — DELETE "Approach II. Finite discretization method fused with neural extrapolation"
-------------------------------------------------------------------------------
PREVIOUS: the entire `\subsection{...}\label{sec:approach2}`, from "We point out that
although in approach~I..." through the loss equation and the paragraph ending
"...restricting to only first order automatic differentiation."

NEW: delete entirely.

WHY: It requires jump conditions ([u] = alpha, [mu d_n u] = beta) which do not exist
here. Deleting it removes `\label{eq:jump_nn1}` and `\label{eq:jump_nn2}`, which forces
item 25 — Appendix B references them.

-------------------------------------------------------------------------------
ITEM 16 — "Approach I", decision required
-------------------------------------------------------------------------------
PREVIOUS: the whole subsection — the finite volume equations, the least-squares gradient
operator, zeta/gamma definitions, the Bias Slow substitution rules, Algorithm 1.

DECIDE between:
  (a) KEEP as inherited context. Add at the start:
      "For completeness we summarise the jump-condition treatment of the bootstrapped
       method~\cite{bochkov2020}; the Robin formulation of the present work does not
       use it, and is given in Section~\ref{sec:robin_disc}."
  (b) REPLACE with the Robin cut-cell derivation (integrate the PDE over a cell, apply
      the divergence theorem, substitute the Robin condition on the Gamma piece).

WHY: The Robin assembly calls none of this. `compute_Ax_and_b_preconditioned_fn_Robin`
never touches `self.u_mp_fn`, and the regression machinery is reachable only from the
jump-condition path. Option (b) is the stronger paper but is real writing;
option (a) is honest and cheap.

IMPORTANT: keep `\label{sec:approach1}` alive under either choice — item 13's loss
passage references it.

-------------------------------------------------------------------------------
ITEM 17 — Jacobi preconditioner equation
-------------------------------------------------------------------------------
PREVIOUS:
    a_{ii} = \sum_{s=-,+}\left[ k^s_{i,i}|V^s_{i,i}| + ... \right]

NEW:
    a_{ii} = k_{i,i}|V^{-}_{i,i}|
             + \frac{\mu_{i-\frac12,i}A^{-}_{i-\frac12,i} + \mu_{i+\frac12,i}A^{-}_{i+\frac12,i}}{\Delta x}
             + \frac{\mu_{i,i-\frac12}A^{-}_{i,i-\frac12} + \mu_{i,i+\frac12}A^{-}_{i,i+\frac12}}{\Delta y}
             + \int_{\Gamma \cap V_{i,i}} \alpha \, dA

WHY: Two changes. The sum over s = -,+ is wrong (one region only), and the Robin term
`alpha_ell` is part of the diagonal in the code:
    diag_coeff = k_m*V_m + sum(coeffs) + coeff_integral_au
Omitting it understates the diagonal at exactly the cells that carry the boundary
condition.

-------------------------------------------------------------------------------
ITEM 18 — "Learning rate scheduling"
-------------------------------------------------------------------------------
PREVIOUS:
    ...we use the exponential decay scheduler provided by Optax to control the learning
    rate in the Adam optimizer: r_k = r_0 alpha^{k/T} [...] By default, we set T = 100,
    alpha = 0.975, starting from an initial value of r_0 = 10^{-2} and clip gradients by
    maximum global gradient norm...

NEW: replace this subsubsection with figures/optimization_section.tex, which gives the
     cosine schedule
         r_k = (r_0/2)[1 + cos(pi k / K)],  r_0 = 1e-3, K = full horizon
     AND adds a new subsubsection "Gradient scale and the optimizer stability constant".

WHY: Every number in the current text is wrong for your runs — you use cosine, not
exponential, at 1e-3, not 1e-2, and there is no gradient clipping on the adamw path.
The added subsubsection is where your epsilon contribution lives: it derives
    ||grad L|| ~ C dx^3 e      and      e_floor ~ epsilon / (C dx^3)
and reports the 18.7x improvement at N_x = 64.

-------------------------------------------------------------------------------
ITEM 19 — DELETE "Domain switching optimization scheme" and Algorithm 2
-------------------------------------------------------------------------------
PREVIOUS: the subsubsection beginning "The linear system suffers from worse condition
number in the domain with more variability in diffusion coefficient..." plus
`\begin{algorithm}...\label{alg:domain_switching}...\end{algorithm}`.

NEW: delete both.

WHY: Dead code — the method is `__update` with two leading underscores, which Python
name-mangles; the live path is `update_fn = self.update`. It is also inapplicable in
principle: it alternates optimization between two networks based on which side has the
larger diffusion coefficient, and you have one network and one region.

-------------------------------------------------------------------------------
ITEM 20 — "Multi-GPU parallelization", decision required
-------------------------------------------------------------------------------
PREVIOUS: describes data-parallel training across GPUs.

DECIDE: keep as a library capability, but add
    "All results reported in this work were obtained on a single GPU."

WHY: `multi_gpu: false` in every run. The text does not claim you used it, but adjacent
to a results section a reader will assume so.

===============================================================================
SECTION 5 — Numerical results
===============================================================================

-------------------------------------------------------------------------------
ITEM 21 — Opening equations
-------------------------------------------------------------------------------
PREVIOUS:
    k^\pm u^\pm - \nabla\cdot(\mu^\pm\nabla u^\pm) = f^\pm,  x in Omega^\pm
    [u] = \alpha,        x in Gamma
    [\mu\partial_n u] = \beta,  x in Gamma

NEW -- OPTION A (recommended, avoids duplicating the equations):

    We consider elliptic problems with Robin boundary conditions on an embedded
    interface, as defined by Eqs.~\eqref{eq:pde}--\eqref{eq:robin}.

NEW -- OPTION B (repeat the equations here for readability; note NO \label, to avoid
                 clashing with the labels in the problem statement):

    We consider examples for solution to elliptic problems of the form
    \begin{align*}
        k\,u - \nabla \cdot (\mu \nabla u) &= f, & \mathbf{x} &\in \Omega^{-}, \\
        \alpha\, u + \mu\, \partial_{n} u &= g, & \mathbf{x} &\in \Gamma .
    \end{align*}

WHY: Must match Eqs. (1)-(2) of the problem statement. Leaving the jump form here
contradicts the Introduction within the same document.

NOTE ON TERMINOLOGY: "elliptic" classifies the differential operator, not the boundary
condition -- the principal symbol mu|xi|^2 is definite, so the operator is elliptic
regardless of whether the boundary condition is Dirichlet, Neumann or Robin. The phrase
"elliptic problems with Robin boundary conditions" is correct as written.

NOTE ON align*: use the starred form if you take Option B, so the repeated equations
are not numbered a second time.

-------------------------------------------------------------------------------
ITEM 21b — Opening paragraphs of Section "Numerical results"
-------------------------------------------------------------------------------
PREVIOUS (paragraph 1, DELETE ENTIRELY):
    Using different features of JAX-DIPS one can compose solvers with different training
    configurations; i.e., single/multi-resolution, single/multi-batch, and
    single/multi-GPU, and domain alternating training. Moreover, the neural
    extrapolation method discussed in Section~\ref{sec:approach2} provides an
    alternative solver. Below we implement and compare numerical accuracy and
    performance of these strategies.

PREVIOUS (paragraph 2, EDIT):
    For each accuracy metric we report order of convergence [...] measuring the
    $L^\infty$ error of solution and its gradient over all the grid points in the
    domain [...]

NEW: replace BOTH with figures/results_setup.tex

WHY: Paragraph 1 advertises multi-resolution, multi-batch, multi-GPU, domain
alternating training and neural extrapolation. You use none of them, and its reference
to Section~\ref{sec:approach2} breaks once item 15 deletes that section. Paragraph 2's
equation is correct and is kept, but its surrounding sentence claims gradient errors you
do not report and says errors are measured "over all the grid points in the domain"
when they are restricted to Omega^-.

The replacement also supplies the training-configuration paragraph the paper currently
lacks anywhere: architecture, optimizer, schedule, epsilon, epochs, batch, single GPU,
and the evaluation protocol including the eval-grid convergence check.

-------------------------------------------------------------------------------
ITEM 22 — Order-of-convergence definition
-------------------------------------------------------------------------------
PREVIOUS: p = log_2( err(2h)/err(h) )

NEW: keep unchanged.

WHY: This is the convention your tables use, and it matches the reference paper's
(verified against their 2^6 -> 2^7 row, which discriminates log_2 from L/(N-1)).

-------------------------------------------------------------------------------
ITEM 23 — Subsections 5.1 through 5.5, decision required
-------------------------------------------------------------------------------
PREVIOUS, all inherited jump-condition results:
    \subsection{Accuracy in the bulk: no interface}
    \subsection{Accuracy on spherical interface: single-resolution, single batch, single GPU}
    \subsection{Accuracy on star interface: single GPU, domain switching, neural extrapolation, and batching}
    \subsection{Time complexity and parallel scaling on GPU clusters}
    \subsection{Comparison with other methods} (+ LPBE, + Analysis)

DECIDE between:
  (a) DELETE all five.
  (b) KEEP, moved after your results, under a heading such as
      "\subsection{Results inherited from the jump-condition formulation}"
      with a lead sentence: "The following results are reproduced from~\cite{...} for
      context and do not concern the Robin formulation."

WHY: These are not your results. Two are actively confusing: 5.2 is a "sphere" table
with 2^3 RMSE = 3.7e-2 against your sphere's 1.35e-2, and 5.3 is a "star" table with
completely different values. A reader comparing them will think your numbers disagree
with themselves. Also note 5.3's title advertises domain switching and neural
extrapolation, both of which you are deleting.

-------------------------------------------------------------------------------
ITEM 24 — DELETE placeholder figure
-------------------------------------------------------------------------------
PREVIOUS:
    \textcolor{red}{Example of referring to the figure \ref{fig:myName}.}
    \begin{figure}[!ht] ... {myFigure.pdf} ... \label{fig:myName} \end{figure}

NEW: delete both.

WHY: Template leftover; `myFigure.pdf` does not exist, so this is a compile error.

===============================================================================
SECTION 6 — Discussion, Conclusion, front/back matter
===============================================================================

-------------------------------------------------------------------------------
ITEM 25 — "Current shortcomings and future improvements"
-------------------------------------------------------------------------------
PREVIOUS: discusses better preconditioners, adaptive mesh refinement, more expressive
architectures.

NEW: keep all of that, and add:

    Several limitations of the present study should be noted. All results use a single
    random initialization (seed 42); the sensitivity of the converged solution to
    initialization has not been characterized. The sweep spans four resolutions, giving
    two intervals in the asymptotic regime, so the fitted orders are measured over a
    limited range. The learned voxel-level preconditioner implemented in JAX-DIPS was
    not enabled in these experiments; all results use the fixed Jacobi preconditioner.

WHY: Each of these is something a reviewer will find. Stating them is cheap and
converts a potential objection into evidence you understand the study's scope.

-------------------------------------------------------------------------------
ITEM 26 — Conclusion
-------------------------------------------------------------------------------
PREVIOUS:
    We developed a differentiable multi-GPU framework for solving partial differential
    equations with jump conditions across irregular interfaces in three spatial
    dimensions.

NEW:
    We extended the neural bootstrapping method to elliptic problems with Robin
    boundary conditions on irregular interfaces in three spatial dimensions, and
    demonstrated first-order convergence across three geometries. We further identified
    a failure mode in which preconditioning of the finite-volume residual drives the
    loss gradients below the numerical-stability constant of the optimizer, producing a
    solution-error floor that grows under refinement; removing it restores monotone
    convergence and improves the error at the finest resolution by a factor of 18.7.

WHY: Same reasoning as the abstract. The conclusion currently describes the prior work.

-------------------------------------------------------------------------------
ITEM 27 — CRediT authorship contribution statement
-------------------------------------------------------------------------------
PREVIOUS: lists Pouria Mistani, Samira Pakravan, Rajesh Ilango, Frederic Gibou.

NEW: rewrite for the actual author list — Yararbas, Dickman, Younas, Long, Bresnahan,
Gibou. Template:

    \textbf{Onur Yararbas}: Methodology, Software, Validation, Investigation,
    Visualization, Writing -- original draft.
    \textbf{[Name]}: [roles].
    \textbf{Frederic Gibou}: Conceptualization, Methodology, Supervision,
    Writing -- review & editing.

WHY: The author block and CRediT statement name disjoint sets of people. This will be
caught in production if not before.

===============================================================================
APPENDICES
===============================================================================

-------------------------------------------------------------------------------
ITEM 28 — "Level-set method" (sec:levelset)
-------------------------------------------------------------------------------
NEW: keep unchanged.
WHY: Foundational and accurate.

-------------------------------------------------------------------------------
ITEM 29 — "Interpolation methods" (sec:interp)
-------------------------------------------------------------------------------
PREVIOUS: describes trilinear and quadratic non-oscillatory interpolation as if in use.

NEW: keep the subsection, and add as its final sentence:

    The test cases reported in Section~\ref{sec:results} prescribe the level-set
    function and all coefficients analytically, so these interpolation schemes are not
    exercised by those results; they are used when the level set or coefficients are
    supplied on a grid.

WHY: In the Robin path every coefficient is bound directly to an analytic function
(`phi_interp_fn = sim_state_fn.phi_fn`, and the interpolation line is commented out).
The schemes are a genuine library capability but play no part in your numbers.

-------------------------------------------------------------------------------
ITEM 30 — "Geometric integration" (sec:geoint)
-------------------------------------------------------------------------------
PREVIOUS: describes the Min & Gibou five-tetrahedron decomposition.

NEW: keep, and add after the first paragraph:

    This machinery is what imposes the Robin condition. For a cell crossed by $\Gamma$,
    the coefficient $\int_{\Gamma \cap V_{i,j,k}} \alpha \, dA$ appearing in
    Eq.~\eqref{eq:robin} and the corresponding integral of $g$ are evaluated by the
    decomposition described here; the boundary condition enters the discretisation
    through no other route.

WHY: This appendix is MORE central to the Robin paper than it was to the original. In
the jump-condition formulation the geometry supported the extrapolation rules; here it
is the sole mechanism by which the boundary condition is applied.

-------------------------------------------------------------------------------
ITEM 31 — DELETE "Solving interface problems with physics-informed neural networks"
-------------------------------------------------------------------------------
PREVIOUS: the entire final appendix, with L_interface, L_bulk, L_boundary.

NEW: delete entirely.

WHY: Built on [u] = alpha and [mu d_n u] = beta throughout, and its two equations
reference `\eqref{eq:jump_nn1}` and `\eqref{eq:jump_nn2}` from Approach II. This item
and item 15 must be done together or the PDF will show `??`.

===============================================================================
AFTER EDITING — verification
===============================================================================

  grep -c 'fig:placeholder'  main.tex     -> expect 0
  grep -c 'eq:jump_nn'       main.tex     -> expect 0
  grep -c 'myFigure'         main.tex     -> expect 0
  grep -c 'sec:approach1'    main.tex     -> expect >= 2 (label + reference)
  grep -c 'star_convergence_2' main.tex   -> expect 0
  grep -c 'usepackage{float}' main.tex    -> expect 1  (tables use [H])
  grep -c 'TODO'             main.tex     -> expect 0

Then compile twice and check the log for:
  - "multiply-defined labels"
  - "There were undefined references"
