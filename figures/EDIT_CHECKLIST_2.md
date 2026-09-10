# EDIT CHECKLIST 2 — review of the modified manuscript

Ordered by severity. Items A1-A5 will break the compile or the reading; do those first.

===============================================================================
A. BLOCKERS
===============================================================================

-------------------------------------------------------------------------------
A1 — THE DOMAIN PARAGRAPH IS CUT IN HALF BY THE FIGURES  (most serious)
-------------------------------------------------------------------------------
WHAT HAPPENED: the six figures and four tables were pasted into the MIDDLE of a
sentence. Currently the text reads:

    The three test geometries are of intrinsically different size: a sphere of radius
    $0.5$, a star of base radius
        % SPHERE_ROBIN RMSE AND LINF
        \begin{figure}...\end{figure}       <- six figures
        \begin{table}...\end{table}         <- three convergence tables
    $1.183$ and the union of two stars of base radius $0.8$ centered at ...

FIX: move ALL six figures and the three convergence tables to AFTER the paragraph
ends, i.e. after "...compared only within a geometry." and after
Table~\ref{tab:domains}. The sentence must read continuously:

    The three test geometries are of intrinsically different size: a sphere of radius
    $0.5$, a star of base radius $1.183$ and the union of two stars of base radius
    $0.8$ centered at $(\mp 0.5,\, \pm 0.5,\, \mp 0.5)$. The edge length $L$ is
    therefore not a free parameter but follows from the geometry. [...]

WHY: as written the sentence is unreadable and LaTeX will float the figures anywhere,
so the two halves may end up pages apart.

-------------------------------------------------------------------------------
A2 — DUPLICATE \label{eq:loss} AND DUPLICATE LOSS PARAGRAPH
-------------------------------------------------------------------------------
WHAT HAPPENED: both figures/network_representation.tex AND figures/loss_passage.tex
were pasted. They are ALTERNATIVES, not companions — my error for not saying so. The
loss equation now appears twice, both carrying \label{eq:loss}.

FIX: KEEP the first block (from "Because the Robin problem..." through "...rather than
solved for nodal unknowns."). DELETE the second block entirely, i.e. everything from

    Solution networks are evaluated on sampled points in the domain while the
    parameters of these networks are optimized using the loss function. The solution
    network is evaluated at the grid points and its parameters are optimized ...

down through

    ... are weighted comparably to full interior cells.

WHY: duplicate \label gives "multiply-defined labels", \eqref{eq:loss} resolves to the
wrong one, and the same equation is stated twice with different constant factors (one
has the 1/2, one does not).

-------------------------------------------------------------------------------
A3 — DUPLICATE "Below we present the treatment of the Robin boundary condition"
-------------------------------------------------------------------------------
WHAT HAPPENED: this paragraph appears twice in Section "JAX-DIPS", once inside the
TODO markers and once immediately after them.

FIX: delete one copy, and delete the surrounding \textbf{[TODO...]} markers.

-------------------------------------------------------------------------------
A4 — THE DIAGONAL EQUATION: four errors, not one
-------------------------------------------------------------------------------
LOCATION: \subsubsection{Preconditioners are ideal network regularizers for solving PDEs}
          -- the FINAL paragraph of that subsubsection.

DELETE everything from
    "In this work we use the Jacobi pre-conditioner. Basically, every element of the
     left-hand-side ($Au$) ... given by:"
through the equation and the closing sentence
    "Note that for memory efficiency we never explicitly compute the matrix, instead we
     compute the effect of matrix product of $Au$."

PASTE figures/diagonal_equation.tex in its place. It contains the intro sentence, the
corrected equation, the symbol definitions, and the memory-efficiency note.

THE FOUR ERRORS IT FIXES:

 (1) Lost subscript underscores. `V^{-}{i,i}`, `\mu{i-\frac12,i}`, `\int{\Gamma...}`
     and a literal `, dA`. My error -- the checklist was a .md file and markdown
     consumed the underscores as emphasis.

 (2) Index `i,i` is not a grid point. Everywhere else the paper indexes grid points as
     (i,j) in 2D or (i,j,k) in 3D; faces sit at (i +- 1/2, j, k) and so on. "i,i" is
     the diagonal of the GRID, not of the matrix. This error is inherited from the
     original paper, not introduced by the edits.

 (3) The symbol did not match the loss. Eq.(loss) calls the diagonal d_{ijk}; this
     equation called it a_{ii} and never connected them. Now it defines d_{ijk}.

 (4) THE ROBIN TERM WAS MISSING. The code computes
         diag_coeff = k_m*V_m + sum(coeffs[0,2,4,6,8,10]) + coeff_integral_au
     and that last term -- (mu_Gamma * integral of alpha over Gamma∩V) / (mu_Gamma +
     alpha_Gamma |delta|) -- is the Robin contribution to the diagonal. Omitting it
     understates the diagonal at precisely the cells that carry the boundary condition.
     This matters beyond bookkeeping: the epsilon argument in Section~\ref{sec:eps}
     rests on d_{ijk} ~ 6 mu Delta x at interface cells, which is read off this equation.

 Also: the equation is now 3D (six faces, Delta x / Delta y / Delta z). The inherited
 version had only x and y, from the 2D presentation in the Approach I section.

-------------------------------------------------------------------------------
A5 — BROKEN REFERENCE: \ref{sec:approach2}
-------------------------------------------------------------------------------
WHAT HAPPENED: Approach II was deleted (correct), but the paragraph at the END of
Section "Numerical results" still references it:

    Moreover, the neural extrapolation method discussed in Section~\ref{sec:approach2}
    provides an alternative solver.

FIX: this is item 21b — delete that whole paragraph. See B1.

===============================================================================
B. NOT YET APPLIED FROM CHECKLIST 1
===============================================================================

-------------------------------------------------------------------------------
B1 — Item 21b: the two stale paragraphs at the end of "Numerical results"
-------------------------------------------------------------------------------
DELETE both of these, which currently sit AFTER the tables:

    "Using different features of JAX-DIPS one can compose solvers with different
     training configurations; ... Below we implement and compare numerical accuracy
     and performance of these strategies."

    "For each accuracy metric we report order of convergence. ... h = \min(h_x,h_y,h_z)"

REPLACE with figures/results_setup.tex, placed BEFORE the figures (right after the
domain table). It supplies the training configuration, the evaluation protocol, and
keeps the order-of-convergence equation in corrected form.

-------------------------------------------------------------------------------
B2 — Item 1: the abstract was never replaced
-------------------------------------------------------------------------------
The abstract is still the original, describing "elliptic problems with jump conditions
across irregular interfaces" and never mentioning Robin.
REPLACE with figures/abstract.tex.

-------------------------------------------------------------------------------
B3 — Item 5: wrong figure filename
-------------------------------------------------------------------------------
PREVIOUS: \includegraphics[width=0.5\linewidth]{star_convergence_2.png}
NEW:      \includegraphics[width=0.5\linewidth]{star_convergence.png}

-------------------------------------------------------------------------------
B4 — Item 6: figure captions are still the thin originals
-------------------------------------------------------------------------------
All six still read "Different accuracy measures, RMSE and $L^\infty$ across 4 different
resolutions" / "Illustration of numerical solution and absolute error on a cross section
of the domain across resolutions". Replace with the fuller captions in checklist 1
item 6 — they must state that errors are measured in $\Omega^-$ only, that the colour
scales are shared, and that the reference line's offset is arbitrary.

-------------------------------------------------------------------------------
B5 — Remove all remaining TODO markers
-------------------------------------------------------------------------------
\textbf{[TODO: REVIEW CHANGES]} and variants appear in: Problem statement (x2),
Neural network approximators (x2), JAX-DIPS section opening (x2), the regression
subsection, Preconditioners, Learning rate scheduling, Multi-GPU, Numerical results,
Discussion, Conclusion. All will render in the PDF.

===============================================================================
C. STALE JUMP-CONDITION TEXT STILL PRESENT
===============================================================================

-------------------------------------------------------------------------------
C1 — End of "Neural network approximators for the solution"
-------------------------------------------------------------------------------
PREVIOUS:
    In the remainder of this manuscript we present details of applying NBM to solving
    elliptic problems with discontinuities across irregular interfaces.
NEW:
    In the remainder of this manuscript we present details of applying NBM to elliptic
    problems with Robin boundary conditions on irregular interfaces.

-------------------------------------------------------------------------------
C2 — Opening of Section "JAX-DIPS"
-------------------------------------------------------------------------------
PREVIOUS:
    We developed an end-to-end differentiable library for solving the elliptic problems
    with discontinuities in solution and solution gradient across irregular geometries.
NEW:
    We developed an end-to-end differentiable library for solving elliptic problems on
    irregular geometries represented implicitly by a level-set function, supporting both
    interfacial jump conditions and, as presented here, Robin boundary conditions.

-------------------------------------------------------------------------------
C3 — Discussion, "Spatial gradient calculation..." subsection, last sentence of para 2
-------------------------------------------------------------------------------
PREVIOUS:
    In the case of interfacial PDEs with jump conditions considered here, these
    conservation laws are explicitly considered in Section~\ref{sec:approach1} that
    govern the solution flux across the interface given the jump conditions.
NEW:
    In the Robin problem considered here the conservation law is imposed on each cut
    cell through the surface integral of the boundary condition over
    $\Gamma \cap V_{i,j,k}$, so the flux leaving the interface is represented exactly
    at the discrete level.

-------------------------------------------------------------------------------
C4 — Discussion: two references to a deleted section
-------------------------------------------------------------------------------
"the numerical results in Section~4.5 demonstrate that even shallow multilayer
perceptrons..."  and  "comparisons of Section~4.5 demonstrate the superior
computational efficiency of NBM ... compared to PINN-like frameworks such as INN"

Section 4.5 (the INN/LPBE comparison) has been deleted, so both claims are now
unsupported. Either delete the two sentences, or restore the comparison section.

-------------------------------------------------------------------------------
C5 — Discussion, "Current shortcomings": add the limitations paragraph
-------------------------------------------------------------------------------
Item 25 of checklist 1 was not applied. Add after the existing first paragraph:

    Several limitations of the present study should be noted. All results use a single
    random initialization; the sensitivity of the converged solution to initialization
    has not been characterized. The sweep spans four resolutions, giving two intervals
    in the asymptotic regime, so the fitted orders are measured over a limited range.
    The learned voxel-level preconditioner implemented in JAX-DIPS was not enabled;
    all results use the fixed Jacobi preconditioner.

-------------------------------------------------------------------------------
C6 — "Finite discretization method fused with regression-based extrapolation"
-------------------------------------------------------------------------------
Still carries \textbf{TODO: EITHER DELETE OR DO WITH ROBIN B.C} and its entire body is
the jump-condition machinery. This is checklist 1 item 16 and still needs your decision.
Whichever you choose, KEEP \label{sec:approach1} — the loss paragraph references it.

===============================================================================
D. MISSING CONTENT
===============================================================================

-------------------------------------------------------------------------------
D1 — The results section has no narrative
-------------------------------------------------------------------------------
Section "Numerical results" currently contains the domain discussion, then figures and
tables, then two stale paragraphs. There is no text that says what the results show.
At minimum, add after the tables:

    Across all three geometries the solution error decreases monotonically under
    refinement in both norms. Fitted over the three finest resolutions, the observed
    orders are 1.04, 0.98 and 1.14 in RMSE and 1.06, 1.17 and 1.45 in $L^{\infty}$ for
    the sphere, star and two-star geometries respectively, consistent with the
    first-order accuracy of the interface projection~\eqref{eq:taylor_robin}. The error
    fields are smooth and show no accumulation at the interface: the ratio of RMS error
    within three cells of $\Gamma$ to that in the bulk lies between 1.05 and 1.36 at
    every resolution, and does not grow under refinement.

    At the coarsest resolution the two-star geometry is under-resolved. Its interface is
    covered by 18 grid points, all of which lie in cells cut by $\Gamma$, so no equation
    enforces the bulk operator and the discrete system is underdetermined; training
    reaches a residual at machine precision while the solution error exceeds the
    amplitude of the exact solution. This point is retained in Table~\ref{tab:conv-star3}
    for completeness but is not indicative of the scheme's convergence behaviour.

-------------------------------------------------------------------------------
D2 — CRediT and Acknowledgement sections were deleted
-------------------------------------------------------------------------------
Both \section*{CRediT authorship contribution statement} and
\section*{Acknowledgement} are gone. Elsevier requires CRediT. Restore with the
correct author list (checklist 1 item 27).

-------------------------------------------------------------------------------
D3 — Bibliography file name
-------------------------------------------------------------------------------
\bibliography{JAX-DIPS-Robin} — confirm the .bib file is actually named
JAX-DIPS-Robin.bib. It was JAX-DIPS.bib before.

===============================================================================
E. VERIFY AFTER EDITING
===============================================================================

  grep -c 'TODO'                 main.tex   -> 0
  grep -c 'fig:placeholder'      main.tex   -> 0
  grep -c 'sec:approach2'        main.tex   -> 0
  grep -c 'star_convergence_2'   main.tex   -> 0
  grep -c 'label{eq:loss}'       main.tex   -> 1     (currently 2)
  grep -c 'Section~4.5'          main.tex   -> 0
  grep -c 'jump condition'       main.tex   -> only in the literature review and
                                                the deliberate contrast in the
                                                problem statement
  grep -c 'usepackage{float}'    main.tex   -> 1     (tables use [H]; NOT present yet)

NOTE: \usepackage{float} is MISSING from your preamble. Every table uses [H], which
requires it. Add it next to \usepackage{booktabs}.
