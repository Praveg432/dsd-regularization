# Cover Letter

November 6, 2026

To the Editor-in-Chief, Neurocomputing

Dear Editor,

Please consider our manuscript, "Where Gap-Adaptive Spectral Regularization Helps, and Where It Does
Not: A Characterization and Operating Envelope for Differential Spectral Damping," for publication in
Neurocomputing.

Many kernel classifiers, twin SVM, least-squares SVM, Fisher discriminant analysis and ridge
classification among them, build their decision rule by inverting an ill-conditioned system matrix. A
recent preprint introduced Differential Spectral Damping (DSD), which regularizes such an inverse by
the local eigengap rather than the eigenvalue magnitude, and reported a large accuracy gain for twin
SVM. This manuscript settles the two questions that preprint left open: whether the idea generalizes,
and what can be proved about it.

Our contributions are: (i) an empirical result showing DSD improves ridge and Fisher classification on
real high-dimensional data, and that the gain survives a strengthened baseline set including truncated
SVD and a tuned spectral filter, not only Tikhonov; (ii) a characterization theorem establishing that
the DSD inverse is bounded, generalizes Tikhonov, and uniquely imposes a gap-monotone reliability
ordering; and (iii) a mapped operating envelope of four settings in which DSD provably reduces to
scalar regularization, each explained by a concrete mechanism.

We have been deliberate about scope and honesty, and we want to flag three points for the editor:

- The contribution is a characterization, not a performance guarantee. We attempted and failed to
  prove an excess-risk superiority over magnitude filters, and we state plainly why it is false. The
  theorem we do prove makes no risk claim.
- We correct the record on the earlier preprint: its conjecture that DSD improves algorithmic
  stability is contradicted by our measurements, and we retract it in the manuscript.
- We present the four negative boundary results as a first-class contribution. A method whose scope is
  mapped is more useful than one whose scope is asserted, and we believe this honesty strengthens
  rather than weakens the paper.

The work builds on a preprint (arXiv:2606.23407), which we cite as the origin of the method; this
submission extends it substantially with new classifiers, stronger baselines, the theorem, and the
operating envelope. It is original, not under review elsewhere, and all datasets are public. Every
reported number is reproducible from a committed result file. The author declares no competing
interests.

We believe the combination of a validated structural theorem, a robust real-data result against the
full spectral-regularizer family, and a precisely drawn operating envelope will be of interest to the
Neurocomputing readership working on kernel methods and spectral regularization.

Sincerely,

Praveg Vashishtha
Department of Computer Science and Engineering, Indian Institute of Technology Patna, India
Sopra Banking Software India
praveg\_pa2503mth259@iitp.ac.in
