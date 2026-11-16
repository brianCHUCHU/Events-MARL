CS238 Project Status Update

Main text and reference: page 1. Appendix A contains the pilot tables and brief reproducibility notes; Appendix B contains the AI Use Disclosure, both on page 2.

The current report covers twelve completed seed-42 runs: seven Pressure Plate settings, including the learned gate and coverage pilots with zero query penalty, and five Navigation settings. All settings within an environment share a training budget (200 rounds for Pressure Plate; 400 for Navigation).

Peak round denotes the earliest round attaining a positive peak success rate, rather than the total training duration. Final Q is measured at the common final round and includes initialization. Queries refresh a whole team in Pressure Plate and an individual agent adapter in Navigation.

Overleaf: upload this source package, select CS238_Project_Status_Update.tex as the main document, and compile with pdfLaTeX.

To use your existing proposal project, replace its main .tex with CS238_Project_Status_Update.tex. It uses the same icml2026 package. The AI Use Disclosure and bibliography are embedded directly in the main source; no separate disclosure .tex, external .bib, or figure files are required.

Local compilation:
  latexmk -pdf CS238_Project_Status_Update.tex

The ICML style files are unmodified files from the official archive recorded in template-source.txt. Main-body fonts and margins are unchanged.
