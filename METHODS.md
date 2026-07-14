# Reparametrization of GFN2-xTB for the selenocysteine active site

*Methods writeup for the ORCA-referenced, Selenium-extended GFN2-xTB re-fit produced with
this repository. Bracketed `[...]` items are placeholders to fill or cite.*

## Reference data
Reference single-point calculations were performed with ORCA [ver.] at the **B3LYP-D4/def2-TZVP**
level on the capped quantum-mechanical region of the MsrB selenocysteine variant (47 atoms,
C₁₂H₂₇N₄O₂S₁Se₁; net charge +1; singlet), using the `EnGrad` task so that each calculation
yields the total energy, the Cartesian nuclear gradient, and the atomic (Mulliken and Löwdin)
charges. Calculations were carried out in the gas phase on the isolated QM region, following the
rationale that fitting the *intrinsic* semiempirical Hamiltonian—rather than a specific
electrostatic embedding—yields parameters that transfer between environments and avoids
overfitting to a particular MM context [Velázquez-Libera et al., *J. Chem. Theory Comput.* 2025,
21, 5118]. Configurations were sampled along the characterized reaction path, comprising
**48 nodes** (positions along the reaction coordinate) with **50 replica snapshots** per node,
for a total of **2400 reference structures**. Each structure was stored as a compact
per-structure JSON record (total energy, and, per atom, element, Cartesian coordinates, Mulliken
charge, and gradient).

## ORCA reference ingestion
The optimization engine is the multiobjective evolutionary reparametrization framework of
Velázquez-Libera et al. (`gfn2-xtb_paramfitter`), which natively consumes Gaussian outputs via
cclib. To use **ORCA** reference data we added a program-agnostic ingestion layer that parses the
compact ORCA `EnGrad` JSON records and assembles them into the array-based dataset consumed by the
framework's JSON objective (`ECGFittingV3Json`/`JSONCurveHandler`). During assembly, units are
harmonized to those expected by the fitter—energies converted to kcal · mol⁻¹, coordinates to
Bohr, gradients retained as the energy gradient (dE/dx, Eₕ · Bohr⁻¹; internally negated to
forces), and charges in e—and the replica snapshots are grouped into independent relative-energy
"curves". This route also sidesteps program-specific conventions (e.g., the sign difference
between Gaussian "Forces" = −gradient and ORCA "CARTESIAN GRADIENT" = +gradient). Per-system
structure exclusions and train/validation partitioning (by replica index) are applied at this
stage; for the present system a full self-consistent-field convergence screen with stock
GFN2-xTB flagged no failures, so all 2400 structures were retained.

Because several near–transition-state geometries have small HOMO–LUMO gaps, all xTB single points
were evaluated with Fermi smearing at an electronic temperature of **500 K** and up to **300 SCF
iterations**, a setting verified to guarantee convergence across the dataset while perturbing
well-behaved structures negligibly (per-atom Mulliken change ≲ 0.003 e).

## Extension of the tunable parameter set to selenium
In the original work the tunable GFN2-xTB parameters comprise the element-specific parameters of
H, C, N, O and S (75 parameters). Selenium is present in GFN2-xTB at its stock values but is not
tunable, which would leave the **reactive selenocysteine center frozen**. We therefore extended
the optimizable set to include **selenium (Z = 34)**, mirroring exactly the treatment of its
lighter chalcogen homologue sulfur. The 20 element-specific Se parameters were exposed to the
optimizer: the shell level energies and Slater exponents of the s, p and d shells; the chemical
hardness (GAM) and third-order Hubbard term (GAM3); the coordination-number level shifts (KCN,
s/p/d); the atomic dipole and quadrupole polarizabilities (DPOL, QPOL); the repulsion parameters
(REPA, REPB); the H₀ coordination polynomials (POLY, s/p/d); and the shell multipole parameters
(LPAR, p/d). This brings the total to **95 optimizable parameters**. The selenium slots are
introduced by injecting optimization placeholders into a copy of the parameter-file template at
run time, leaving the bundled parameter library unmodified; the parameter set is selectable
(H/C/N/O/S versus H/C/N/O/S/Se), so S-only systems are unaffected. Initial guesses (and the
search-domain center) for the new selenium parameters are the stock GFN2-xTB values.

## Multiobjective reparametrization
Parameters were optimized against the DFT reference by simultaneously minimizing the discrepancy
in three properties—(i) the **relative potential-energy profile** along each replica curve (both
its shape, via the non-redundant matrix of pairwise energy differences, and its RMSE after
mean-centering), (ii) the **atomic Mulliken charges**, and (iii) the **atomic gradients** (norm
of the force-difference vector). These are combined into a single global score, defined as the
product of the gradient, charge and energy scores, each constructed with exponential penalty
terms that heavily weight large local deviations [Velázquez-Libera et al.]. Because the present
reaction spans ~60–100 kcal · mol⁻¹, the energy-penalty reference scale was set accordingly to
keep the objective finite and well-conditioned. The score was minimized with the Dynamic Domain
Genetic Algorithm (DDGA): a Latin-hypercube-initialized population (seeded with the stock
GFN2-xTB parameters), tight variable bounds of ±2 % about the stock values, and dynamic expansion
of the search domain toward promising regions [population 80, up to 50 generations with early
stopping after 12 generations without improvement]. The training set comprised the reaction-path
replicas; held-out replicas and a per-node cross-validation provided out-of-sample (interpolation
and extrapolation) error estimates. The optimized parameters are exported as a standard GFN2-xTB
parameter file.

## Use in free-energy (PMF) calculations
The re-fitted GFN2-xTB parameters serve as the QM Hamiltonian in subsequent QM/MM simulations, in
which the reduced computational cost relative to DFT permits the extensive conformational sampling
required to converge the potential of mean force along the reaction coordinate [method, e.g.
Adaptive String Method / umbrella sampling], while reproducing the higher-level reference
energetics that stock GFN2-xTB fails to capture at the selenocysteine reactive center.

---

### Key references
- C. Bannwarth, S. Ehlert, S. Grimme. *J. Chem. Theory Comput.* **2019**, 15, 1652 (GFN2-xTB).
- E. Caldeweyher et al. *J. Chem. Phys.* **2019**, 150, 154122 (D4 dispersion).
- J. L. Velázquez-Libera, R. Recabarren, E. Vöhringer-Martinez, Y. Salgueiro, J. J. Ruiz-Pernía,
  J. Caballero, I. Tuñón. *J. Chem. Theory Comput.* **2025**, 21, 5118 (MOOP/DDGA reparametrization).
- [Adaptive String Method / PMF reference, e.g. Zinovjev & Tuñón, *J. Phys. Chem. A* **2017**, 121, 9764.]
