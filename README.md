# Static Range and Points-to Analyzer (LLVM based)

This repository contains two out-of-tree LLVM **analysis/optimisation passes** built as plugins using the new PassManager:

## 1. Range Analysis
- Intraprocedural **interval analysis** over LLVM IR integers.
- Computes an over-approximation of possible values for each SSA integer: intervals `[min, max]` (with `±∞` as needed).
- Initialization: constants → singleton intervals; arguments/unknowns → `[-∞, +∞]` (Top).
- Transfer functions for common ops: `+`, `-`, `*`, `/` (conservative), bitwise `& | ^`, shifts `<< >>` (with width-aware bounds).
- Honors overflow hints when present: uses `nsw`/`nuw` to keep bounds tight; otherwise falls back to wrap-safe conservative intervals.
- **Branch refinement**: uses `icmp` predicates on CFG edges to tighten ranges (e.g., from `if (x < 10)` infer `x ∈ [-∞, 9]` on the true edge).
- **Loops**: iterates to a fixpoint with **widening** (and optional narrowing) to ensure termination.
- Produces simplifications when intervals collapse to singletons or make conditions constant:
  - Constant-fold arithmetic with singleton intervals.
  - Remove always-true/always-false branches and unreachable blocks.
  - Tighten comparisons (e.g., replace `x < 1000` with `true` if `x ∈ [0, 10]`).
- Reporting: prints the inferred interval for each relevant SSA value (e.g., ``%i.3 ∈ [0, n-1]``) and notes infeasible edges.
- Limitations (by design for simplicity): intraprocedural only; non-relational (doesn’t track relations like `x = y + 1`); ignores pointer-derived arithmetic.

## 2. Points-to Analysis
- A **flow-insensitive, Andersen-style points-to analysis**.
- Tracks which memory objects (`alloca`, globals, `malloc`-like heap allocs) a pointer value may refer to.
- Very simple: handles address-of assignments (`x = &y`), pointer copies (`x = y`), and basic heap/global/alloca objects.
- Prints the set of possible objects for each pointer SSA value.
