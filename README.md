# Static Range and Points-to Analyzer (LLVM based)

This repository contains two out-of-tree LLVM **analysis/optimisation passes** built as plugins using the new PassManager:

## 1. Points-to Analysis
- A **flow-insensitive, Andersen-style points-to analysis**.
- Tracks which memory objects (`alloca`, globals, `malloc`-like heap allocs) a pointer value may refer to.
- Very simple: handles address-of assignments (`x = &y`), pointer copies (`x = y`), and basic heap/global/alloca objects.
- Prints the set of possible objects for each pointer SSA value.
