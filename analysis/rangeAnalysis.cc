#include <map>
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/CFG.h"

using namespace llvm;

namespace {

//
// Range : Represent a range of integer values
//
class Range {
private:
    int lower, upper;   // Lower and Upper bounds (both inclusive)

public:
    // Constructors
    Range(int a) : lower(a), upper(a) {};
    Range(int lower, int upper) : lower(lower), upper(upper) {};
    Range();

    // Functions to perform arithmetic on Ranges
    static Range addRange(Range r1, Range r2);
    static Range subRange(Range r1, Range r2);
    static Range mulRange(Range r1, Range r2);
    static Range mergeRange(Range r1, Range r2);
    static Range intersectRange(Range r1, Range r2);

    void print(raw_ostream &os);
    // Comparison operators for Ranges
    friend bool operator< (const Range& lhs, const Range& rhs);
    friend bool operator==(const Range& lhs, const Range& rhs);
    friend bool operator!=(const Range& lhs, const Range& rhs);
};

// Special Ranges to represent Top and Bottom of the Lattice
static Range FULL_RANGE(INT_MIN, INT_MAX);
static Range EMPTY_RANGE(INT_MAX, INT_MIN);

Range::Range() {
    lower = EMPTY_RANGE.lower;
    upper = EMPTY_RANGE.upper;
}

void Range::print(raw_ostream &os) { 
    if (*this == EMPTY_RANGE) os << "[EMPTY]";
    else os << "[" << lower << "," << upper << "]";
}

Range Range::addRange(Range r1, Range r2) {
    // TODO
    return EMPTY_RANGE;
}

Range Range::subRange(Range r1, Range r2) {
    // TODO
    return EMPTY_RANGE;
}

Range Range::mulRange(Range r1, Range r2) {
    // TODO
    return EMPTY_RANGE;
}

Range Range::mergeRange(Range r1, Range r2) {
    // TODO
    return EMPTY_RANGE;
}

Range Range::intersectRange(Range r1, Range r2) {
    // TODO
    return EMPTY_RANGE;
}

/// Helper function to print Range
raw_ostream& operator<<(raw_ostream& os, Range r) { r.print(os); return os; }

// Comparison operators for Ranges
bool operator< (const Range& lhs, const Range& rhs) {
    if (lhs == EMPTY_RANGE) return true;
    if (rhs == EMPTY_RANGE) return false;
    return std::tie(lhs.lower, lhs.upper) < std::tie(rhs.lower, rhs.upper);
}
bool operator==(const Range& lhs, const Range& rhs) {
    return std::tie(lhs.lower, lhs.upper) == std::tie(rhs.lower, rhs.upper);
}
bool operator!=(const Range& lhs, const Range& rhs) {
    return std::tie(lhs.lower, lhs.upper) != std::tie(rhs.lower, rhs.upper);
}

//
// BasicBlockState : Represent the data flow facts of a Basic Block
//
class BasicBlockState {
private:
    // TODO Declare structures to keep track of data flow facts

public:
    BasicBlockState () {}

    // TODO Implement the transfer functions

    // TODO Implement the meet function

    void print(raw_ostream &os) {
        // Print data for debugging
        os << "Basic Block State\n";
    }
};

//
// Intra-procedural Data flow Analysis
//
void analyseFunction(Function &F) {
    std::map<BasicBlock*, BasicBlockState*> BBState;

    // TODO Implement Kildall's algorithm to analyse function

    if (false) { // Set to true for debugging
        for (BasicBlock &bb: F) {
            errs() << bb << "\n";
            BBState[&bb]->print(errs());
            errs() << "\n";
        }
    }

    //
    // Print the results
    //
    errs() << "Function " << F.getName() << "\n";
    // The below code snippet iterates over the Debug Info records to find out the names
    // of local variables, and adds then to programVariables map
    std::map<StringRef, Value*> programVariables;
    for (BasicBlock &bb : F)
        for (Instruction &inst: bb)
            for (DbgVariableRecord &DVR : filterDbgVars(inst.getDbgRecordRange()))
                programVariables[DVR.getVariable()->getName()] = DVR.getAddress();
    for (BasicBlock &bb : F) {
        if (succ_size(&bb) == 0) {
            for (auto entry: programVariables) {
                StringRef variableName = entry.first;
                Value *variableValue = entry.second;
                errs() << variableName << " : ";
                // TODO Print the Range computed by the analysis for the given variable
                errs() << "\n";
            }
        }
    }
    errs() << "\n";
}

//
// Registering the Function Pass (Don't change any code below)
//
class RangeAnalysisPass : public PassInfoMixin<RangeAnalysisPass> {
public:
    static bool isRequired() { return true; }

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        analyseFunction(F);
        return PreservedAnalyses::all();
    };
};

}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        .APIVersion = LLVM_PLUGIN_API_VERSION,
        .PluginName = "Range Analysis Pass",
        .PluginVersion = "v0.1",
        .RegisterPassBuilderCallbacks = [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM, ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "range-analysis") {
                        FPM.addPass(RangeAnalysisPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
