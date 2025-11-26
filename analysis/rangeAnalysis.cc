#include <iostream>
#include <map>
#include <set>
#include <limits>
#include <vector>
#include <algorithm>
#include <climits>
#include <queue>

#include "llvm/IR/InstrTypes.h"
#include "llvm/Pass.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/CFG.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

class Range {
private:
    int lower, upper;

    static int sat(long long v) {
        if (v > (long long)INT_MAX) return INT_MAX;
        if (v < (long long)INT_MIN) return INT_MIN;
        return (int)v;
    }

public:
    Range(int a) : lower(a), upper(a) {}
    Range(int l, int u) : lower(l), upper(u) {}
    Range() : lower(INT_MAX), upper(INT_MIN) {} // EMPTY

    static Range FULL() { return Range(INT_MIN, INT_MAX); }
    static Range EMPTY() { return Range(INT_MAX, INT_MIN); }

    int lo() const { return lower; }
    int hi() const { return upper; }
    bool isEmpty() const { return lower > upper; }

    static Range addRange(Range r1, Range r2) {
        if (r1.isEmpty() || r2.isEmpty()) return EMPTY();
        return Range(sat((long long)r1.lower + (long long)r2.lower),
                     sat((long long)r1.upper + (long long)r2.upper));
    }

    static Range subRange(Range r1, Range r2) {
        if (r1.isEmpty() || r2.isEmpty()) return EMPTY();
        return Range(sat((long long)r1.lower - (long long)r2.upper),
                     sat((long long)r1.upper - (long long)r2.lower));
    }

    static Range mulRange(Range r1, Range r2) {
        if (r1.isEmpty() || r2.isEmpty()) return EMPTY();
        long long a = (long long)r1.lower * (long long)r2.lower;
        long long b = (long long)r1.lower * (long long)r2.upper;
        long long c = (long long)r1.upper * (long long)r2.lower;
        long long d = (long long)r1.upper * (long long)r2.upper;
        long long mn = std::min({a, b, c, d});
        long long mx = std::max({a, b, c, d});
        return Range(sat(mn), sat(mx));
    }

    static Range mergeRange(Range r1, Range r2) {
        if (r1.isEmpty()) return r2;
        if (r2.isEmpty()) return r1;
        return Range(std::min(r1.lower, r2.lower), std::max(r1.upper, r2.upper));
    }

    static Range intersectRange(Range r1, Range r2) {
        if (r1.isEmpty() || r2.isEmpty()) return EMPTY();
        int l = std::max(r1.lower, r2.lower);
        int u = std::min(r1.upper, r2.upper);
        return (l > u) ? EMPTY() : Range(l, u);
    }

    bool operator==(const Range& other) const {
        return lower == other.lower && upper == other.upper;
    }

    bool operator!=(const Range& other) const {
        return !(*this == other);
    }

    void print(raw_ostream &os) const {
        if (isEmpty()) os << "[EMPTY]";
        else os << "[" << lower << "," << upper << "]";
    }
};

// Special Ranges to match code 1
static Range FULL_RANGE = Range::FULL();
static Range EMPTY_RANGE = Range::EMPTY();

class BasicBlockState {
public:
    std::map<Value*, Range> varRanges;

    Range get(Value *V) const {
        auto it = varRanges.find(V);
        return (it != varRanges.end()) ? it->second : Range::EMPTY();
    }

    void set(Value *V, Range R) {
        varRanges[V] = R;
    }

    bool meet(const BasicBlockState &other) {
        bool changed = false;
        for (auto const& [val, other_range] : other.varRanges) {
            if (varRanges.find(val) == varRanges.end()) {
                varRanges[val] = other_range;
                changed = true;
            } else {
                Range oldRange = varRanges[val];
                Range newRange = Range::mergeRange(oldRange, other_range);
                if (oldRange != newRange) {
                    varRanges[val] = newRange;
                    changed = true;
                }
            }
        }
        return changed;
    }

    bool operator==(const BasicBlockState &other) const {
        if (varRanges.size() != other.varRanges.size()) return false;
        for (auto const& [val, range] : varRanges) {
            auto it = other.varRanges.find(val);
            if (it == other.varRanges.end() || it->second != range) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const BasicBlockState &other) const {
        return !(*this == other);
    }
};

// Helper utilities
class RangeAnalysisUtils {
public:
    static AllocaInst* getUnderlyingAlloca(Value *Ptr) {
        Ptr = stripCasts(Ptr);
        if (auto *AI = dyn_cast<AllocaInst>(Ptr))
            return AI;
        if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
            if (auto *AI2 = dyn_cast<AllocaInst>(stripCasts(GEP->getPointerOperand())))
                return AI2;
        return nullptr;
    }

    static Value* stripCasts(Value *V) {
        while (auto *BC = dyn_cast<BitCastOperator>(V))
            V = BC->getOperand(0);
        return V;
    }

    static std::set<std::pair<BasicBlock*, BasicBlock*>> findBackEdges(Function &F) {
        SmallVector<std::pair<const BasicBlock*, const BasicBlock*>, 8> llvmBackEdges;
        FindFunctionBackedges(F, llvmBackEdges);
        std::set<std::pair<BasicBlock*, BasicBlock*>> backEdges;
        for (const auto& edge : llvmBackEdges) {
            backEdges.insert({const_cast<BasicBlock*>(edge.first),
                            const_cast<BasicBlock*>(edge.second)});
        }
        return backEdges;
    }

    static std::map<StringRef, Value*> collectProgramVariables(Function &F) {
        std::map<StringRef, Value*> programVariables;
        for (BasicBlock &bb : F) {
            for (Instruction &inst: bb) {
                for (DbgVariableRecord &DVR : filterDbgVars(inst.getDbgRecordRange())) {
                    programVariables[DVR.getVariable()->getName()] = DVR.getAddress();
                }
            }
        }
        for (auto& arg : F.args()) {
            if(arg.hasName()) {
                programVariables[arg.getName()] = &arg;
            }
        }
        return programVariables;
    }
};

// Range evaluation utilities
class RangeEvaluator {
private:
    BasicBlockState &state;
    std::map<BasicBlock*, BasicBlockState> &out;

public:
    RangeEvaluator(BasicBlockState &s, std::map<BasicBlock*, BasicBlockState> &o)
        : state(s), out(o) {}

    Range getRangeForValue(Value *V) {
        if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
            return Range(CI->getSExtValue(), CI->getSExtValue());
        }
        if (LoadInst* LI = dyn_cast<LoadInst>(V)) {
            Value* ptr = LI->getPointerOperand();
            if (AllocaInst* AI = RangeAnalysisUtils::getUnderlyingAlloca(ptr)) {
                if (state.varRanges.count(AI)) return state.varRanges[AI];
            }
        }
        if (state.varRanges.count(V)) {
            return state.varRanges[V];
        }
        return FULL_RANGE;
    }

    Range getRangeFromPred(Value *V, BasicBlock* pred) {
        if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
            return Range(CI->getSExtValue(), CI->getSExtValue());
        }
        if (out[pred].varRanges.count(V)) {
            return out[pred].varRanges[V];
        }
        if(LoadInst* LI = dyn_cast<LoadInst>(V)){
            Value* ptr = LI->getPointerOperand();
            if (AllocaInst* AI = RangeAnalysisUtils::getUnderlyingAlloca(ptr)) {
                if (out[pred].varRanges.count(AI)) return out[pred].varRanges[AI];
            }
        }
        return FULL_RANGE;
    }
};

// Branch condition refinement
class BranchRefinement {
public:
    static void applyBranchRefinement(BasicBlock *pred, BasicBlock *curr,
                                    BasicBlockState &predOutState) {
        BranchInst *br = dyn_cast<BranchInst>(pred->getTerminator());
        if (!br || !br->isConditional()) return;

        ICmpInst *compare = dyn_cast<ICmpInst>(br->getCondition());
        if (!compare) return;

        Value* var_val = compare->getOperand(0);
        Value* const_val = compare->getOperand(1);

        Value* var_ptr = nullptr;
        ConstantInt* K = nullptr;
        ICmpInst::Predicate p = compare->getPredicate();

        if (isa<LoadInst>(var_val) && isa<ConstantInt>(const_val)) {
            var_ptr = RangeAnalysisUtils::getUnderlyingAlloca(
                dyn_cast<LoadInst>(var_val)->getPointerOperand());
            K = dyn_cast<ConstantInt>(const_val);
        } else if (isa<LoadInst>(const_val) && isa<ConstantInt>(var_val)) {
            var_ptr = RangeAnalysisUtils::getUnderlyingAlloca(
                dyn_cast<LoadInst>(const_val)->getPointerOperand());
            K = dyn_cast<ConstantInt>(var_val);
            p = CmpInst::getSwappedPredicate(p);
        }

        if (!var_ptr || !K || !predOutState.varRanges.count(var_ptr)) return;

        Range oldRange = predOutState.varRanges[var_ptr];
        int64_t k_val = K->getSExtValue();
        bool isTrueBranch = (curr == br->getSuccessor(0));

        if (!isTrueBranch) {
            p = CmpInst::getInversePredicate(p);
        }

        Range constraintRange = FULL_RANGE;
        switch (p) {
            case ICmpInst::ICMP_SGT: constraintRange = Range(k_val + 1, INT_MAX); break;
            case ICmpInst::ICMP_SGE: constraintRange = Range(k_val, INT_MAX); break;
            case ICmpInst::ICMP_SLT: constraintRange = Range(INT_MIN, k_val - 1); break;
            case ICmpInst::ICMP_SLE: constraintRange = Range(INT_MIN, k_val); break;
            default: break;
        }
        predOutState.varRanges[var_ptr] = Range::intersectRange(oldRange, constraintRange);
    }
};

// Transfer function for individual instructions
class TransferFunction {
private:
    RangeEvaluator &evaluator;
public:
    TransferFunction(RangeEvaluator &eval) : evaluator(eval) {}

    void processInstruction(Instruction &I, BasicBlockState &state) {
        // Use the new integrated transfer function logic
        transferFunction(I, state);
    }

private:

    AllocaInst* getUnderlyingAlloca(Value *Ptr) {
        return RangeAnalysisUtils::getUnderlyingAlloca(Ptr);
    }

    Range eval(Value *Val, BasicBlockState &St) {
        return evaluator.getRangeForValue(Val);
    }

    // Integrated transfer function with the new logic
    void transferFunction(Instruction &I, BasicBlockState &St) {
        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
            St.varRanges[AI] = Range::FULL();
            return;
        }

        if (auto *SI = dyn_cast<StoreInst>(&I)) {
            Value *Val = SI->getValueOperand();
            Value *Ptr = SI->getPointerOperand();
            if (auto *AI = getUnderlyingAlloca(Ptr)) {
                Range R = eval(Val, St);
                St.varRanges[AI] = R;
            }
            return;
        }

        if (auto *LI = dyn_cast<LoadInst>(&I)) {
            Range R = eval(LI, St);
            St.varRanges[&I] = R;
            return;
        }

        if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
            Range L = eval(BO->getOperand(0), St);
            Range R = eval(BO->getOperand(1), St);
            Range Res;
            switch (BO->getOpcode()) {
                case Instruction::Add: Res = Range::addRange(L, R); break;
                case Instruction::Sub: Res = Range::subRange(L, R); break;
                case Instruction::Mul: Res = Range::mulRange(L, R); break;
                default: Res = Range::FULL(); break;
            }
            St.varRanges[&I] = Res;
            return;
        }

        if (auto *PN = dyn_cast<PHINode>(&I)) {
            Range Acc = Range::EMPTY();
            for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i) {
                Value *InV = PN->getIncomingValue(i);
                BasicBlockState tmp = St;

                Range R = eval(InV, tmp);
                Acc = (Acc == Range::EMPTY()) ? R : Range::mergeRange(Acc, R);
            }
            if (Acc == Range::EMPTY()) Acc = Range::FULL();
            St.varRanges[&I] = Acc;
            return;
        }

        // NOTE : Used LLM to bypass scanf and associated calls
        if (auto *CB = dyn_cast<CallBase>(&I)) {
            Function *Callee = CB->getCalledFunction();
            if (Callee && (Callee->getName() == "scanf" || Callee->getName() == "__isoc99_scanf")) {
                for (unsigned a = 0; a < CB->arg_size(); ++a) {
                    Value *Arg = CB->getArgOperand(a);
                    if (Arg->getType()->isPointerTy()) {
                        if (AllocaInst *AI = getUnderlyingAlloca(Arg)) {
                            St.varRanges[AI] = Range::FULL();
                        }
                    }
                }
            }
            if (!I.getType()->isVoidTy()) {
                St.varRanges[&I] = Range::FULL();
            }
            return;
        }
    }

    void processPHINode(PHINode *PN, BasicBlockState &state) {
        Range phirange = Range::EMPTY();
        for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i) {
            Value* val = PN->getIncomingValue(i);
            BasicBlock* pred = PN->getIncomingBlock(i);
            phirange = Range::mergeRange(phirange, evaluator.getRangeFromPred(val, pred));
        }
        state.varRanges[PN] = phirange;
    }

    void processStoreInst(StoreInst *SI, BasicBlockState &state) {
        Value *valToStore = SI->getValueOperand();
        Value *ptr = SI->getPointerOperand();
        if (AllocaInst* AI = RangeAnalysisUtils::getUnderlyingAlloca(ptr)) {
            state.varRanges[AI] = evaluator.getRangeForValue(valToStore);
        }
    }

    void processCallInst(CallInst *CI, BasicBlockState &state) {
        Function *calledFunc = CI->getCalledFunction();
        if (calledFunc) {
            for (unsigned i = 0; i < CI->arg_size(); ++i) {
                Value *arg = CI->getArgOperand(i);
                if (arg->getType()->isPointerTy()) {
                    if (AllocaInst* AI = RangeAnalysisUtils::getUnderlyingAlloca(arg)) {
                        state.varRanges[AI] = Range::FULL();
                    }
                }
            }
        }
    }

    void processBinaryOperator(BinaryOperator *BO, BasicBlockState &state) {
        Range r1 = evaluator.getRangeForValue(BO->getOperand(0));
        Range r2 = evaluator.getRangeForValue(BO->getOperand(1));
        switch (BO->getOpcode()) {
            case Instruction::Add: state.varRanges[BO] = Range::addRange(r1, r2); break;
            case Instruction::Sub: state.varRanges[BO] = Range::subRange(r1, r2); break;
            case Instruction::Mul: state.varRanges[BO] = Range::mulRange(r1, r2); break;
            default: state.varRanges[BO] = Range::FULL();
        }
    }
};

// Main dataflow analysis engine Kildall's algorithm
class DataFlowAnalysis {
private:
    Function &F;
    std::map<BasicBlock*, BasicBlockState> in, out;
    std::set<std::pair<BasicBlock*, BasicBlock*>> backEdges;

public:
    DataFlowAnalysis(Function &function) : F(function) {
        backEdges = RangeAnalysisUtils::findBackEdges(F);
    }

    void runAnalysis() {
        initializeWorklist();
        processWorklist();
    }

    BasicBlockState getFinalState() {
        BasicBlockState finalState;
        for (BasicBlock &bb : F) {
            if (succ_size(&bb) == 0) {
                finalState.meet(out[&bb]);
            }
        }
        return finalState;
    }

private:
    void initializeWorklist() {
        // Initialize all states
        for (BasicBlock &bb : F) {
            in[&bb] = BasicBlockState();
            out[&bb] = BasicBlockState();
        }
    }

    void processWorklist() {

        int globalIterations = 0;
        const int MAX_ITERATIONS = 1000;

        std::queue<BasicBlock*> wk;
        std::set<BasicBlock*> wk_set;


        for (BasicBlock &basicblock : F) {
            wk.push(&basicblock);
            wk_set.insert(&basicblock);
        }

        while (!wk.empty() && globalIterations < MAX_ITERATIONS) {
            globalIterations += 1;

            BasicBlock *bb = wk.front();
            wk.pop();
            wk_set.erase(bb);

            BasicBlockState newInState = computeInState(bb);
            in[bb] = newInState;

            BasicBlockState oldOutState = out[bb];
            BasicBlockState newOutState = computeOutState(bb, newInState);
            out[bb] = newOutState;

            if (oldOutState != newOutState) {
                addSuccessorsToWorklist(bb, wk, wk_set);
            }
        }
    }

    BasicBlockState computeInState(BasicBlock *bb) {
        BasicBlockState inState;

        if (bb == &F.getEntryBlock()) {
            // Initialize function arguments
            for (Argument &arg : F.args()) {
                inState.varRanges[&arg] = FULL_RANGE;
            }
        } else {
            // Meet from all predecessors
            for (BasicBlock *pred : predecessors(bb)) {
                BasicBlockState predOutState = out[pred];

                // Apply branch condition refinement
                BranchRefinement::applyBranchRefinement(pred, bb, predOutState);

                // Apply widening for back edges
                if (backEdges.count({pred, bb})) {
                    applyWidening(bb, predOutState);
                }

                inState.meet(predOutState);
            }
        }
        return inState;
    }

    BasicBlockState computeOutState(BasicBlock *bb, BasicBlockState inState) {
        RangeEvaluator evaluator(inState, out);
        TransferFunction transfer(evaluator);

        for (Instruction &I : *bb) {
            transfer.processInstruction(I, inState);
        }
        return inState;
    }

    void applyWidening(BasicBlock *bb, BasicBlockState &predOutState) {
        for (auto const& [val, oldRange] : in[bb].varRanges) {
            if (predOutState.varRanges.count(val)) {
                Range newRange = predOutState.varRanges[val];
                if (oldRange != newRange) {
                    predOutState.varRanges[val] = FULL_RANGE;
                }
            }
        }
    }

    void addSuccessorsToWorklist(BasicBlock *bb, std::queue<BasicBlock*> &wk,
                               std::set<BasicBlock*> &wk_set) {
        for (BasicBlock *succ : successors(bb)) {
            if (wk_set.find(succ) == wk_set.end()) {
                wk.push(succ);
                wk_set.insert(succ);
            }
        }
    }
};

// Results printer
class ResultsPrinter {
public:
    static void printResults(Function &F, const BasicBlockState &finalState) {
        errs() << "Function " << F.getName() << "\n";

        auto programVariables = RangeAnalysisUtils::collectProgramVariables(F);

        for (BasicBlock &bb : F) {
            if (succ_size(&bb) == 0) {
                for (auto entry: programVariables) {
                    StringRef variableName = entry.first;
                    Value *variableValue = entry.second;
                    errs() << variableName << " : ";

                    Range finalRange = FULL_RANGE;
                    if (finalState.varRanges.count(variableValue)) {
                       finalRange = finalState.varRanges.find(variableValue)->second;
                       if (finalRange == EMPTY_RANGE) {
                           finalRange = FULL_RANGE;
                       }
                    }
                    finalRange.print(errs());
                    errs() << "\n";
                }
                break; // Only print once
            }
        }
        errs() << "\n";
    }
};

// Main analysis entry point
void analyseFunction(Function &F) {
    if (F.isDeclaration()) return;

    DataFlowAnalysis analysis(F);
    analysis.runAnalysis();

    BasicBlockState finalState = analysis.getFinalState();
    ResultsPrinter::printResults(F, finalState);
}

class RangeAnalysisPass : public PassInfoMixin<RangeAnalysisPass> {
public:
    static bool isRequired() { return true; }
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        analyseFunction(F);
        return PreservedAnalyses::all();
    }
};

} // end anonymous namespace

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