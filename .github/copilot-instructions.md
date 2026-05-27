# Project Instructions — YANE Controlled Workflow

## Core Principle

Operate as a controlled workflow with explicit planning, implementation,
testing, and review phases. Treat the agent roles below as work phases, not as
separate autonomous agents.

Always:

* Understand before modifying.
* Analyze before implementing.
* Test before finalizing.
* Review before marking complete.
* Create a new branch for each new feature or non-trivial change. Branch names should be descriptive (`feature/...`, `fix/...`, `docs/...`).

Never:

* Blindly loop.
* Randomly patch errors.
* Refactor unrelated code.
* Continue implementation when the root cause is unclear.

---

# Workflow Phases

## 1. Planner Phase

Responsibilities:

* Read and understand the relevant code.
* Analyze the problem.
* Identify affected files.
* Create a minimal implementation strategy.
* Identify risks, dependencies, and required tests.

The Planner Phase MUST NOT modify code. For small, obvious changes, the plan
may be brief, but it still needs to identify the intended files and validation.

Required Output for non-trivial task execution:

```md
## Plan

### Goal
...

### Affected Files
- ...

### Implementation Strategy
1. ...
2. ...
3. ...

### Risks
- ...

### Required Tests
- ...

### Abort Conditions
- ...
```

---

## 2. Implementer Phase

Responsibilities:

* Implement ONLY the approved plan.
* Modify ONLY necessary files.
* Avoid unrelated refactors.
* Stop immediately if unexpected architecture problems appear.

Required Output for non-trivial task execution:

```md
## Implementation Report

### Modified Files
- ...

### Changes
- ...

### Reasoning
- ...

### Uncertainties
- ...
```

---

## 3. Test Phase

Responsibilities:

* Write missing tests.
* Update outdated tests.
* Execute the smallest relevant test suite first.
* Escalate to broader test suites when the change affects shared behavior,
  architecture, public APIs, checkpoints, or performance.
* Document all failures precisely.

Recommended Commands:

```bash
pytest -m ci
pytest
pytest --cov=yane --cov-report=term-missing
```

Testing policy:

* For narrow corrections, run targeted tests first, then `pytest -m ci` when
  practical.
* For new features, shared core behavior, or architecture changes, run
  `pytest -m ci` and the relevant full or coverage suite before completion.
* If a required command is too slow or unavailable, state that explicitly and
  explain which validation was run instead.

Required Output for non-trivial task execution:

```md
## Test Report

### Executed Commands
- ...

### Result
PASS / FAIL

### Failures
- ...

### Suspected Cause
- ...
```

---

## 4. Reviewer Phase

Responsibilities:

* Review architecture.
* Review code quality.
* Detect duplication.
* Detect edge cases.
* Validate test coverage.
* Decide whether the implementation is acceptable.

Required Output for non-trivial task execution:

```md
## Review Report

### Accepted
YES / NO

### Problems Found
- ...

### Risk Assessment
- LOW / MEDIUM / HIGH

### Decision
- APPROVE
- REPLAN REQUIRED
```

---

# Workflow State Machine

Always follow this order:

1. PLAN
2. IMPLEMENT
3. TEST
4. REVIEW

Decision Rules:

* If Review passes:

  * Mark task complete.

* If tests fail:

  * Run the Failure Analysis Protocol.
  * Return to PLAN with the new evidence.

* If Review does not approve:

  * Run the Failure Analysis Protocol.
  * Return to PLAN with the review findings.

A failed test or rejected review always invalidates at
least part of the current plan and requires a new plan before further code
changes.

Continue until the task is completed, the task is proven invalid, progress is
blocked, or no revised plan with new evidence is available.

---

# Failure Analysis Protocol

When something fails:

DO NOT blindly retry.

Generate:

```md
## Failure Analysis

### What Failed
...

### Reproduction Steps
...

### Likely Root Cause
...

### Is Original Plan Still Valid?
YES / NO

### Decision
- REPLAN
```

After failure analysis:

* Return to Planner Phase.
* Generate a new strategy based on the failure evidence.
* Restart workflow.

---

# Task Handling Rules

Follow the user's requested task.

If the request is unclear:

* Ask for clarification before implementing.
* Do not invent roadmap work.

If the user asks to choose work autonomously:

* Select exactly one coherent task.
* State why it was selected.
* Keep the scope narrow.

A task is ONLY complete if:

* Tests pass
* Review passes
* No unresolved critical issues remain

---

# Large File Rules

For files larger than 300 lines:

Before editing:

1. Analyze structure
2. Create modification plan
3. Identify extraction opportunities
4. Minimize risk surface

Never perform large rewrites without planning first.

---

# Test Design Requirements

Every new feature SHOULD include:

* At least one dedicated test
* Edge case coverage where appropriate

Testing standards:

* Prefer the existing local test style.
* Use `unittest.TestCase` where the surrounding tests already use it.
* Mark fast, reliable tests with `@pytest.mark.ci`.

Coverage targets:

* Global coverage should stay >= 90%.
* `core/` should stay >= 90%.
* `evolution/` should stay >= 90%.
* Do not lower meaningful coverage for changed behavior.

---

# Performance Rules

For performance-sensitive changes:

* Use profiling tools.
* Never optimize based on intuition alone.

Recommended tools:

* cProfile
* py-spy

Always provide:

* Before measurement
* After measurement
* Explanation of improvement

---

# Response Style

For full workflow reports, end with:

```md
**Next Step:** ...
```

For normal questions, reviews, or small changes, keep the response concise and
natural. Do not force boilerplate.

---

# Runtime Prompt

Use this prompt to start a larger task:

```md
Start the YANE Multi-Agent Workflow.

Goal:
[INSERT TASK HERE]

Requirements:
- Use Planner -> Implementer -> Tester -> Reviewer workflow
- Work on exactly one task
- Replan if root cause becomes unclear
- Run evidence-driven iterations until the task is complete or blocked
- No unrelated refactors
- Generate reports after each phase
- Stop if architecture assumptions become invalid
```
