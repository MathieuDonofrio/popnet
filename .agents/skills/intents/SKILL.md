---
name: intents
description: Create, update, and review project intents describing problems and desired outcomes.
---

# Intents

## Workflow

1. Read relevant project context and existing intents.
2. Discuss the problem, desired outcome, and scope with the user.
3. Draft or revise the intent. Existing code is evidence, not authority.
4. Apply the quality criteria and report unresolved questions.

## Identity

- Use sequential IDs: `INT-0001`.
- Filename: `INT-0001-short-title.md`.
- Keep IDs stable and never reuse retired IDs.
- Check uniqueness and references before merge.

## Template

```markdown
---
id: INT-0001
title: <Desired outcome>
---

## Problem
<What problem or unmet need motivates this work?>

## Desired outcome
<What should become possible or improve, and for whom?>

## Scope
<What is included and explicitly excluded?>

## Success assessment
<How will we judge whether the intended outcome was achieved?>

## Assumptions
<Optional conditions this intent depends on.>

## Open questions
- <Optional unresolved decision.>
```

All metadata and sections are required except Assumptions and Open questions.

## Writing

Describe why the work matters and what outcome is wanted.
Avoid implementation choices unless they are genuine constraints.
Keep detailed behavioral obligations in requirements.

Make success assessable without inventing targets or promising
uncertain results. For research, distinguish obtaining credible
evidence from proving the hoped-for hypothesis.

## Quality

Local:
- Clear: The problem and desired outcome are understandable.
- Focused: Describes one coherent goal.
- Bounded: Scope and exclusions are explicit.
- Assessable: Defines a meaningful way to judge success.
- Implementation-independent: Remains meaningful across design choices.

Global:
- Consistent: Does not conflict with other intents or project constraints.
- Non-redundant: Does not unnecessarily duplicate another intent.
- Relevant: Addresses a justified project or stakeholder need.

## Validation

Assess the outcome using the stated success assessment.
Passing requirement checks alone does not establish that the intent
was achieved.