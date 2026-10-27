---
name: requirements
description: Create, update, and review requirements linked to project intents.
---

# Requirements

## Workflow

1. Read relevant intents, specifications, and existing requirements.
2. Draft or revise the requirement. Existing code is evidence, not authority.
3. Apply the quality criteria and report unresolved questions.

## Identity

- Use sequential IDs: `REQ-0001`.
- Filename: `REQ-0001-short-title.md`.
- Keep IDs stable and never reuse retired IDs.
- Check uniqueness and references before merge.

## Template

```markdown
---
id: REQ-0001
title: <Required behavior or constraint>
intents:
  - INT-0001
specifications: []
---

## Obligation
<The component shall ...>

## Rationale
<Optional justification.>

## Open questions
- <Optional unresolved decision.>
```

All metadata and the obligation are required.
Specifications are repository-relative files or sections that define
part of the required behavior. Use `[]` when the obligation is self-contained.

## Writing

Keep documentation concise. State obligations in requirements and details in specifications; avoid duplication.

Use "shall" for obligations. Prefer EARS-style patterns when useful:

- The component shall ...
- When <event>, the component shall ...
- While <state>, the component shall ...
- If <unwanted condition>, then the component shall ...
- Where <feature is supported>, the component shall ...

Other precise forms are allowed. Use equations, tables, or references
when clearer. Define necessary conditions, bounds, and tolerances.
Keep rationale separate and do not invent missing decisions.

## Quality

Local:
- Atomic: One independently assessable obligation.
- Unambiguous: One reasonable interpretation with its references.
- Verifiable: Compliance can be assessed with evidence.
- Sufficient: Necessary contract details are defined or referenced.
- Implementation-independent: Design choices require justification.

Global:
- Consistent: No conflicting obligations.
- Non-redundant: No unnecessary duplication.
- Feasible: Achievable within project constraints.
- Necessary: Supports a linked intent or justified constraint.

Retain derived requirements only when managing them separately adds value.

## Verification

Tests and other checks reference requirement IDs.
Requirements define compliance; checks assess it; results record evidence.