---
name: architecture-decisions
description: Create, update, and review architectural decision records (ADRs).
---

# Architecture Decisions

## Workflow

1. Read relevant intents, requirements, specifications, existing ADRs, and implementation.
2. Identify the architectural decision and its motivation.
3. Draft or revise the ADR using the established format.
4. Verify the decision, rationale, and consequences against available evidence.
5. Apply the quality criteria and report unresolved questions.

Existing code is evidence of the chosen architecture, not necessarily its original rationale. Do not invent motivations or decisions.

## Identity

- Use sequential IDs: `ADR-0001`.
- Filename: `ADR-0001-short-title.md`.
- Keep IDs stable and never reuse retired IDs.
- Use YAML front matter with required `id`, `title`, and `status`.
- Allowed statuses: `proposed`, `accepted`, `superseded`.

## Template

```markdown
---
id: ADR-NNNN
title: <Decision title>
---

## Context
<What problem or constraint motivated the decision?>

## Decision
<What architectural approach was chosen?>

## Rationale
<Why was this approach preferred?>

## Consequences
<What benefits, limitations, and tradeoffs result?>
```

All metadata and sections are required.

Link related intents, requirements, or other ADRs when useful, without duplicating their content.

## Writing

- Document significant architectural decisions, not routine implementation details.
- Focus on the decision and its reasoning rather than describing the implementation.
- Keep records concise and self-contained.
- Avoid duplicating requirements, specifications, or in-code documentation.
- Distinguish implemented decisions from planned architecture.
- Record new decisions rather than silently rewriting superseded ones.

## Quality

- Justified: The decision addresses a meaningful problem or constraint.
- Clear: The chosen approach and rationale are unambiguous.
- Accurate: Claims are supported by confirmed decisions or evidence.
- Complete: Important consequences and tradeoffs are identified.
- Consistent: The decision agrees with relevant project artifacts.
- Maintainable: Avoid unnecessary details likely to become outdated.

## Verification

Review related intents, requirements, specifications, and implementation for consistency.

Report conflicts or unconfirmed rationale rather than making assumptions.

Do not change implementation behavior solely to match an ADR.