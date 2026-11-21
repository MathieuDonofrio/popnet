---
name: tool-creation
description: Create or update repository developer tools
---

# Tool creation

- Keep tools repository-agnostic. Take the checkout, input files, project name, and publishing settings from arguments or repository metadata as needed. Put project-specific values and URLs in the calling workflow or documentation.
- Use a small CLI with clear errors and document how to run it. Provide structured output when another program needs it. For an offline report, embed data safely.
- Keep each tool and its dependencies, tests, and usage guide under its own directory in `tools/`. Verify the normal entry point and any CI or report integration you changed.
- Prefer python tools with a requirements.txt for all dependencies, when deviating from this ask the user and justify the deviation.
