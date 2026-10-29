---
description: Learn from this repository's Claude Code sessions from before TokenEconomizer was installed
argument-hint: "[--dry-run] [--limit N]"
---
The arguments I typed after the command are: "$ARGUMENTS" (none reads up to `backfill_sessions` earlier sessions, 30 by default).
Run this exact command with the Bash tool, then show me its output verbatim in a code block:

```
te="${CLAUDE_PLUGIN_ROOT}/scripts/te"; [ -x "$te" ] || te="${TOKENECONOMIZER_HOME:-$HOME/.tokeneconomizer}/bin/te"; "$te" --host claude --cwd "$(pwd)" backfill $ARGUMENTS
```

Never add `--limit` or remove `--dry-run` yourself. If it lists tools left out, say in one sentence that `/tokeneconomizer:toolprune restore` gives them back.
