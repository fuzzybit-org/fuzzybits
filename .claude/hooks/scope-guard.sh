#!/usr/bin/env bash
# PreToolUse hook (Bash): block the GitHub and git commands that docs/planning.md §8
# reserves for the maintainer. Exit 2 blocks the call and shows stderr to the agent.
#
# Blocked:
#   - merging PRs (gh pr merge, gh api .../merge);
#   - pushing to develop or main, and force pushes;
#   - applying approval, status:ready or interrupt labels;
#   - creating, editing or deleting labels and milestones;
#   - setting a milestone on an issue or PR.
# The user can still run any of these themselves.
exec python3 -c '
import json, re, shlex, sys

RESERVED = {"scope-exception", "size-exception", "surface-approved", "status:ready",
            "wrong-answer", "security", "release-blocker"}

def block(what):
    sys.stderr.write(f"scope-guard: {what} is reserved for the maintainer (docs/planning.md §8). "
                     "Ask the user to do it.\n")
    sys.exit(2)

try:
    cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
except Exception:
    sys.exit(0)
if not cmd:
    sys.exit(0)

# Split on shell separators, then tokenise each simple command.
for part in re.split(r"(?:&&|\|\||[;|&\n])", cmd):
    try:
        t = shlex.split(part)
    except ValueError:
        t = part.split()
    while t and ("=" in t[0] and not t[0].startswith("-")):   # leading VAR=value
        t = t[1:]
    if not t:
        continue
    if t[0] == "gh" and len(t) >= 3:
        sub = t[1:3]
        if sub == ["pr", "merge"]:
            block("merging a PR")
        if t[1] == "label" and t[2] in ("create", "edit", "delete", "clone"):
            block("changing labels")
        if t[1] in ("issue", "pr") and t[2] in ("create", "edit"):
            if any(a in ("--milestone", "-m") or a.startswith("--milestone=") for a in t):
                block("setting a milestone")
            labels = []
            for i, a in enumerate(t):
                if a in ("--label", "--add-label", "-l") and i + 1 < len(t):
                    labels += t[i + 1].split(",")
                elif a.startswith(("--label=", "--add-label=")):
                    labels += a.split("=", 1)[1].split(",")
            hit = RESERVED.intersection(l.strip() for l in labels)
            if hit:
                block(f"applying the label {sorted(hit)[0]!r}")
        if t[1] == "api":
            joined = " ".join(t)
            method = next((t[i + 1].upper() for i, a in enumerate(t) if a in ("-X", "--method") and i + 1 < len(t)), "GET")
            if re.search(r"pulls/\d+/merge", joined) and method in ("PUT", "POST"):
                block("merging a PR")
            if re.search(r"/(milestones|labels)\b", joined) and method in ("POST", "PATCH", "PUT", "DELETE"):
                if "/issues/" not in joined or RESERVED.intersection(re.findall(r"[\w:-]+", joined)) or "milestones" in joined:
                    block("changing milestones or labels")
    if t[0] == "git" and len(t) >= 2 and t[1] == "push":
        args = t[2:]
        if any(a in ("-f", "--force") or a.startswith("--force") for a in args):
            block("force-pushing")
        refs = [a for a in args if not a.startswith("-")][1:]  # after the remote
        if any(r.split(":")[-1].removeprefix("refs/heads/") in ("develop", "main") for r in refs):
            block("pushing to develop or main")
sys.exit(0)
'
