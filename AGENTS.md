# Shared agent workflow

At the beginning of a task, read `PROJECT_STATUS_he.md` and `AGENT_CHAT_he.md`.
The assignment PDF remains the authoritative specification; the shared journal
records observations and proposals, not additional assignment requirements.

- Use `AGENT_CHAT_he.md` as the shared asynchronous conversation between agents
  and the user. Append dated, attributed entries with unique IDs and reply-to IDs.
- Distinguish verified findings, hypotheses, proposals, user decisions, and
  completed work. A proposal is not authorization to implement it.
- Before implementing an item, check the current code and branch: an older
  observation may already have been resolved. Record the reviewed commit.
- At handoff, append what changed, validation actually performed, limitations,
  and the next step. Reference earlier entry IDs when responding or resolving them.
- Preserve other agents' entries. Resolve Git conflicts by retaining both
  contributions; do not overwrite history or force-push to resolve journal edits.
- Keep `PROJECT_STATUS_he.md` as the concise current-state summary and
  `A4_worklog_he.md` as material for the submission analysis. Link to the journal
  rather than copying its full discussion into both files.
- Do not put API keys, credentials, or private chain-of-thought in the journal.
  Record conclusions and evidence only. Preserve developer feedback data.

The journal is synchronized through Git commits/pushes/pulls; it is not live
messaging or a lock. Agents on different branches must explicitly transfer
relevant journal commits. Do not assume another agent has read an uncommitted
or unpushed entry.
