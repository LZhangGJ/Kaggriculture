# Submit your agent to the team arena

You do not need to run Pro8 or join the research workflow. Give your coding agent the task below and the file you want evaluated.

## Copy this task

```text
Submit my Kaggriculture agent to our private team arena for evaluation.

Agent file or folder: <PATH>
Display name: <NAME>
Version: <VERSION, or derive one from the file hash>

Read and follow this workflow:
https://github.com/LZhangGJ/Kaggriculture/blob/feature/pro8-arena/research/workflows/pro8_arena/SUBMIT_AGENT.md

Prepare the package and submit it through the arena website using my GitHub
sign-in. If you cannot use a browser, use the documented GitHub CLI route.
Use one route only. Preserve my agent's strategy and include its required files.

Return the submission receipt or issue URL, uploaded file SHA-256, version,
and verified status. Acceptance means queued for evaluation, not active on the
leaderboard. Tell me if login, repository access, or an unknown entry point
blocks you. Do not submit to Kaggle or change the active roster.
```

## 1. Prepare the file

For a standalone Python agent, upload the `.py` file. It must define `agent(observation, configuration)`, `agent(observation)`, or the supported `my_agent` equivalent. The website adds the arena interface wrapper.

For an agent with supporting files, use a ZIP with `main.py` and its dependencies at the root, or a Kaggle `.tar.gz` package. Include only files needed to run the agent. The website accepts files up to **64 MiB**. Logs belong on stderr. Packages cannot depend on network downloads or GPU access during a game.

For a custom executable or a program that already reads JSONL, include `arena.json` at the ZIP root:

```json
{"run":["python","main.py"]}
```

Optional `build` and `resources` fields follow [the arena manifest](../../../tools/arena/example-manifest.json). Commands are arrays of arguments, not shell strings. Build commands run in the sandbox. A JSONL agent reads one object containing `observation` and `configuration` per stdin line, and writes one action object per stdout line, flushing each response.

Do not add `arena.json` to an ordinary Python-callable package unless you also supply the appropriate wrapper: a custom run command bypasses the website's automatic callable bridge. The arena uses its own pinned referee contract; the repository's older RL setup instructions are not required for submission.

## 2. Upload through the website

Open [the private arena](https://kaggriculture-arena.tail0d430d.ts.net/), sign in with a GitHub account that has repository access, and choose **Submit agent**.

Enter the name, version and file. Keep the default Kaggle interface for a Python callable. Choose JSONL under Optional settings if the program already implements that protocol. Submit once and save the returned receipt. Inspect **Your submissions** for status before retrying an uncertain upload; do not send the same package through GitHub as well.

If sign-in requires the user's involvement, leave the prepared file ready and ask them to finish sign-in. Do not request their password, copy browser cookies, or create new credentials.

## 3. Confirm what happened

| Status | Meaning | Next step |
|---|---|---|
| Queued / registered / pending | Upload or intake accepted; evaluation is pending | Keep the receipt; check again later |
| Validation failed / rejected | Packaging, interface or sandbox check failed | Read the error; fix the package and submit a new version |
| Placement-rated | Initial evaluation finished | Review results; request coordinator review for active-roster admission |
| Active | The exact version is in the active roster | It enters newly scheduled arena rounds |
| Archived | The version is outside the active roster | Its historical results remain available |

An upload does not automatically replace an active agent or the best team agent. Placement runs outside the roster cap. Daily tournaments take priority, so placement can wait while a tournament runs. Do not promise an immediate rating or claim that a queued upload has passed evaluation.

Return a short receipt:

```text
Agent: <name and version>
Uploaded file: <path and SHA-256>
Receipt / issue: <identifier or URL>
Verified status: <status and time checked>
Arena agent ID: <if assigned; otherwise pending>
Next step: <waiting for evaluation / correction needed / admission review / active>
```

The uploaded file hash and the arena agent ID are different identifiers; packaging can change the stored archive hash.

## GitHub CLI route for coding agents without a browser

This uses the existing private-repository intake. It requires `gh` authentication and permission to create a release and issue in `LZhangGJ/Kaggriculture`. No code commit, PR, main-branch merge, or server access is needed. If the account lacks release permission, use the website instead.

1. Prepare a **JSONL-ready ZIP**, at most 256 MiB, and its SHA-256. Unlike the website, issue intake does not wrap a Python callable automatically. In a checkout of `feature/pro8-arena`, use `tools.arena.kaggle_export.pack(files, destination)` to add the existing callable bridge, then set `run` to `["python","_arena_bridge.py"]`. For an existing JSONL program, use its actual run command. Do not execute an untrusted package merely to assemble it.
2. Upload the ZIP to a uniquely named **draft release**, targeting `feature/pro8-arena`. Include the agent name, version and hash in the release notes. Save the release identifier so a retry can reuse it instead of creating another release.
3. Read the uploaded asset's **numeric REST API ID** from that release, then open one issue with a title beginning `[Arena]`. Write the body to a file and pass it with `--body-file`. Use the manifest below with real values.
4. Return the issue URL. The coordinator polls open issues from authorized collaborators; it does not require the issue-template Action to be merged into the default branch. Keep the issue open and the release asset available while intake is pending.

Commands below contain placeholders for the coding agent to replace. Use the user's shell's quoting rules; never embed the agent's source in a shell command.

```text
gh auth status
gh release create UNIQUE_TAG agent.zip --repo LZhangGJ/Kaggriculture --target feature/pro8-arena --draft --title "AGENT VERSION" --notes-file release-notes.txt
gh api repos/LZhangGJ/Kaggriculture/releases --paginate
gh issue create --repo LZhangGJ/Kaggriculture --title "[Arena] AGENT VERSION" --body-file arena-issue.md
```

Find the release by its unique `tag_name`, then the ZIP under `assets`; use `assets[].id`, not a download URL or a GraphQL node ID. Check an existing release and issue before repeating a command whose result was unclear. Do not overwrite an asset after its hash has entered an intake issue.

Write `arena-issue.md` in this form:

````text
### Agent manifest
```json
{
  "name": "My agent",
  "author": "YOUR_GITHUB_LOGIN",
  "version": "v1",
  "run": ["python", "_arena_bridge.py"],
  "archive_asset": 123456,
  "sha256": "REPLACE_WITH_ZIP_SHA256",
  "origin": {"kind": "team"}
}
```
````

The default run command above is for the callable bridge. Change it for an existing JSONL executable. The author must identify the submitting teammate; do not label a team agent as a public notebook. For intake status, inspect the arena's **Agent roster** and the published [results page](../../arena_live/README.md). An issue URL alone proves only that the request was filed.
