# Private arena website

Live site: https://kaggriculture-arena.tail0d430d.ts.net/

Use **Continue with GitHub**. The app requests public identity only. Current repository members can view the results without joining Tailscale. The sign-in page is public; results are protected.

## Access and data

- WRX90 refreshes the repository's collaborator IDs through GitHub every minute. It uses its existing GitHub CLI credentials; those credentials never go to the mini PC.
- The server checks the signed-in user's numeric GitHub ID against that list on every protected request. Removed members lose access after the next successful refresh, with a maximum five-minute stale-data window. If refresh fails for five minutes, the site denies access until it recovers.
- GitHub OAuth uses state, PKCE, and an exact HTTPS callback. The server discards user tokens after reading identity. Sessions use random server-side IDs, expire after eight hours, and end when the service restarts.
- Cookies have Secure, HttpOnly, SameSite=Lax, and the `__Host-` prefix. Responses disable caching and framing. The site loads no external scripts, fonts, or analytics.
- Only the checked `site/data.json` export reaches the web server. It does not serve agent files, seeds, source code, credentials, or directory listings.
- The dashboard refreshes every minute. Data advances when the arena report completes. Elo and each daily Bradley–Terry table retain their own update timestamps. Incomplete tournaments remain provisional.

## Services

The mini PC runs `arena-web.service` as a dedicated `arena-web` system user. Code lives in `/opt/arena-web`; root-owned configuration is in `/etc/arena-web/config.json`; read-only exports are in `/var/lib/arena-web`. The service has no Docker membership, no home-directory access, and no write access to those paths. It can write only to the upload spool at `/var/spool/arena-web`. Uploaded code never runs in the web service. Its limits are one CPU and 256 MiB RAM.

Waitress binds only to `127.0.0.1:8766`. Tailscale Funnel terminates HTTPS on port 443 and proxies to it. Do not expose port 8766 or serve the data directory directly. Funnel publishes the login endpoint; the application enforces repository membership.

WRX90 runs the separate `arena-web-sync.timer` every minute. Its service calls `python -m tools.arena.web_sync <arena-root>`. Private `web-host.json` supplies the SSH alias and incoming directory. The mini PC's root-owned `arena-web-import` helper validates and atomically installs the two JSON exports. The web process cannot edit the helper, membership list, or results.

No secrets belong in git. The OAuth app's client secret resides only in the root-owned mini PC configuration. For rotation, install the replacement there and restart the web service; existing sessions will end. Keep the OAuth app configured for public identity only, no wildcard callbacks, and no device flow.

## Checks and recovery

Run `python -m unittest discover -s tests/arena -q` after installing `tools/arena/requirements-web.txt`. The web tests cover anonymous access, forbidden file paths, nonmembers, membership revocation, stale authorization, state forgery, callback reuse, host validation, logout origin checks, and GitHub failures.

Inspect `systemctl status arena-web.service` on the mini PC and `systemctl --user status arena-web-sync.timer` on WRX90. The service does not log request URLs, cookies, or tokens. Check `/healthz` for process health; it contains no results. A failed GitHub check must never fall back to allowing everyone.

To take the website offline, run `sudo tailscale funnel --https=443 off` on the mini PC, then stop `arena-web.service`. This does not stop the continuous arena or daily tournaments.

The security checks are targeted tests, not an independent security audit. Host administrators remain trusted, and any result already downloaded by an authorized user cannot be recalled.

## Upload an agent

Sign in and choose **Submit agent**. Enter a name and upload a Python file, ZIP, or Kaggle `.tar.gz` package, up to 64 MiB. Version defaults to a file hash. The default interface is `agent(observation, configuration)` or `agent(observation)` in `main.py`. Choose JSONL under Optional settings for stdin/stdout agents. A ZIP may include `arena.json` containing `run`, optional `build`, and optional `resources` for a custom entry point; commands run only inside the sandbox.

**Your submissions** shows queued, registered, pending, placement-rated, active, or validation failure. Intake runs at the next coordinator tick; placement games receive the fair-share allocation after daily tournament priority work. Passing validation does not automatically replace a champion. No GitHub issue, release, or commit is needed.

The upload endpoint requires login and the correct Origin header. Each member can upload ten files per hour. Pending files share a 1 GiB queue limit. The mini PC stores uploads as inert bytes; WRX90 copies them, verifies their hash, checks archive safety, and submits them through the existing sandbox pipeline. Successful transfers allow the mini PC to discard its binary copy; receipts stay visible. Raw uploaded files and private receipts never go to GitHub. The member list and leaderboard exports remain read-only to the web service.
