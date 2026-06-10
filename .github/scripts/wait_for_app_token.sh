#!/usr/bin/env bash
# Wait for a freshly minted GitHub App installation token to go live.
#
# On private repos, actions/create-github-app-token can return a token before
# its installation auth has propagated, so the first authenticated call may
# transiently fail with "Repository not found" / "Not Found" / 404 (the same
# race that broke the perimeter-banner git fetch). Probe a cheap read
# (GET /repos/{owner}/{repo}) with bounded exponential backoff so the caller's
# first REAL API call runs only after the token is recognized. Fails closed
# (non-zero) if the token never goes live, so callers never proceed on a token
# that can't authenticate.
#
# Required env:
#   GH_TOKEN            -- the minted App installation token (NOT the default
#                          GITHUB_TOKEN, which is already live at job start).
#   GITHUB_REPOSITORY   -- owner/repo (auto-set by GitHub Actions).
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN (minted App token) must be set}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"

attempt=1
max_attempts=4
delay=5

while true; do
  if output=$(gh api "repos/${GITHUB_REPOSITORY}" --silent 2>&1); then
    echo "App token live for ${GITHUB_REPOSITORY} (attempt ${attempt})."
    exit 0
  fi
  status=$?
  printf '%s\n' "$output" >&2

  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "::error::App token never became live for ${GITHUB_REPOSITORY} after ${attempt} attempts"
    exit "$status"
  fi

  # Retry only the transient post-mint propagation signatures; a non-transient
  # error (bad key, app uninstalled, outage) fails immediately. Match with a
  # here-string, not `printf | grep -q`: grep -q exits on first match and would
  # SIGPIPE printf under pipefail, misclassifying a matching (transient) error
  # as non-retryable on large output.
  if ! grep -qiE \
      'Repository not found|Not Found|HTTP (401|403|404|429|5[0-9][0-9])|rate limit|timed? out|TLS|connection reset|Could not resolve|Failed to connect|temporarily unavailable|remote end hung up' \
      <<<"$output"; then
    echo "::error::App token probe failed with a non-retryable error"
    exit "$status"
  fi

  echo "::warning::App token not live yet (attempt ${attempt}/${max_attempts}); retrying in ${delay}s"
  sleep "$delay"
  attempt=$((attempt + 1))
  delay=$((delay * 2))
done
