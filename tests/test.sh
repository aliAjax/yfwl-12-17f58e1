#!/bin/bash
set -euo pipefail
umask 077

LOG_DIR=/logs/verifier
REWARD_FILE="$LOG_DIR/reward.txt"
STATUS_FILE="$LOG_DIR/verification-status.txt"
TEST_LOG="$LOG_DIR/test-stdout.txt"
TRUSTED_PYTHON=/usr/local/bin/python3
TRUSTED_TIMEOUT=/usr/bin/timeout
SITE_PACKAGES=/usr/lib/python3/dist-packages
INNER_TIMEOUT_SEC=150
TEST_RUNNER_UID=10002
TEST_RUNNER_GID=10002
VERIFIER_UID=$(id -u)
VERIFIER_GID=$(id -g)
PYTEST_GROUP_PID=""
completed=0

mkdir -p "$LOG_DIR"
chown "$VERIFIER_UID:$VERIFIER_GID" "$LOG_DIR" 2>/dev/null || true
chmod 0755 "$LOG_DIR"

write_reward() {
  local value="$1"
  chmod 0644 "$REWARD_FILE" 2>/dev/null || true
  printf '%s\n' "$value" > "$REWARD_FILE"
  chmod 0444 "$REWARD_FILE"
}

write_status() {
  chmod 0644 "$STATUS_FILE" 2>/dev/null || true
  printf '%s\n' "$1" > "$STATUS_FILE"
  chmod 0444 "$STATUS_FILE"
}

cleanup_runner() {
  if [ -n "$PYTEST_GROUP_PID" ]; then
    /bin/kill -KILL -- "-$PYTEST_GROUP_PID" 2>/dev/null || true
  fi
  if [ "$VERIFIER_UID" -eq 0 ]; then
    /usr/bin/pkill -KILL -u "$TEST_RUNNER_UID" 2>/dev/null || true
  fi
}

on_exit() {
  cleanup_runner
  if [ "$completed" -ne 1 ]; then
    write_status "INCOMPLETE: verifier exited before pytest completion"
    write_reward 0.000
  fi
}

on_signal() {
  cleanup_runner
  write_status "INCOMPLETE: verifier interrupted before pytest completion"
  write_reward 0.000
  completed=1
  exit 0
}

trap on_exit EXIT
trap on_signal INT TERM HUP

write_reward 0.000
write_status "INCOMPLETE: pytest has not completed"

REPORT_DIR=$(mktemp -d "$LOG_DIR/.report.XXXXXX")
chmod 0700 "$REPORT_DIR"
REPORT="$REPORT_DIR/pytest-results.xml"
PROVENANCE="$REPORT_DIR/provenance.json"
rm -f "$REPORT" "$PROVENANCE" "$TEST_LOG" "$LOG_DIR/scoring-report.json"
exec 9>"$REPORT"
chmod 0600 "$REPORT"

set +e
env -i \
  HOME=/tmp \
  PATH=/usr/local/bin:/usr/bin:/bin \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH="$SITE_PACKAGES:/tests" \
  /usr/bin/setsid \
  "$TRUSTED_TIMEOUT" --signal=TERM --kill-after=10s "${INNER_TIMEOUT_SEC}s" \
  "$TRUSTED_PYTHON" -S -s -P -B -u -m pytest -p no:cacheprovider -rA \
  --junitxml="$REPORT" /tests/test_outputs.py > "$TEST_LOG" 2>&1 &
PYTEST_GROUP_PID=$!
wait "$PYTEST_GROUP_PID"
RC=$?
set -e
exec 9>&-
cleanup_runner
cat "$TEST_LOG" 2>/dev/null || true

if [ "$RC" -ne 0 ] && [ "$RC" -ne 1 ]; then
  case "$RC" in
    124|125|126|127|137|143)
      write_status "INCOMPLETE: pytest did not complete; runner exited with code $RC"
      ;;
    *)
      write_status "INCOMPLETE: pytest exited before a scoreable report with code $RC"
      ;;
  esac
  write_reward 0.000
elif [ ! -s "$REPORT" ]; then
  write_status "INCOMPLETE: pytest completion report missing"
  write_reward 0.000
else
  if env -i \
    HOME=/tmp \
    PATH=/usr/local/bin:/usr/bin:/bin \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$SITE_PACKAGES:/tests" \
    "$TRUSTED_PYTHON" -S -s -P -B -u - \
    "$REPORT" "$PROVENANCE" "$RC" "$VERIFIER_UID" "$TEST_RUNNER_UID" \
    /tests/test_outputs.py /tests/scoring.json <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

report = Path(sys.argv[1])
provenance = Path(sys.argv[2])
pytest_rc = int(sys.argv[3])
verifier_uid = int(sys.argv[4])
isolated_uid = int(sys.argv[5])
tests_path = Path(sys.argv[6])
config_path = Path(sys.argv[7])

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()

report_stat = report.stat()
active_uid = os.getuid()
candidate_uid = isolated_uid if active_uid == 0 else active_uid
if (
    verifier_uid != active_uid
    or report_stat.st_uid != active_uid
    or stat.S_IMODE(report_stat.st_mode) != 0o600
):
    raise SystemExit("JUnit report ownership or mode is invalid")
payload = {
    "schema_version": 1,
    "pytest_returncode": pytest_rc,
    "pytest_uid": active_uid,
    "verifier_uid": verifier_uid,
    "candidate_uid": candidate_uid,
    "report_sha256": digest(report),
    "report_size": report_stat.st_size,
    "tests_sha256": digest(tests_path),
    "config_sha256": digest(config_path),
}
provenance.write_text(
    json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.chmod(provenance, 0o600)
PY
  then
    if env -i \
      HOME=/tmp \
      PATH=/usr/local/bin:/usr/bin:/bin \
      LANG=C.UTF-8 \
      LC_ALL=C.UTF-8 \
      PYTHONNOUSERSITE=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$SITE_PACKAGES:/tests" \
      "$TRUSTED_PYTHON" -S -s -P -B /tests/score.py \
      --config /tests/scoring.json \
      --junit "$REPORT" \
      --provenance "$PROVENANCE" \
      --report-out "$LOG_DIR/scoring-report.json" \
      --reward-out "$REWARD_FILE"; then
      REWARD=$(cat "$REWARD_FILE")
      case "$REWARD" in
        0.[0-9][0-9][0-9]|1.000) ;;
        *)
          write_status "INCOMPLETE: weighted scorer produced invalid reward"
          write_reward 0.000
          completed=1
          exit 0
          ;;
      esac
      chmod 0444 "$REWARD_FILE"
      write_status "COMPLETED: weighted scoring finished with reward=$REWARD"
    else
      write_status "INCOMPLETE: weighted scoring rejected the report"
      write_reward 0.000
    fi
  else
    write_status "INCOMPLETE: verifier provenance generation failed"
    write_reward 0.000
  fi
fi

completed=1
exit 0
