#!/usr/bin/env bash
# Switch to the highest-performance nvpmodel mode (MAXN_SUPER on Orin Nano Super / JetPack 6.2,
# MAXN otherwise) and lock clocks. Looks the mode ID up from /etc/nvpmodel.conf instead of
# hard-coding it, because IDs differ between Orin modules and JetPack releases.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Run with sudo" >&2; exit 1; }

CONF=/etc/nvpmodel.conf
[ -f "$CONF" ] || { echo "$CONF not found" >&2; exit 1; }

pick() { grep -E "^< *POWER_MODEL .*NAME=$1( |>)" "$CONF" | sed -E 's/.*ID=([0-9]+).*/\1/' | head -n1; }
ID="$(pick MAXN_SUPER)"; NAME=MAXN_SUPER
if [ -z "$ID" ]; then ID="$(pick MAXN)"; NAME=MAXN; fi
[ -n "$ID" ] || { echo "No MAXN mode in $CONF; available:"; grep POWER_MODEL "$CONF"; exit 1; }

echo "Setting nvpmodel $NAME (id $ID)"
# nvpmodel may ask to reboot when changing between some modes; answer 'no' and report.
echo no | nvpmodel -m "$ID" || true
nvpmodel -q
jetson_clocks
echo "Done. If nvpmodel asked for a reboot, reboot and re-run this script."
