#!/usr/bin/env bash
# Usage: image_status.sh <repository> <tag>, e.g. cosmonomic/pulsarlm-build <tag>.
# Prints "exists" or "missing" as GHCR answers 200 or 404 for the tag's
# manifest, and exits 1 on any other answer. Reads GITHUB_ACTOR and GITHUB_TOKEN.
set -euo pipefail

repository=$1
tag=$2
token=$(curl -fsS -u "$GITHUB_ACTOR:$GITHUB_TOKEN" "https://ghcr.io/token?scope=repository:$repository:pull" | jq -r .token)
status=$(curl -sS -o /dev/null -w '%{http_code}' --head \
  -H "Authorization: Bearer $token" \
  -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json" \
  "https://ghcr.io/v2/$repository/manifests/$tag")
case $status in
  200) echo exists ;;
  404) echo missing ;;
  *)
    echo "ghcr.io answered $status for $repository:$tag" >&2
    exit 1
    ;;
esac
