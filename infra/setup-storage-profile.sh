#!/usr/bin/env bash
# Configure the AWS CLI profile `hackathon-storage` from SecretStash `hackathon-object-storage`
# without printing the key values. Run it yourself: `! bash infra/setup-storage-profile.sh`
set -euo pipefail
export PATH="/root/.nebius/bin:$PATH" AI_AGENT="${AI_AGENT:-claude-code}"

SECRET_ID=REDACTED_SECRET_ID   # hackathon-object-storage, group31-project
PROFILE=hackathon-storage

payload=$(nebius --profile hackathon mysterybox payload get --secret-id "$SECRET_ID" --format json)
field() { python3 -I -c 'import json,sys; d=json.loads(sys.stdin.read()); print(next(e["string_value"] for e in d["data"] if e["key"]==sys.argv[1]))' "$1" <<<"$payload"; }

aws configure set aws_access_key_id     "$(field AWS_ACCESS_KEY_ID)"     --profile "$PROFILE"
aws configure set aws_secret_access_key "$(field AWS_SECRET_ACCESS_KEY)" --profile "$PROFILE"
aws configure set region                "$(field AWS_DEFAULT_REGION)"    --profile "$PROFILE"
aws configure set endpoint_url          "$(field AWS_ENDPOINT_URL)"      --profile "$PROFILE"
aws configure set output json --profile "$PROFILE"
unset payload
chmod 600 ~/.aws/credentials ~/.aws/config

echo "Profile '$PROFILE' configured (key expires $(nebius --profile hackathon mysterybox payload get --secret-id "$SECRET_ID" --format json | python3 -I -c 'import json,sys; print(next(e["string_value"] for e in json.load(sys.stdin)["data"] if e["key"]=="ACCESS_KEY_EXPIRES_AT"))'))."
