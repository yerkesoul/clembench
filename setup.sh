#!/bin/bash
#python3 -m venv venv
#source venv/bin/activate
pip3 install -r requirements.txt

cat <<EOF >key_default.json
{
  "openai": {
    "organisation": "",
    "api_key": ""
  },
  "anthropic": {
    "api_key": ""
  },
  "alephalpha": {
    "api_key": ""
  },
  "huggingface": {
    "api_key": ""
  }
}
EOF

cat <<EOF >key_gpt4.json
{
  "openai": {
    "organisation": "",
    "api_key": ""
  },
  "anthropic": {
    "api_key": ""
  },
  "alephalpha": {
    "api_key": ""
  },
  "huggingface": {
    "api_key": ""
  }
}
EOF

echo "Please add the keys to the key files (key_default.json, key_gpt4.json) manually."
