#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 S3_BUCKET S3_PREFIX"
    echo "Example: $0 my-bucket my/prefix/"
    exit 1
}

if [[ $# -ne 2 ]]; then
    echo "Error: Incorrect number of arguments."
    usage
fi

S3_BUCKET="$1"
S3_PREFIX="$2"

if ! command -v s5cmd &> /dev/null; then
    echo "Error: 's5cmd' is not installed. Please install it before running this script."
    exit 1
fi

if [[ ! -d "/results/" ]]; then
    echo "Error: Directory '/results/' does not exist."
    exit 1
fi

echo "Copying files from '/results/' to 's3://${S3_BUCKET}/${S3_PREFIX}'..."
s5cmd cp "/results/" "s3://${S3_BUCKET}/${S3_PREFIX}"

echo "Files successfully copied to 's3://${S3_BUCKET}/${S3_PREFIX}'"
