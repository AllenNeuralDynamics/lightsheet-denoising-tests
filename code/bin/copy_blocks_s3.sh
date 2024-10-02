#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 UPLOAD_DIR S3_BUCKET S3_PREFIX"
    echo "Example: $0 /path/to/upload my-bucket my/prefix/"
    exit 1
}

if [[ $# -ne 3 ]]; then
    echo "Error: Incorrect number of arguments."
    usage
fi

UPLOAD_DIR="$1"
S3_BUCKET="$2"
S3_PREFIX="$3"

if ! command -v s5cmd &> /dev/null; then
    echo "Error: 's5cmd' is not installed. Please install it before running this script."
    exit 1
fi

if [[ ! -d "${UPLOAD_DIR}" ]]; then
    echo "Error: Directory '${UPLOAD_DIR}' does not exist."
    exit 1
fi

echo "Copying files from '${UPLOAD_DIR}' to 's3://${S3_BUCKET}/${S3_PREFIX}'..."
s5cmd cp "${UPLOAD_DIR}" "s3://${S3_BUCKET}/${S3_PREFIX}"

echo "Files successfully copied to 's3://${S3_BUCKET}/${S3_PREFIX}'"
