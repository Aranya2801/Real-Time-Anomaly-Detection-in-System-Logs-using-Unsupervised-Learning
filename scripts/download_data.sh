#!/usr/bin/env bash
# Fetches the datasets used in the experiments into ./data
#
#   ./scripts/download_data.sh          # small samples (~10 MB, enough to reproduce results/)
#   ./scripts/download_data.sh --full   # also full HDFS_v1 (~1.5 GB unzipped) from Zenodo
set -euo pipefail
mkdir -p data && cd data

if [ ! -d loghub ]; then
  # 2k-line samples with ground-truth templates for 16 systems (parser benchmark)
  git clone --depth 1 https://github.com/logpai/loghub.git
fi

if [ ! -f HDFS/anomaly_label.csv ]; then
  # 100k-line HDFS sample + block labels, from the LogPAI loglizer repo
  tmp=$(mktemp -d)
  git clone --depth 1 https://github.com/logpai/loglizer.git "$tmp/loglizer"
  mkdir -p HDFS
  cp "$tmp/loglizer/data/HDFS/HDFS_100k.log_structured.csv" "$tmp/loglizer/data/HDFS/anomaly_label.csv" HDFS/
  rm -rf "$tmp"
fi

if [ "${1:-}" = "--full" ]; then
  curl -L -o HDFS_v1.zip "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
  unzip -o HDFS_v1.zip -d HDFS_v1
  echo "full HDFS: python experiments/hdfs_benchmark.py --logs data/HDFS_v1/HDFS.log --labels data/HDFS_v1/preprocessed/anomaly_label.csv"
fi
echo "done"
