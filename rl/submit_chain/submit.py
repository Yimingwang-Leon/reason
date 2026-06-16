#!/usr/bin/env python3
"""create_submission。用法: submit.py --tag TC --desc "描述"
读 /tmp/u_{tag}_token.txt。"""
import argparse, time
from kaggle.api.kaggle_api_extended import KaggleApi, ApiCreateSubmissionRequest

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--desc", required=True)
ap.add_argument("--comp", default="nvidia-nemotron-model-reasoning-challenge")
a = ap.parse_args()

api = KaggleApi(); api.authenticate()
tok = open(f"/tmp/u_{a.tag}_token.txt").read().strip()
with api.build_kaggle_client() as kc:
    for i in range(20):
        try:
            r = ApiCreateSubmissionRequest()
            r.competition_name = a.comp
            r.blob_file_tokens = tok
            r.submission_description = a.desc
            res = kc.competitions.competition_api_client.create_submission(r)
            print("SUBMIT OK", getattr(res, "message", res)); break
        except Exception as e:
            print("retry", i, str(e)[:100]); time.sleep(8)
