#!/usr/bin/env python3
"""创建 kaggle 提交上传 session。用法: mk_session.py --zip /tmp/x.zip --tag TC
落盘 /tmp/u_{tag}_url.txt + /tmp/u_{tag}_token.txt(tag 隔离防并发踩踏)。"""
import argparse, os
from kaggle.api.kaggle_api_extended import KaggleApi, ApiStartSubmissionUploadRequest

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--comp", default="nvidia-nemotron-model-reasoning-challenge")
a = ap.parse_args()

api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kc:
    req = ApiStartSubmissionUploadRequest()
    req.competition_name = a.comp
    req.file_name = os.path.basename(a.zip)
    req.content_length = os.path.getsize(a.zip)
    req.last_modified_epoch_seconds = int(os.path.getmtime(a.zip))
    r = kc.competitions.competition_api_client.start_submission_upload(req)
    open(f"/tmp/u_{a.tag}_url.txt", "w").write(r.create_url)
    open(f"/tmp/u_{a.tag}_token.txt", "w").write(r.token)
print("SESSION OK", a.tag)
