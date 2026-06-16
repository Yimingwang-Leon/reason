#!/bin/sh
# 分块断点续传上传。用法: curl_up.sh <zip> <tag> [chunk_mb]
# 读 /tmp/u_{tag}_url.txt;临时块文件 /tmp/u_{tag}_chunk.bin(tag 隔离防并发踩踏)
FILE=$1; TAG=$2; CHUNK_MB=${3:-8}
[ -z "$FILE" ] || [ -z "$TAG" ] && { echo "usage: curl_up.sh <zip> <tag> [chunk_mb]"; exit 1; }
SIZE=$(stat -f%z "$FILE")
URL=$(cat /tmp/u_${TAG}_url.txt)
CHUNK=$((CHUNK_MB*1024*1024))
OFF=0; FAILS=0
while [ $OFF -lt $SIZE ]; do
  END=$((OFF+CHUNK)); [ $END -gt $SIZE ] && END=$SIZE
  LEN=$((END-OFF))
  dd if="$FILE" bs=1m iseek=$((OFF/1048576)) count=$(( (LEN+1048575)/1048576 )) 2>/dev/null | head -c $LEN > /tmp/u_${TAG}_chunk.bin
  CODE=$(curl -s -o /tmp/u_${TAG}_resp.txt -w "%{http_code}" --max-time 120 --retry 0 -X PUT \
    -H "Content-Range: bytes ${OFF}-$((END-1))/${SIZE}" --data-binary @/tmp/u_${TAG}_chunk.bin "$URL")
  if [ "$CODE" = "200" ] || [ "$CODE" = "201" ]; then echo "UPLOAD_COMPLETE"; exit 0; fi
  if [ "$CODE" = "308" ]; then OFF=$END; FAILS=0
    [ $((OFF % (64*1024*1024))) -eq 0 ] && echo "$((OFF/1048576))MB/$((SIZE/1048576))MB"
    continue
  fi
  FAILS=$((FAILS+1)); [ $FAILS -gt 400 ] && { echo "GIVE_UP code=$CODE"; exit 2; }
  sleep $((FAILS<6 ? FAILS : 6)) 2>/dev/null || sleep 5
  # 查服务端已收偏移
  Q=$(curl -s -o /dev/null -w "%{http_code}" --max-time 60 -X PUT -H "Content-Range: bytes */${SIZE}" -H "Content-Length: 0" "$URL")
  RANGE=$(curl -sI --max-time 60 -X PUT -H "Content-Range: bytes */${SIZE}" -H "Content-Length: 0" "$URL" 2>/dev/null | grep -i "^range:" | grep -oE "[0-9]+$")
  [ -n "$RANGE" ] && OFF=$((RANGE+1))
  [ "$Q" = "200" ] || [ "$Q" = "201" ] && { echo "UPLOAD_COMPLETE"; exit 0; }
done
echo "UPLOAD_COMPLETE"; exit 0
