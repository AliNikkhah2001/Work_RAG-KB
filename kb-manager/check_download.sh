while true; do
  size=$(du -m ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/blobs/* | awk '{print $1}')
  echo "Current size: ${size}MB"
  sleep 30
done
