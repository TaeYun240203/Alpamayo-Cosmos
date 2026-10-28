#!/usr/bin/env bash
# RunPod Secure Cloud on-demand price/stock for N GPUs, overall and per data center (public GraphQL, no API key).
# Usage: bash runpod_stock.sh [N=8] ["NVIDIA B200" "NVIDIA H100 80GB HBM3" ...]
N=${1:-8}; shift
GPUS=("$@"); [ ${#GPUS[@]} -eq 0 ] && GPUS=("NVIDIA B200" "NVIDIA B300 SXM6 AC" "NVIDIA H200" "NVIDIA H100 80GB HBM3")
q() { curl -s https://api.runpod.io/graphql -H 'content-type: application/json' -d "$1"; }
ids=$(printf '\\"%s\\",' "${GPUS[@]}"); ids="[${ids%,}]"
price() {  # $1 = extra lowestPrice input, e.g. dataCenterId:\"US-CA-2\"
  q "{\"query\":\"query { gpuTypes(input:{ids:$ids}) { id lowestPrice(input:{gpuCount:$N, secureCloud:true $1}) { uninterruptablePrice stockStatus } } }\"}" |
    python3 -c 'import json,sys; print("  ".join(g["id"].replace("NVIDIA ","")+"=$"+str(g["lowestPrice"]["uninterruptablePrice"])+"/"+str(g["lowestPrice"]["stockStatus"]) for g in json.load(sys.stdin)["data"]["gpuTypes"] if g["lowestPrice"]["uninterruptablePrice"]))'
}
echo "$(date '+%F %T') ${N}x Secure Cloud, any DC: $(price '')"
for dc in $(q '{"query":"query { dataCenters { id } }"}' | python3 -c 'import json,sys; print(" ".join(d["id"] for d in json.load(sys.stdin)["data"]["dataCenters"]))'); do
  r=$(price ", dataCenterId:\\\"$dc\\\""); [ -n "$r" ] && echo "  $dc: $r"
done
