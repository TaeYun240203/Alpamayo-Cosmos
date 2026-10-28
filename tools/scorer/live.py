"""Score one episode live: the collect skill's live-buffer.py streams the judge-only inputs from the stack to stdin.
They are held in memory, scored with scorer.score at end of stream, and only the labels parquet (summary and events
in its metadata) is written. The raw judge inputs are never stored.

    python live.py --topics                 # topics live-buffer.py must subscribe to (from thresholds.yaml)
    python live.py -o EP.labels.parquet < stream

Stream frames: [u16 index][u64 receive ns][u32 length][payload]; index 0xFFFF carries a JSON header
{topic, index, type, definition} (the ROS1 connection header) before that topic's first message.
"""
import argparse
import json
import os
import struct
import sys

from rosbags.typesys import Stores, get_typestore, get_types_from_msg

from bag import EXTRACT, episode_from_rows
from scorer import HERE, KMap, load_config, score, write_labels

FRAME = struct.Struct("<HQI")
HEADER = 0xFFFF


def read_stream(stream, cfg):
    """-> rows for bag.episode_from_rows; payloads are deserialized and reduced right away."""
    store = get_typestore(Stores.ROS1_NOETIC)
    key_of = {v: k for k, v in cfg["topics"].items() if v}
    rows, topic_of = {}, {}
    while True:
        head = stream.read(FRAME.size)
        if len(head) < FRAME.size:
            return rows
        index, t_ns, size = FRAME.unpack(head)
        payload = stream.read(size)
        if len(payload) < size:  # truncated by a killed producer: keep what arrived complete
            return rows
        if index == HEADER:
            h = json.loads(payload)
            pkg, name = h["type"].split("/")
            msgtype = f"{pkg}/msg/{name}"
            store.register({k: v for k, v in get_types_from_msg(h["definition"], msgtype).items()
                            if k not in store.fielddefs})
            topic_of[h["index"]] = (key_of.get(h["topic"]), msgtype)
            continue
        key, msgtype = topic_of[index]
        if key:
            rows.setdefault(key, []).append((t_ns, EXTRACT[key](store.deserialize_ros1(payload, msgtype))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=HERE / "thresholds.yaml")
    ap.add_argument("-o", "--out", help="labels parquet")
    ap.add_argument("--topics", action="store_true", help="print the topics to stream and exit")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.topics:
        print(" ".join(v for k, v in cfg["topics"].items() if v and k != "ego_gt"))
        return
    if not args.out:
        ap.error("-o is required")
    labels, summary = score(episode_from_rows(read_stream(sys.stdin.buffer, cfg), cfg), KMap.from_files(cfg), cfg)
    write_labels(labels, summary, args.out + ".tmp")
    os.replace(args.out + ".tmp", args.out)  # the mover ships only complete files
    brief = {k: v for k, v in summary.items() if k != "events"}
    print(json.dumps(brief, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
