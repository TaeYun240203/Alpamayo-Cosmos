"""live.py: a judge-input stream (framed raw ROS1 messages) is scored like a bag; only labels come out."""
import io
import json

import numpy as np
import pandas as pd
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

import live
from scorer import score
from test_scorer import KM, cfg

STORE = get_typestore(Stores.ROS1_NOETIC)
STORE.register(get_types_from_msg("int32 unique_id\nint32 type\nstring name\nfloat64 heading\n"
                                  "geometry_msgs/Vector3 velocity\ngeometry_msgs/Vector3 acceleration\n"
                                  "geometry_msgs/Vector3 size\ngeometry_msgs/Vector3 position", "morai_msgs/msg/ObjectStatus"))
STORE.register(get_types_from_msg("std_msgs/Header header\nfloat64 global_offset_x\nfloat64 global_offset_y\n"
                                  "float64 global_offset_z\nmorai_msgs/ObjectStatus[] collision_object\n"
                                  "morai_msgs/ObjectStatus[] collision_objecta", "morai_msgs/msg/CollisionData"))
STORE.register(get_types_from_msg("std_msgs/Header header\nstring trafficLightIndex\nint16 trafficLightType\n"
                                  "int16 trafficLightStatus", "morai_msgs/msg/GetTrafficLightStatus"))
T = STORE.types
HEADER = T["std_msgs/msg/Header"](seq=0, stamp=T["builtin_interfaces/msg/Time"](sec=0, nanosec=0), frame_id="map")
V = T["geometry_msgs/msg/Vector3"]


def odom(x, speed):
    pose = T["geometry_msgs/msg/Pose"](position=T["geometry_msgs/msg/Point"](x=x, y=0.0, z=0.0),
                                       orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0))
    twist = T["geometry_msgs/msg/Twist"](linear=V(x=speed, y=0.0, z=0.0), angular=V(x=0.0, y=0.0, z=0.0))
    return T["nav_msgs/msg/Odometry"](header=HEADER, child_frame_id="base_link",
                                      pose=T["geometry_msgs/msg/PoseWithCovariance"](pose=pose, covariance=np.zeros(36)),
                                      twist=T["geometry_msgs/msg/TwistWithCovariance"](twist=twist, covariance=np.zeros(36)))


def collision(on, x=0.0):
    obj = T["morai_msgs/msg/ObjectStatus"](unique_id=7, type=2, name="", heading=0.0, velocity=V(x=0.0, y=0.0, z=0.0),
                                           acceleration=V(x=0.0, y=0.0, z=0.0), size=V(x=0.0, y=0.0, z=0.0),
                                           position=V(x=x, y=0.0, z=0.0))
    return T["morai_msgs/msg/CollisionData"](header=HEADER, global_offset_x=0.0, global_offset_y=0.0,
                                             global_offset_z=0.0, collision_object=[obj] if on else [], collision_objecta=[])


def stream(messages):
    out, index = io.BytesIO(), {}
    for t_ns, topic, msgtype, msg in messages:
        if topic not in index:
            index[topic] = len(index)
            head = json.dumps(dict(topic=topic, index=index[topic], type=msgtype.replace("/msg/", "/"),
                                   definition=STORE.generate_msgdef(msgtype)[0])).encode()
            out.write(live.FRAME.pack(live.HEADER, t_ns, len(head)) + head)
        raw = bytes(STORE.serialize_ros1(msg, msgtype))
        out.write(live.FRAME.pack(index[topic], t_ns, len(raw)) + raw)
    out.write(live.FRAME.pack(0, 0, 99)[:5])  # truncated tail from a killed producer is ignored
    return io.BytesIO(out.getvalue())


def light(index, status):
    return T["morai_msgs/msg/GetTrafficLightStatus"](header=HEADER, trafficLightIndex=index, trafficLightType=0,
                                                     trafficLightStatus=status)


def test_stream_is_scored_like_a_bag(tmp_path):
    c, msgs = cfg(), []
    for k in range(250):                                   # 5 s at 50 Hz, 5 m/s along the synthetic route from x=110
        t_ns = 10**9 + k * 20_000_000
        msgs.append((t_ns, c["topics"]["ego_status"], "nav_msgs/msg/Odometry", odom(110 + 0.1 * k, 5.0)))
        msgs.append((t_ns, c["topics"]["collision"], "morai_msgs/msg/CollisionData", collision(50 <= k < 60 or 62 <= k < 70, 110 + 0.1 * k + 3.0)))
        msgs.append((t_ns, c["topics"]["signal"], "morai_msgs/msg/GetTrafficLightStatus", light("L1", 1 if k < 125 else 16)))
    rows = live.read_stream(stream(msgs), c)
    assert sorted(rows) == ["collision", "ego_status", "signal"] and len(rows["ego_status"]) == 250
    labels, summary = score(live.episode_from_rows(rows, c), KM, c)
    assert summary["sources"]["pose"] == c["topics"]["ego_status"] and np.isclose(labels.speed_kph.iloc[10], 18.0)
    assert summary["counts"]["collision"] == 2 and summary["penalty_s"] == 30   # absent from 2 messages -> again
    assert "cannot be redone" in summary["note"]
    # The state of the next judged stop line's light is kept per frame (signal head label); raw messages are not.
    live.write_labels(labels, summary, tmp_path / "ep.labels.parquet")
    saved = pd.read_parquet(tmp_path / "ep.labels.parquet")
    assert saved.signal.iloc[10] == "red" and saved.signal.iloc[-1] == "green"
