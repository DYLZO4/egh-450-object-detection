import cv2
import numpy as np
import os
import argparse
from rosbags.highlevel import AnyReader
from pathlib import Path

# ---- CONFIG (defaults, can be overridden via CLI args below) ----
topic = '/depthai_node/image/compressed'   # confirm this matches exactly (see list_topics below)
save_every_n = 2# 60fps / 30 = ~2fps sampling, good starting point for labeling
img_format = 'jpg'         # match source since it's already compressed/JPEG
jpg_quality = 95
# ----------------

parser = argparse.ArgumentParser(description='Extract frames from a rosbag topic to image files (no ROS install required).')
parser.add_argument('bag_path', help='Path to the .bag file (or bag directory, for ROS2-style bags)')
parser.add_argument('out_dir', help='Directory to save extracted frames to')
parser.add_argument('--topic', default=topic, help=f'Image topic to extract (default: {topic})')
parser.add_argument('--every', type=int, default=save_every_n, help=f'Save 1 out of every N frames (default: {save_every_n})')
parser.add_argument('--list-topics', action='store_true', help='List all topics in the bag and exit (no extraction)')
args = parser.parse_args()

bag_path = args.bag_path
out_dir = args.out_dir
topic = args.topic
save_every_n = args.every

encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if img_format == 'jpg' else []

with AnyReader([Path(bag_path)]) as reader:
    if args.list_topics:
        print('Topics in bag:')
        for conn in reader.connections:
            print(f'  {conn.topic}  ({conn.msgtype})  [{conn.msgcount} msgs]')
        raise SystemExit(0)

    os.makedirs(out_dir, exist_ok=True)

    connections = [c for c in reader.connections if c.topic == topic]
    if not connections:
        available = ', '.join(c.topic for c in reader.connections)
        raise SystemExit(f'Topic "{topic}" not found in bag. Available topics: {available}')

    is_compressed = 'Compressed' in connections[0].msgtype

    count = 0
    saved = 0

    for conn, timestamp, rawdata in reader.messages(connections=connections):
        if count % save_every_n == 0:
            msg = reader.deserialize(rawdata, conn.msgtype)

            if is_compressed:
                np_arr = np.frombuffer(bytes(msg.data), np.uint8)
                cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                # raw sensor_msgs/Image: reshape based on encoding/height/width
                np_arr = np.frombuffer(bytes(msg.data), np.uint8)
                cv_img = np_arr.reshape(msg.height, msg.width, -1)
                if cv_img.shape[2] == 1:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
                elif msg.encoding.lower() in ('rgb8',):
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

            if cv_img is None:
                print(f'Warning: failed to decode frame at count {count}, skipping.')
                count += 1
                continue

            filename = os.path.join(out_dir, f'frame_{saved:06d}_{timestamp}.{img_format}')
            cv2.imwrite(filename, cv_img, encode_params)
            saved += 1

        count += 1

print(f'Done. Read {count} messages on "{topic}", saved {saved} frames to "{out_dir}".')
