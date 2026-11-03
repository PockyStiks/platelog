import argparse
import json
import cv2
from datetime import timedelta, datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from collections import Counter
from pymediainfo import MediaInfo
from ultralytics import YOLO
from fast_alpr import ALPR

# ---- settings ----
FRAMES_PER_SECOND = 5 # look at x frames/s instead of all 30
TRUCK_MODEL = "yolo11n.pt" # small COCO model, downloaded on first run
TRUCK_CLASSES = [5, 7] # COCO class ids: 5 = bus, 7 = truck (add 2 for car)
MIN_TRUCK_CONFIDENCE = 0.4 # lower confidence -> not a truck
MIN_TRUCK_FRAMES = 3 # a "truck" seen in fewer sampled frames is probably a false detection
MIN_PLATE_CONFIDENCE = 0.55 # lower confidence -> not a plate
MIN_PLATE_WIDTH = 40 # in px, narrower plates are hopeless, skip OCR
MIN_SHARPNESS = 10 # blurrier crops are not worth reading
MIN_AGREEING_READS = 3 # fewer agreeing reads -> UNREADABLE
MIN_OCR_CONFIDENCE = 0.5 # lower confidence -> UNREADABLE
SITE_TIMEZONE = "America/Moncton" # time zone to use for truck sightings
TIME_FORMAT = "%Y-%m-%d %H:%M:%S %Z"
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp"}

# Reads metadata and returns the start time of the clips
def recording_start(path):
    zone = ZoneInfo(SITE_TIMEZONE)

    # MediaInfo reads the recording time from the file header
    # The value is None when the file has no timestamp
    stamp = MediaInfo.parse(str(path)).general_tracks[0].encoded_date
    if stamp:
        when = datetime.strptime(stamp.replace("UTC", "").strip(), "%Y-%m-%d %H:%M:%S")
        when = when.replace(tzinfo=timezone.utc)
        return when.astimezone(zone)

    return None # if file doesn't have recorded time in metadata

# Determins sharpness of the image
def sharpness(image):
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

# Looks for a plate inside one truck crop and reads it. Returns an observation (dict)
def read_plate(truck_crop, alpr):
    obs = {"crop": None, "sharp": 0.0, "text": "", "conf": 0.0} # crop = the plate, None if no plate found

    detections = alpr.detector.predict(truck_crop)
    if not detections:
        return obs

    det = max(detections, key=lambda d: d.confidence) # one plate per truck: take the most confident
    x1, y1, x2, y2 = det.bounding_box.clamp(truck_crop.shape[1], truck_crop.shape[0]).xyxy # keep the coords inside the image
    crop = truck_crop[y1:y2, x1:x2]
    if crop.size == 0:
        return obs

    obs["crop"] = crop
    obs["sharp"] = sharpness(crop)

    # Read the text only if the crop is big and sharp enough
    if x2 - x1 >= MIN_PLATE_WIDTH and obs["sharp"] >= MIN_SHARPNESS:
        result = alpr.ocr.predict(crop)
        if result and result.text:
            conf = result.confidence
            if isinstance(conf, list): # one number per character slot
                conf = sum(conf[:len(result.text)]) / len(result.text) # average real characters only
            obs["text"] = result.text
            obs["conf"] = conf

    return obs

# Tracks the trucks through a clip and reads the plate on each one in every sampled frame.
# Returns one dict per truck: {"obs": [one observation per sampled frame], "image": the truck's best picture}
def read_clip(path, yolo, alpr):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / FRAMES_PER_SECOND))            # keep every step-th frame

    trucks = {} # tracker id -> truck
    index = 0
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if index % step == 0:
            # persist=False on a clip's first frame resets the tracker, so ids start fresh for each clip
            result = yolo.track(frame, persist=(index > 0), classes=TRUCK_CLASSES,
                                conf=MIN_TRUCK_CONFIDENCE, verbose=False, device="cpu")[0]

            if result.boxes.id is not None: # the tracker found at least one truck
                for box, track_id in zip(result.boxes.xyxy.int().tolist(), result.boxes.id.int().tolist()):
                    x1, y1, x2, y2 = box
                    truck_crop = frame[y1:y2, x1:x2]
                    if truck_crop.size == 0:
                        continue

                    truck = trucks.setdefault(track_id, {"obs": [], "image": None, "image_area": 0})

                    # Keep the biggest view of the truck as its picture (for trucks without a readable plate)
                    area = (x2 - x1) * (y2 - y1)
                    if area > truck["image_area"]:
                        truck["image"] = truck_crop.copy()
                        truck["image_area"] = area

                    obs = read_plate(truck_crop, alpr)
                    obs["t"] = index / fps
                    truck["obs"].append(obs)
        index += 1

    cap.release()
    return [t for t in trucks.values() if len(t["obs"]) >= MIN_TRUCK_FRAMES]

# Chose a plate if different frames show different plate readings
def decide_plate(observations, min_reads, min_conf):
    # Keep only reads that look like a plate (4-8 characters; empty reads are dropped too)
    texts = [o["text"] for o in observations if 4 <= len(o["text"]) <= 8]
    if not texts:
        return "UNREADABLE", 0.0

    # Majority vote: the text seen most often across frames wins
    plate, votes = Counter(texts).most_common(1)[0]

    # Confidence = share of reads that agree times the average OCR confidence of those reads
    agreeing = [o for o in observations if o["text"] == plate]
    ocr_conf = sum(o["conf"] for o in agreeing) / len(agreeing)
    confidence = votes / len(texts) * ocr_conf

    # A confident wrong plate is worse than no plate: when in doubt, say UNREADABLE
    if votes < min_reads:
        return "UNREADABLE", confidence
    if confidence < min_conf:
        return "UNREADABLE", confidence

    return plate, confidence

# Formats a single truck in to an documentable event
def make_event(clip, number, truck, start, out_dir):
    plate, confidence = decide_plate(truck["obs"], MIN_AGREEING_READS, MIN_OCR_CONFIDENCE)

    # Best image: the sharpest plate crop from frames that read the winning text.
    # No plate read: show the truck instead.
    if plate == "UNREADABLE":
        image = truck["image"]
    else:
        agreeing = [o for o in truck["obs"] if o["text"] == plate]
        image = max(agreeing, key=lambda o: o["sharp"])["crop"]

    frame_path = out_dir / "frames" / f"{clip.stem}_truck{number}.jpg"
    cv2.imwrite(str(frame_path), image)

    # First and last time the truck (not just its plate) was in view
    first, last = truck["obs"][0]["t"], truck["obs"][-1]["t"]
    if start is None:
        first_seen = last_seen = "MISSING"
    else:
        first_seen = (start + timedelta(seconds=first)).strftime(TIME_FORMAT)
        last_seen = (start + timedelta(seconds=last)).strftime(TIME_FORMAT)

    return {
        "clip": clip.name,
        "plate": plate,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "confidence": round(confidence, 3),
        "best_frame": str(frame_path.relative_to(out_dir)),
    }

def main():
    parser = argparse.ArgumentParser(description="Read truck plates from video clips.")
    parser.add_argument("clips", type=Path, help="folder with video clips")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output folder")
    args = parser.parse_args()

    (args.out / "frames").mkdir(parents=True, exist_ok=True)

    clips = []
    for path in args.clips.iterdir():
        if path.suffix.lower() in VIDEO_TYPES:
            clips.append(path)
    clips.sort()

    # Load the models once, reuse for every frame of every clip
    yolo = YOLO(TRUCK_MODEL)
    alpr = ALPR(detector_model="yolo-v9-t-640-license-plate-end2end",
                ocr_model="cct-s-v2-global-model",
                detector_conf_thresh=MIN_PLATE_CONFIDENCE)

    events = []
    for clip in clips:
        start = recording_start(clip)
        trucks = read_clip(clip, yolo, alpr)
        for number, truck in enumerate(trucks, start=1):
            events.append(make_event(clip, number, truck, start, args.out))
        print(f"{clip.name}: {len(trucks)} truck(s)")

    (args.out / "events.json").write_text(json.dumps(events, indent=2))
    print(f"wrote {args.out / 'events.json'}")

if __name__ == "__main__":
    main()
