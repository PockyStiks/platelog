import argparse
import json
import cv2
import time
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
TIME_FORMAT = "%Y-%m-%d %H:%M:%S %Z" # time format to use for truck sighting time 
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp"} # Allowed clip file types

# Ground truth written by hand, many plates are unreadable to me due to low resolution/compression
GROUND_TRUTH = {
    "IMG_5177.MOV" : ["UNREADABLE"],
    "IMG_5178.MOV" : ["PUY052"],
    "IMG_5179.MOV" : ["PUY427"],
    "IMG_5181.MOV" : ["UNREADABLE"],
    "IMG_5182.MOV" : ["UNREADABLE"]
}

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

# Determines sharpness of the image
def sharpness(image):
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

# Tracks the trucks through a clip and reads the plate on each one in every sampled frame
# Returns one dict per truck: {"obs": [one observation per sampled frame], "image": the truck's best picture}
def read_clip(path, yolo, alpr):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / FRAMES_PER_SECOND)) # keep every step-th frame

    trucks = {} # {tracker id : truck}, keeps track of all trucks in this clip
    index = 0 # frame index
    while True:
        ok, frame = cap.read()

        # If the video is over, break
        if not ok:
            break

        if index % step == 0: # for each step-th frame
            # Find trucks in frame. persist=False on a clip's first frame resets the tracker, so ids start fresh for each clip
            result = yolo.track(frame, persist=(index > 0), classes=TRUCK_CLASSES,
                                conf=MIN_TRUCK_CONFIDENCE, verbose=False, device="cpu")[0]

            if result.boxes.id is not None: # the tracker found at least one truck
                for box, track_id in zip(result.boxes.xyxy.int().tolist(), result.boxes.id.int().tolist()): # for each box, truck id pair
                    x1, y1, x2, y2 = box
                    truck_crop = frame[y1:y2, x1:x2]
                    if truck_crop.size == 0: # if crop is empty, skip
                        continue

                    # Returns existing truck entry based on truck_id, or creates a new truck_id
                    # with the specified default fields if truck_id doesn't exist (new truck)
                    truck = trucks.setdefault(track_id, {"obs": [], "image": None, "image_area": 0})

                    # Keep the biggest view of the truck as its picture  to use for trucks without a readable plate
                    area = (x2 - x1) * (y2 - y1)
                    if area > truck["image_area"]:
                        truck["image"] = truck_crop.copy()
                        truck["image_area"] = area
                    
                    obs = read_plate(truck_crop, alpr) # get plate from truck crop
                    obs["t"] = index / fps # offset time
                    truck["obs"].append(obs) # append this observation to the truck's observation hsitory
        index += 1

    cap.release()
    return [t for t in trucks.values() if len(t["obs"]) >= MIN_TRUCK_FRAMES]

# Looks for a plate inside one truck crop and reads it. Returns an observation (dict)
def read_plate(truck_crop, alpr):
    obs = {"crop": None, "sharp": 0.0, "text": "", "conf": 0.0} # crop = the plate, None if no plate found

    # Gets plate detections
    detections = alpr.detector.predict(truck_crop)
    if not detections:
        return obs

    # Get the most confident plate detection in case of multiples plates detected in the truck crop
    det = max(detections, key=lambda d: d.confidence)
    x1, y1, x2, y2 = det.bounding_box.clamp(truck_crop.shape[1], truck_crop.shape[0]).xyxy # keep the coords inside the image
    crop = truck_crop[y1:y2, x1:x2] # crop out the plate
    if crop.size == 0: # if crop is empty, return obs with empty results
        return obs
    obs["crop"] = crop
    obs["sharp"] = sharpness(crop)

    # Read the text only if the crop is big and sharp enough
    if x2 - x1 >= MIN_PLATE_WIDTH and obs["sharp"] >= MIN_SHARPNESS:
        result = alpr.ocr.predict(crop) # preform OCR 

        # get OCR confidence
        if result and result.text:
            conf = result.confidence
            if isinstance(conf, list): # one number per character slot
                conf = sum(conf[:len(result.text)]) / len(result.text) # average conf of real characters only, not trailing white spaces
            obs["text"] = result.text
            obs["conf"] = conf

    return obs

# Chose a plate if different frames show different plate readings
def decide_plate(observations, min_reads, min_conf):
    # Keep only reads that look like a plate (4-8 characters; empty reads are dropped too)
    texts = [o["text"] for o in observations if 4 <= len(o["text"]) <= 8]
    if not texts:
        return "UNREADABLE", 0.0

    # Majority vote: the text seen most often across frames wins
    plate, votes = Counter(texts).most_common(1)[0] # Counter() returns [(text : count), ...]

    # Confidence = share of reads that agree multiplied by the average OCR confidence of those reads
    agreeing = [o for o in observations if o["text"] == plate] # list of agreeing plates
    ocr_conf = sum(o["conf"] for o in agreeing) / len(agreeing) # average conf for agreeing plates
    confidence = votes / len(texts) * ocr_conf # n cases for the chosen plate / total cases * average conf of agreeing plates  

    # Returns UNREADABLE if confidence is below the threshold or there are too few readings across frames
    if votes < min_reads:
        return "UNREADABLE", confidence
    if confidence < min_conf:
        return "UNREADABLE", confidence

    return plate, confidence

# Formats a single truck in to a documentable event
def make_event(clip, number, truck, start, out_dir):
    plate, confidence = decide_plate(truck["obs"], MIN_AGREEING_READS, MIN_OCR_CONFIDENCE)

    # Best image: the sharpest plate crop from frames that read the winning text
    # No plate read: show the truck instead
    if plate == "UNREADABLE":
        image = truck["image"]
    else:
        agreeing = [o for o in truck["obs"] if o["text"] == plate]
        image = max(agreeing, key=lambda o: o["sharp"])["crop"]

    # Save crop of plate or truck
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
        "best_frame": str(frame_path.relative_to(out_dir))
    }

def write_metrics(clips, events, out_dir, execution_time):
    truck_tp = 0
    truck_fp = 0
    truck_fn = 0

    plate_tp = 0
    plate_fp = 0
    plate_fn = 0
    unreadable_correct = 0

    results = []

    for clip in clips:
        truths = GROUND_TRUTH[clip.name]

        if truths is None: # if no ground truth for this clip
            continue

        detected = [event["plate"] for event in events if event["clip"] == clip.name] # plates detected in current clip
        results.append(f"{clip.name}: {' / '.join(detected) or 'NO EVENT'}")

        # Evaluate truck detection
        truck_tp += min(len(truths), len(detected))
        truck_fp += max(0, len(detected) - len(truths))
        truck_fn += max(0, len(truths) - len(detected))

        # Evaluate plate recognition
        for truth, plate in zip(truths, detected):
            if truth == "UNREADABLE":
                if plate == "UNREADABLE":
                    unreadable_correct += 1
                else:
                    plate_fp += 1

            elif plate == truth:
                plate_tp += 1

            else:
                plate_fn += 1
                if plate != "UNREADABLE":
                    plate_fp += 1

    # Calculate metrics
    truck_precision = truck_tp / (truck_tp + truck_fp) if truck_tp + truck_fp else 0
    truck_recall = truck_tp / (truck_tp + truck_fn) if truck_tp + truck_fn else 0

    plate_precision = plate_tp / (plate_tp + plate_fp) if plate_tp + plate_fp else 0
    plate_recall = plate_tp / (plate_tp + plate_fn) if plate_tp + plate_fn else 0

    # Write results
    text = "# Evaluation Results\n\n"

    text += "## Ground Truth\n\n"
    text += "| Clip | Expected Plates |\n"
    text += "|---|---|\n"

    for clip, plates in GROUND_TRUTH.items():
        text += f"| {clip} | {', '.join(plates)} |\n"

    text += "\n## Detected Plates\n\n"
    text += "| Clip | Detected Plates |\n"
    text += "|---|---|\n"

    for result in results:
        clip, plates = result.split(": ", 1)
        text += f"| {clip} | {plates} |\n"

    text += "\n## Truck Detection\n\n"
    text += f"- True Positives: {truck_tp}\n"
    text += f"- False Positives: {truck_fp}\n"
    text += f"- False Negatives: {truck_fn}\n"
    text += f"- Precision: {truck_precision:.2f}\n"
    text += f"- Recall: {truck_recall:.2f}\n"

    text += "\n## Plate Recognition\n\n"
    text += f"- Correctly Read: {plate_tp}\n"
    text += f"- Incorrectly Read: {plate_fp}\n"
    text += f"- Not Read: {plate_fn}\n"
    text += f"- Correctly Identified Unreadable: {unreadable_correct}\n"
    text += f"- Precision: {plate_precision:.2f}\n"
    text += f"- Recall: {plate_recall:.2f}\n"

    text += f"\n## Execution Time\n"
    text += f"- Total runtime: {execution_time:.2f} seconds\n"

    (out_dir / "RESULTS.md").write_text(text)

def main():
    start_time = time.perf_counter() # measures execution time

    # Parses CLI arguments
    parser = argparse.ArgumentParser(description="Read truck plates from video clips.")
    parser.add_argument("clips", type=Path, help="folder with video clips")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output folder")
    args = parser.parse_args()

    (args.out / "frames").mkdir(parents=True, exist_ok=True) # creates output dir

    # Gets clips in input dir and checks for valid filetype
    clips = []
    for path in args.clips.iterdir():
        if path.suffix.lower() in VIDEO_TYPES:
            clips.append(path)
    clips.sort()

    # Loads the models once, reuse for every frame of every clip
    yolo = YOLO(TRUCK_MODEL)
    alpr = ALPR(detector_model="yolo-v9-t-640-license-plate-end2end",
                ocr_model="cct-s-v2-global-model",
                detector_conf_thresh=MIN_PLATE_CONFIDENCE)

    # Process clips
    events = []
    for clip in clips:
        start = recording_start(clip)
        trucks = read_clip(clip, yolo, alpr)
        for number, truck in enumerate(trucks, start=1):
            events.append(make_event(clip, number, truck, start, args.out))
        print(f"{clip.name}: {len(trucks)} truck(s)")

    # Write output files
    (args.out / "events.json").write_text(json.dumps(events, indent=2))
    print(f"wrote {args.out / 'events.json'}")

    execution_time = time.perf_counter() - start_time # ends timer

    write_metrics(clips, events, args.out, execution_time)
    print(f"wrote {args.out / 'RESULTS.md'}")

if __name__ == "__main__":
    main()
