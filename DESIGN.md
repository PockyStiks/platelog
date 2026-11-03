# Design – PlateLog

## Architecture

```mermaid
flowchart LR
    A["Input video clips"] --> B["Frame sampling (5 FPS)"]
    B --> C["YOLO truck detection and tracking"]
    C --> D["Crop detected trucks"]
    D --> E["Plate detection (Fast-ALPR)"]
    E --> F["OCR recognition"]
    F --> G["Aggregate readings across frames"]
    G --> H["Majority voting and confidence filtering"]
    H --> I["Truck event generation"]
    I --> J["events.json"]
    I --> K["Best representative images"]
    J --> L["RESULTS.md"]
````

## Technology choices

**Selected: YOLO + Fast-ALPR**

YOLO detects and tracks trucks, while Fast-ALPR detects and reads license plates. Majority voting across frames improves recognition reliability.

Both run locally, with no API costs or internet dependency during inference.

**Alternatives considered:**

- **VLM:** More computationally expensive and potentially inconsistent for a simple OCR task.
- **Cloud/API:** Introduces recurring costs, network dependency, and additional latency.

## Accuracy, cost and latency

| Consideration | Approach |
|---|---|
| Accuracy | Multi-frame voting and confidence thresholds. |
| Cost | No API fees; hardware and electricity costs only. |
| Latency | 5 FPS sampling; clips processed sequentially. |
| Offline | Fully local inference. |

## Scaling to a 24/7 deployment

For a fixed scale-house camera:
- Replace video input with a continuous camera stream.
- Finalize events when trucks leave the camera view.
- Use persistent storage (SQLite) instead of JSON.
- Add camera monitoring, automatic recovery, and storage cleanup.

**What breaks at 10x the data?**

Sequential processing could create backlogs, while storing every frame would increase disk usage. Asynchronous processing, bounded queues, and retention policies would address this.

## Future work

With two additional days, I would:
- Test image preprocessing techniques.
- Benchmark alternative YOLO and ALPR models.
- Improve error handling and logging.
- Test under different lighting conditions.

## Project execution

- **Time spent:** Approximately 10 hours.
- **What was cut:** Testing with alternative models and image preprocessing techniques.
