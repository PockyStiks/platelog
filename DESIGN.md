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

**Selected: Local YOLO + Fast-ALPR pipeline**

YOLO is used to detect and track trucks, while Fast-ALPR handles license plate detection and OCR. Multiple observations are combined using majority voting to improve recognition reliability.

This approach offers a good balance between accuracy, latency, and cost. Both frameworks can run locally using existing hardware, making the system suitable for rural sites with unreliable internet connectivity. There are no per-request API costs, and inference does not depend on an internet connection.

**Alternatives considered:**

- **Vision-Language Model (VLM):** More flexible, but excessive for a relatively simple license plate recognition task. It introduces additional inference overhead, requires more powerful hardware, and may produce inconsistent or hallucinated readings.
- **Cloud/API-based recognition:** Could provide access to managed models without maintaining local inference infrastructure. However, internet dependency, potential latency, and recurring API costs make it less suitable for a rural deployment.

## Accuracy, cost and latency

| Consideration | Design decision |
|---|---|
| Accuracy | Multiple frame observations, majority voting, and confidence thresholds reduce incorrect readings. |
| Cost per truck | No API fees; costs are primarily hardware, electricity, and maintenance. |
| Latency | Sampling at 5 FPS reduces computation, but clips are currently processed sequentially rather than in real time. |
| Offline operation | Models run locally, requiring no internet connection during inference. |

## Scaling to a 24/7 scale-house deployment

The current implementation is designed for prerecorded phone clips. For a fixed camera installation, I would keep YOLO tracking, Fast-ALPR, frame sampling, and multi-frame plate aggregation.

The main changes would be:

- Replace video-file input with a continuous camera stream.
- Process frames continuously and finalize an event when a truck leaves the camera's field of view.
- Add persistent event storage (e.g. SQLite) instead of relying exclusively on JSON files.
- Add automatic cleanup and log rotation to prevent storage exhaustion.
- Implement camera health monitoring, automatic recovery, and error handling.
- Use reliable camera timestamps and local time synchronization.

## Limitations and future work

**What breaks at 10x the data?**

Processing time and storage requirements would increase. Sequential clip processing could create a backlog, while retaining every frame and generated image indefinitely would consume significant disk space. I would introduce asynchronous processing, bounded queues, and retention policies.

**What I would do with two additional days:**

- Experiment with image preprocessing techniques such as contrast enhancement, denoising, resizing, and perspective correction.
- Benchmark alternative YOLO and ALPR models to compare recognition accuracy and inference speed.
- Improve error handling, logging, and recovery from failed video reads or model errors.
- Evaluate performance on additional video samples and more varied lighting conditions.

## Project execution

- **Time spent:** Approximately 10 hours.
- **What changed:** The initial implementation used Fast-ALPR alone. It was replaced with a YOLO + Fast-ALPR pipeline to include truck detection and tracking, as required by the project.
- **What was cut:** Testing with other models and preprocessing steps, which could potentially lead to better results.
