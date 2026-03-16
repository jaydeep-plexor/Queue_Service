import cv2
import numpy as np
import time
import threading
import logging
import requests
import base64
import uvicorn
from typing import List, Optional, Dict, Any, Set
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
from collections import defaultdict
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & GLOBALS
# ==========================================

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model Configuration
YOLO_MODEL_PATH = "yolo11s.pt"  # Ensure this model exists or use 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.4

# Global State for Multi-Camera Support
ACTIVE_STREAMS = {}
streams_lock = threading.Lock()

# FastAPI App Setup
app = FastAPI(
    title="Queue Management API",
    description="Real-time Queue Analysis and Alerting System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. PYDANTIC MODELS
# ==========================================
class BufferlessVideoCapture:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.lock = threading.Lock()
        self.ret = False
        self.latest_frame = None
        self.running = True
        
        # Start the background reader thread
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

        # Block until the first frame arrives
        logger.info(f"Connecting to stream {source} (Waiting for first frame...)")
        start_wait = time.time()
        
        # Wait up to 15 seconds for the RTSP stream to buffer the first frame
        while not self.ret and self.cap.isOpened():
            if time.time() - start_wait > 15.0:
                logger.error(f"Timeout: Could not get initial frame from {source}")
                break
            time.sleep(0.1) 

    def _reader(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.latest_frame = frame
            
            # If the stream drops, prevent a tight CPU-hogging loop
            if not ret:
                time.sleep(0.01)

    def read(self):
        """Called by your YOLO inference loop to get the most recent frame."""
        with self.lock:
            # We return a copy/reference to the latest frame
            return self.ret, self.latest_frame

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop_id):
        """Pass-through for cap.get() to grab properties like width/height."""
        return self.cap.get(prop_id)

    def release(self):
        """Safely shut down the thread and release the camera."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


class QueueConfig(BaseModel):
    queue_id: str
    roi: List[List[int]]  # [[x,y], [x,y], ...]
    max_people: int
    max_time: int  # seconds

class StartAnalysisRequest(BaseModel):
    rtsp_url: str
    camera_id: str
    # NEW FIELDS: The resolution of the user's screen/player
    ref_width: int  
    ref_height: int
    send_image: bool = True
    callback_url: Optional[str] = None
    queues: List[QueueConfig]

class StopAnalysisRequest(BaseModel):
    camera_id: str

class PersonInfo(BaseModel):
    person_id: str
    duration: float

class QueueStatus(BaseModel):
    queue_id: str
    count: int
    persons: List[PersonInfo]
    is_violation: bool

class PollResponse(BaseModel):
    camera_id: str
    timestamp: float
    queues: List[QueueStatus]
    image_base64: Optional[str] = None
    is_active: bool
    error: Optional[str] = None

class GenericResponse(BaseModel):
    status: str
    camera_id: str
    message: Optional[str] = None

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def point_in_polygon(point: tuple, polygon_points: List[List[int]]) -> bool:
    try:
        poly = Polygon(polygon_points)
        return poly.contains(Point(point))
    except Exception:
        return False

def send_alert(callback_url: str, payload: dict):
    """Sends webhook alert in a separate thread"""
    def _req():
        try:
            requests.post(callback_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    threading.Thread(target=_req, daemon=True).start()

def draw_visuals(frame, queue_states, queues_config):
    """Draws ROIs, bounding boxes, and stats on the frame"""
    vis_frame = frame.copy()
    
    # Draw Queues (ROIs)
    for q_conf in queues_config:
        qid = q_conf['queue_id']
        q_state = queue_states.get(qid)
        
        if not q_state: continue

        # Determine color based on violation
        color = (0, 255, 0) # Green (OK)
        if q_state['is_violation']:
            color = (0, 0, 255) # Red (Violation)
        
        # Draw ROI Polygon
        pts = np.array(q_conf['roi'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_frame, [pts], True, color, 2)
        
        # Draw Label
        roi_center = np.mean(pts, axis=0)[0]
        label = f"{qid}: {q_state['count']} ppl"
        cv2.putText(vis_frame, label, (int(roi_center[0]) - 50, int(roi_center[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis_frame

# ==========================================
# 4. ANALYSIS WORKER
# ==========================================

# ==========================================
# 4. ANALYSIS WORKER (Fully Updated)
# ==========================================

def analysis_worker(camera_id: str, rtsp_url: str, queues_config: List[dict], 
                    options: dict, stop_event: threading.Event, state_dict: dict,
                    ref_res: tuple):
    
    logger.info(f"[{camera_id}] Starting Analysis Worker...")
    
    try:
        # Initialize YOLO
        model = YOLO(YOLO_MODEL_PATH)
        
        # Open Video Stream using the optimized Bufferless class
        source = int(rtsp_url) if rtsp_url.isdigit() else rtsp_url
        cap = BufferlessVideoCapture(source)
        
        if not cap.isOpened():
            raise Exception(f"Could not open video source: {rtsp_url}")

        # --- CALCULATE SCALING FACTOR ---
        real_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frontend_w, frontend_h = ref_res

        scale_x = real_width / frontend_w if frontend_w > 0 else 1.0
        scale_y = real_height / frontend_h if frontend_h > 0 else 1.0

        logger.info(f"[{camera_id}] Scaling ROIs: Frontend({frontend_w}x{frontend_h}) -> Native({real_width}x{real_height}). Factor: {scale_x:.2f}, {scale_y:.2f}")

        # Apply Scaling to ALL Queues PERMANENTLY for this session
        for q in queues_config:
            original_roi = q['roi']
            scaled_roi = []
            for point in original_roi:
                sx = int(round(point[0] * scale_x))
                sy = int(round(point[1] * scale_y))
                sx = min(max(0, sx), real_width)
                sy = min(max(0, sy), real_height)
                scaled_roi.append([sx, sy])
            q['roi'] = scaled_roi
        # ------------------------------------------

        # Tracking Data
        queue_trackers = {q['queue_id']: {} for q in queues_config}
        
        # Alert Debouncing: { queue_id: last_alert_timestamp }
        last_alert_time = {q['queue_id']: 0 for q in queues_config}
        ALERT_COOLDOWN = 10 # Seconds between alerts for the same queue

        while not stop_event.is_set():
            ret, frame = cap.read()
            
            # --- ROBUST RECONNECTION LOGIC ---
            if not ret or frame is None:
                logger.warning(f"[{camera_id}] ALERT: Connection dropped. Retrying in 5 seconds...")
                cap.release()
                
                # Wait 5 seconds, checking stop_event so API can still shut it down
                if stop_event.wait(timeout=5.0):
                    break
                    
                cap = BufferlessVideoCapture(source)
                if cap.isOpened():
                    logger.info(f"[{camera_id}] SUCCESS: Reconnected to stream.")
                continue
            # ---------------------------------

            current_time = time.time()
            
            # 1. Run Object Tracking
            results = model.track(frame, persist=True, classes=[0], verbose=False, conf=CONFIDENCE_THRESHOLD, iou=0.4, max_det=50, tracker="bytetrack.yaml", device=0)
            
            current_frame_data = {} # To store in global state
            
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # Snapshot of who is in which queue currently
                present_in_queues = {q['queue_id']: set() for q in queues_config}
                
                # 2. Map People to Queues
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    # Calculate foot point (bottom center) for better queue accuracy
                    foot_point = ((x1 + x2) / 2, y2)
                    
                    for q_conf in queues_config:
                        qid = q_conf['queue_id']
                        if point_in_polygon(foot_point, q_conf['roi']):
                            present_in_queues[qid].add(track_id)
                            
                            # If new to queue, record entry time
                            if track_id not in queue_trackers[qid]:
                                queue_trackers[qid][track_id] = current_time
                            break # Assume person is in only one queue at a time
                
                # 3. Clean up trackers (people who left)
                for q_conf in queues_config:
                    qid = q_conf['queue_id']
                    tracked_ids = list(queue_trackers[qid].keys())
                    for tid in tracked_ids:
                        if tid not in present_in_queues[qid]:
                            del queue_trackers[qid][tid]
                
                # 4. Calculate Stats & Violations
                for q_conf in queues_config:
                    qid = q_conf['queue_id']
                    max_p = q_conf['max_people']
                    max_t = q_conf['max_time']
                    
                    persons_list = []
                    violation_detected = False
                    violation_reason = []

                    # Count violation
                    current_count = len(queue_trackers[qid])
                    if current_count > max_p:
                        violation_detected = True
                        violation_reason.append(f"Count Exceeded ({current_count} > {max_p})")

                    # Time violation
                    for tid, entry_time in queue_trackers[qid].items():
                        duration = current_time - entry_time
                        persons_list.append({"person_id": str(tid), "duration": round(duration, 2)})
                        
                        if duration > max_t:
                            violation_detected = True
                            violation_reason.append(f"Wait Time Exceeded (ID {tid}: {int(duration)}s)")

                    # Prepare Queue Status Object
                    q_status = {
                        "queue_id": qid,
                        "count": current_count,
                        "persons": persons_list,
                        "is_violation": violation_detected
                    }
                    current_frame_data[qid] = q_status

                    # 5. Handle Alerts
                    if violation_detected and options.get('callback_url'):
                        if (current_time - last_alert_time[qid]) > ALERT_COOLDOWN:
                            alert_payload = {
                                "type": "violation",
                                "camera_id": camera_id,
                                "queue_id": qid,
                                "timestamp": current_time,
                                "reasons": violation_reason,
                                "current_status": q_status
                            }
                            send_alert(options['callback_url'], alert_payload)
                            last_alert_time[qid] = current_time

            else:
                # No detections, clear queues
                current_frame_data = {q['queue_id']: {
                    "queue_id": q['queue_id'], "count": 0, "persons": [], "is_violation": False
                } for q in queues_config}
                
                # Clear trackers
                for qid in queue_trackers:
                    queue_trackers[qid].clear()

            # 6. Generate Image (if requested)
            b64_image = None
            if options.get('send_image'):
                vis_frame = draw_visuals(frame, current_frame_data, queues_config)
                _, buffer = cv2.imencode('.jpg', vis_frame)
                b64_image = base64.b64encode(buffer).decode('utf-8')

            # 7. Update Global State
            with streams_lock:
                state_dict['timestamp'] = current_time
                state_dict['queues'] = list(current_frame_data.values())
                state_dict['image_base64'] = b64_image
            
            # Maintain simpler loop rate
            time.sleep(0.05)

        cap.release()
        logger.info(f"[{camera_id}] Worker thread stopped.")

    except Exception as e:
        logger.error(f"[{camera_id}] Error in worker: {e}")
        with streams_lock:
            state_dict['error'] = str(e)
            state_dict['is_active'] = False

# ==========================================
# 5. API ENDPOINTS
# ==========================================

@app.post("/analysis/start", response_model=GenericResponse)
async def start_analysis(req: StartAnalysisRequest):
    """Starts analysis on a camera stream"""
    
    with streams_lock:
        if req.camera_id in ACTIVE_STREAMS and ACTIVE_STREAMS[req.camera_id]['data']['is_active']:
            return GenericResponse(status="already_running", camera_id=req.camera_id, message="Active.")

    queues_config = [q.dict() for q in req.queues]
    options = {
        "send_image": req.send_image,
        "callback_url": req.callback_url
    }
    
    stream_data = {
        "is_active": True,
        "timestamp": time.time(),
        "queues": [],
        "image_base64": None,
        "error": None
    }
    
    stop_event = threading.Event()
    
    # Pass (req.ref_width, req.ref_height) to the worker
    t = threading.Thread(
        target=analysis_worker,
        args=(req.camera_id, req.rtsp_url, queues_config, options, stop_event, stream_data, (req.ref_width, req.ref_height)),
        daemon=True
    )
    t.start()
    
    with streams_lock:
        ACTIVE_STREAMS[req.camera_id] = {
            "thread": t,
            "stop_event": stop_event,
            "data": stream_data
        }
        
    return GenericResponse(status="started", camera_id=req.camera_id)


@app.post("/analysis/stop", response_model=GenericResponse)
async def stop_analysis(req: StopAnalysisRequest):
    """Stops analysis for a specific camera"""
    
    with streams_lock:
        if req.camera_id not in ACTIVE_STREAMS:
            raise HTTPException(status_code=404, detail="Camera ID not found")
        
        stream_obj = ACTIVE_STREAMS[req.camera_id]
        stream_obj['stop_event'].set()
        
        # We mark it inactive immediately, thread will clean up
        stream_obj['data']['is_active'] = False
        
        # Join thread (optional, but good for cleanup confirmation if needed quickly)
        # stream_obj['thread'].join(timeout=2.0)
        
        del ACTIVE_STREAMS[req.camera_id]
        
    return GenericResponse(status="stopped", camera_id=req.camera_id)


@app.get("/analysis/poll/{camera_id}", response_model=PollResponse)
async def poll_results(camera_id: str):
    """Returns real-time queue statistics and image"""
    
    with streams_lock:
        if camera_id not in ACTIVE_STREAMS:
             raise HTTPException(status_code=404, detail="Analysis not running for this camera ID")
        
        raw_data = ACTIVE_STREAMS[camera_id]['data']
        
        # Deep copy or construct response carefully
        return PollResponse(
            camera_id=camera_id,
            timestamp=raw_data['timestamp'],
            queues=raw_data['queues'],
            image_base64=raw_data['image_base64'],
            is_active=raw_data['is_active'],
            error=raw_data['error']
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)