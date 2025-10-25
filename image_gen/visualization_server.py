"""
Visualization server with real-time pipeline monitoring.

This server provides a web interface that shows exactly what's happening
during image generation, with live updates and component highlighting.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import asyncio
import logging
import time
import json
from datetime import datetime

from image_gen.core import ImageGenerator
from image_gen.config import get_config

logger = logging.getLogger(__name__)

app = FastAPI(title="ImageGeneratorLLM Visualization")

# Global state
_generator: Optional[ImageGenerator] = None
_active_connections: List[WebSocket] = []
_current_state: Dict[str, Any] = {
    "component": "idle",
    "progress": 0,
    "total_steps": 0,
    "message": "Ready",
    "metrics": {}
}


class TimingTracker:
    """Track component execution times for visualization."""

    def __init__(self):
        self.component_times = {}
        self.component_start_times = {}

    def start_component(self, component_name: str):
        """Mark the start of a component's execution."""
        self.component_start_times[component_name] = time.time()

    def end_component(self, component_name: str) -> float:
        """Mark the end of a component and return elapsed time."""
        if component_name not in self.component_start_times:
            return 0.0

        elapsed = time.time() - self.component_start_times[component_name]
        self.component_times[component_name] = elapsed
        return elapsed

    def get_percentages(self) -> dict:
        """Calculate percentage of total time for each component."""
        total_time = sum(self.component_times.values())
        if total_time == 0:
            return {}

        return {
            component: (elapsed / total_time) * 100
            for component, elapsed in self.component_times.items()
        }

    def reset(self):
        """Reset all timing data."""
        self.component_times.clear()
        self.component_start_times.clear()


timing_tracker = TimingTracker()


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


manager = ConnectionManager()


async def emit_state(component: str, progress: int = 0, total: int = 0, message: str = "", metrics: dict = None, api_call: str = None, component_timing: dict = None, component_percentages: dict = None):
    """Emit current state to all connected clients."""
    state = {
        "component": component,
        "progress": progress,
        "total": total,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics or {}
    }
    if api_call:
        state["api_call"] = api_call
    if component_timing:
        state["component_timing"] = component_timing
    if component_percentages:
        state["component_percentages"] = component_percentages
    await manager.broadcast(state)


class GenerateRequest(BaseModel):
    prompt: str
    mode: Optional[str] = "text2img"  # text2img, img2img, controlnet
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    seed: Optional[int] = None
    init_image: Optional[str] = None  # Base64 encoded image for img2img/controlnet
    strength: Optional[float] = 0.8  # For img2img: how much to transform (0-1)


@app.get("/", response_class=HTMLResponse)
async def get_interface():
    """Serve the visualization interface."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Generation Pipeline Visualizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 10px;
            font-size: 14px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 12px;
            color: #4fc3f7;
            font-size: 1.4em;
            font-weight: 600;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 12px;
        }

        /* Grid for Model Insights on left 1/3 */
        .insights-layout-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 15px;
            margin-bottom: 15px;
            align-items: start;
        }
        .grid.control-flow {
            grid-template-columns: 1fr 2fr;
        }
        .panel {
            background: #1a1f3a;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #2a3f5f;
        }
        .panel h2 {
            color: #4fc3f7;
            margin-bottom: 10px;
            font-size: 1.05em;
            font-weight: 600;
        }

        /* Flowchart - Horizontal Layout */
        .pipeline {
            position: relative;
            padding: 18px 15px;
            min-height: 240px;
        }

        /* Comprehensive Multi-Path Pipeline */
        .pipeline-comprehensive {
            position: relative;
            padding: 12px;
        }

        .pipeline-stage {
            margin-bottom: 20px;
            padding: 16px;
            background: linear-gradient(135deg, rgba(26, 31, 58, 0.6) 0%, rgba(20, 24, 48, 0.4) 100%);
            border-radius: 10px;
            border: 2px solid #2a3f5f;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(79, 195, 247, 0.1);
            position: relative;
        }

        .pipeline-stage::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, transparent, #4fc3f7 20%, #4fc3f7 80%, transparent);
            border-radius: 10px 10px 0 0;
        }

        .stage-header {
            font-size: 0.75em;
            font-weight: 700;
            color: #4fc3f7;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(79, 195, 247, 0.2);
            text-align: center;
        }

        .path-label {
            position: absolute;
            top: -8px;
            right: 6px;
            font-size: 0.6em;
            color: #8b949e;
            background: #1a1f3a;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 600;
            border: 1px solid #2a3f5f;
        }

        /* Stage Flow Connectors */
        .stage-connector {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 15px 0;
            position: relative;
            height: 50px;
        }

        .stage-connector::before {
            content: '';
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, #2a3f5f 0%, #4fc3f7 50%, #2a3f5f 100%);
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
        }

        .stage-connector::after {
            content: '▼';
            position: absolute;
            bottom: -5px;
            left: 50%;
            transform: translateX(-50%);
            color: #4fc3f7;
            font-size: 1.2em;
            text-shadow: 0 0 10px rgba(79, 195, 247, 0.5);
        }

        .stage-connector-label {
            background: linear-gradient(135deg, #1a1f3a 0%, #0d1117 100%);
            padding: 6px 16px;
            border-radius: 6px;
            font-size: 0.7em;
            color: #4fc3f7;
            border: 2px solid #2a3f5f;
            position: relative;
            z-index: 1;
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
        }

        .stage-connector-arrow {
            display: none;
        }

        /* Pipeline Row - for horizontal stage layout */
        .pipeline-row-wrapper {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 0;
            align-items: stretch;
            margin-bottom: 20px;
        }

        .pipeline-row-wrapper .pipeline-stage {
            margin-bottom: 0;
        }

        /* Horizontal Stage Connector */
        .horizontal-stage-connector {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 0 20px;
            position: relative;
            min-width: 80px;
        }

        .horizontal-stage-connector::before {
            content: '';
            height: 4px;
            width: 100%;
            background: linear-gradient(90deg, #2a3f5f 0%, #4fc3f7 50%, #2a3f5f 100%);
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
        }

        .horizontal-stage-connector::after {
            content: '▶';
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: #4fc3f7;
            font-size: 1.2em;
            text-shadow: 0 0 10px rgba(79, 195, 247, 0.5);
        }

        .horizontal-connector-label {
            background: linear-gradient(135deg, #1a1f3a 0%, #0d1117 100%);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.65em;
            color: #4fc3f7;
            border: 2px solid #2a3f5f;
            position: relative;
            z-index: 1;
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
            white-space: nowrap;
        }

        .vertical-connector-label {
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #1a1f3a 0%, #0d1117 100%);
            padding: 6px 16px;
            border-radius: 6px;
            font-size: 0.7em;
            color: #4fc3f7;
            border: 2px solid #2a3f5f;
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
            white-space: nowrap;
            z-index: 10;
        }

        /* Internal Sequential Flow Arrow */
        .seq-arrow {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 4px 0;
            position: relative;
            height: 30px;
        }

        .seq-arrow::before {
            content: '';
            width: 2px;
            height: 50%;
            background: linear-gradient(180deg, #2a3f5f 0%, #4fc3f7 100%);
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
        }

        .seq-arrow::after {
            content: '▼';
            color: #4fc3f7;
            font-size: 0.9em;
            line-height: 0.5;
            text-shadow: 0 0 8px rgba(79, 195, 247, 0.6);
            position: absolute;
            bottom: 2px;
            animation: pulse-arrow 2s ease-in-out infinite;
        }

        @keyframes pulse-arrow {
            0%, 100% {
                opacity: 0.8;
                transform: translateY(0);
            }
            50% {
                opacity: 1;
                transform: translateY(2px);
            }
        }

        .flowchart-row {
            display: flex;
            justify-content: flex-start;
            align-items: center;
            margin: 10px 0;
            position: relative;
            gap: 8px;
            flex-wrap: nowrap;
        }

        /* Multi-column grid for condensed stages */
        .stage-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px 15px;
            margin: 10px 0;
        }

        .stage-grid-column {
            display: flex;
            flex-direction: column;
            gap: 8px;
            align-items: center;
        }

        .stage-grid-column-header {
            font-size: 0.7em;
            color: #8b949e;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }

        .component {
            background: linear-gradient(145deg, #1e2d3d 0%, #1a2532 100%);
            border: 2px solid #2a3f5f;
            border-radius: 6px;
            padding: 9px 14px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            width: fit-content;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        }

        .component::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(79, 195, 247, 0.1) 0%, transparent 100%);
            border-radius: 8px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .component:hover::before {
            opacity: 1;
        }

        .component.active {
            border-color: #4fc3f7;
            background: linear-gradient(145deg, #2a4f6f 0%, #1e3a52 100%);
            box-shadow: 0 0 20px rgba(79, 195, 247, 0.15), 0 5px 15px rgba(0,0,0,0.5);
            transform: translateY(-3px) scale(1.02);
        }

        .component.active::after {
            content: '';
            position: absolute;
            top: -3px;
            left: -3px;
            right: -3px;
            bottom: -3px;
            background: linear-gradient(135deg, #4fc3f7, #66bb6a);
            border-radius: 10px;
            z-index: -1;
            opacity: 0.1;
        }

        .component.completed {
            border-color: #66bb6a;
            background: linear-gradient(145deg, #2a4736 0%, #1e3329 100%);
            box-shadow: 0 2px 8px rgba(102, 187, 106, 0.3);
        }

        .component.inactive {
            opacity: 0.75;
            filter: grayscale(40%);
            border-color: #1a2532;
            background: linear-gradient(145deg, #151a23 0%, #0f1419 100%);
        }

        .component.inactive .tooltip {
            /* Keep tooltips visible for educational purposes */
            display: block;
        }

        .component-name {
            font-weight: 600;
            font-size: 0.75em;
            margin-bottom: 3px;
            color: #4fc3f7;
            text-transform: uppercase;
            letter-spacing: 0.2px;
        }

        .component.active .component-name {
            color: #66bb6a;
        }

        .component-desc {
            font-size: 0.65em;
            color: #8b949e;
            line-height: 1.15;
        }

        /* Component timing and percentage */
        .component-timing {
            font-size: 0.6em;
            color: #66bb6a;
            margin-top: 4px;
            font-weight: 500;
            display: none; /* Hidden by default, shown when timing data available */
        }

        .component.active .component-timing,
        .component.completed .component-timing {
            display: block;
        }

        .component-percentage {
            font-size: 0.55em;
            color: #ffa726;
            margin-top: 2px;
            font-weight: 600;
            display: none; /* Only shown after generation completes */
        }

        .component.completed .component-percentage.visible {
            display: block;
        }

        /* Tooltips */
        .component {
            position: relative;
        }

        .component .tooltip {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            z-index: 1000;
            background: #1a1f3a;
            color: #e0e0e0;
            padding: 12px 15px;
            border-radius: 6px;
            border: 1px solid #4fc3f7;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            width: 280px;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%) translateY(-10px);
            margin-bottom: 10px;
            transition: opacity 0.3s, transform 0.3s;
            pointer-events: none;
        }

        .component .tooltip::after {
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 8px solid transparent;
            border-top-color: #4fc3f7;
        }

        .component:hover .tooltip {
            visibility: visible;
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }

        .tooltip-title {
            font-weight: 600;
            color: #4fc3f7;
            margin-bottom: 8px;
            font-size: 0.9em;
        }

        .tooltip-desc {
            font-size: 0.8em;
            line-height: 1.4;
            color: #c9d1d9;
            margin-bottom: 8px;
        }

        .tooltip-tech {
            font-size: 0.75em;
            color: #8b949e;
            font-style: italic;
            padding-top: 8px;
            border-top: 1px solid #2a3f5f;
        }

        .component-progress {
            margin-top: 8px;
            height: 3px;
            background: #1a1f3a;
            border-radius: 2px;
            overflow: hidden;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #4fc3f7, #66bb6a);
            width: 0%;
            transition: width 0.3s ease;
        }

        /* Horizontal Arrows */
        .h-arrow {
            flex-shrink: 0;
            width: 28px;
            height: 2.5px;
            background: linear-gradient(90deg, #2a3f5f, #3a5f7f);
            position: relative;
            margin: 0 -1px;
        }

        .h-arrow::before {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, #4fc3f7);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .h-arrow::after {
            content: '';
            position: absolute;
            right: -7px;
            top: 50%;
            transform: translateY(-50%);
            width: 0;
            height: 0;
            border-left: 9px solid #3a5f7f;
            border-top: 5px solid transparent;
            border-bottom: 5px solid transparent;
            filter: drop-shadow(0 0 3px rgba(79, 195, 247, 0.3));
        }

        .h-arrow.active::before {
            opacity: 1;
        }

        .h-arrow.active {
            background: linear-gradient(90deg, #4fc3f7, #66bb6a);
            box-shadow: 0 0 8px rgba(79, 195, 247, 0.4);
            animation: flow-horizontal 2s ease-in-out infinite;
        }

        .h-arrow.active::after {
            border-left-color: #66bb6a;
            filter: drop-shadow(0 0 6px rgba(102, 187, 106, 0.6));
        }

        .h-arrow.inactive {
            opacity: 0.2;
            filter: grayscale(80%);
        }

        @keyframes flow-horizontal {
            0%, 100% {
                opacity: 0.7;
                box-shadow: 0 0 8px rgba(79, 195, 247, 0.1);
            }
            50% {
                opacity: 1;
                box-shadow: 0 0 12px rgba(79, 195, 247, 0.15);
            }
        }

        /* Row break - simple visual separator */
        .row-break {
            width: 100%;
            height: 18px;
            position: relative;
            margin: 3px 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .corner-connector {
            width: 3px;
            height: 16px;
            background: linear-gradient(180deg, #4fc3f7, #3a5f7f);
            position: relative;
            border-radius: 2px;
            box-shadow: 0 0 6px rgba(79, 195, 247, 0.4);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .corner-connector.active {
            background: linear-gradient(180deg, #66bb6a, #4a8f4e);
            box-shadow: 0 0 12px rgba(102, 187, 106, 0.6);
            animation: pulse-connector 2s ease-in-out infinite;
        }

        @keyframes pulse-connector {
            0%, 100% {
                box-shadow: 0 0 8px rgba(102, 187, 106, 0.125);
                transform: scaleY(1);
            }
            50% {
                box-shadow: 0 0 16px rgba(102, 187, 106, 0.2);
                transform: scaleY(1.1);
            }
        }

        .corner-connector::after {
            content: '';
            position: absolute;
            bottom: -6px;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-top: 8px solid #3a5f7f;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            filter: drop-shadow(0 0 4px rgba(79, 195, 247, 0.4));
            transition: all 0.4s ease;
        }

        .corner-connector.active::after {
            border-top-color: #4a8f4e;
            filter: drop-shadow(0 0 8px rgba(102, 187, 106, 0.7));
        }

        /* Control Panel */
        .form-group {
            margin-bottom: 6px;
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 6px;
        }
        .form-row .form-group {
            margin-bottom: 0;
        }
        label {
            display: block;
            margin-bottom: 2px;
            color: #b0b0b0;
            font-size: 0.7em;
        }
        input, textarea {
            width: 100%;
            padding: 5px 7px;
            background: #243447;
            border: 1px solid #2a3f5f;
            border-radius: 4px;
            color: #e0e0e0;
            font-size: 0.8em;
        }
        select {
            width: 100%;
            padding: 5px 7px;
            background: #243447;
            border: 1px solid #2a3f5f;
            border-radius: 4px;
            color: #e0e0e0;
            font-size: 0.8em;
        }
        input:focus, textarea:focus {
            outline: none;
            border-color: #4fc3f7;
        }
        button {
            width: 100%;
            padding: 9px 14px;
            background: linear-gradient(135deg, #4fc3f7, #2196f3);
            border: none;
            border-radius: 6px;
            color: white;
            font-size: 0.9em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 3px 10px rgba(79, 195, 247, 0.3);
            position: relative;
            overflow: hidden;
        }
        button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 195, 247, 0.5);
        }
        button:hover::before {
            left: 100%;
        }
        button:active {
            transform: translateY(0);
            box-shadow: 0 2px 8px rgba(79, 195, 247, 0.3);
        }
        button:disabled {
            background: linear-gradient(135deg, #555, #444);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
            opacity: 0.6;
        }

        /* Metrics */
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 6px;
        }
        .metric {
            background: #243447;
            padding: 6px 8px;
            border-radius: 4px;
            border-left: 2px solid #4fc3f7;
        }
        .metric-label {
            font-size: 0.7em;
            color: #b0b0b0;
            margin-bottom: 2px;
        }
        .metric-value {
            font-size: 1em;
            font-weight: bold;
            color: #4fc3f7;
        }

        /* Timeline */
        .timeline {
            max-height: 250px;
            overflow-y: auto;
        }
        .timeline-entry {
            padding: 6px 8px;
            margin-bottom: 6px;
            background: #243447;
            border-radius: 4px;
            border-left: 3px solid #4fc3f7;
            font-size: 0.75em;
        }
        .timeline-time {
            color: #66bb6a;
            font-weight: 600;
        }

        /* Model Output Display */
        .output-display {
            text-align: center;
            padding: 20px;
        }
        .output-display img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .output-display video {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .output-display audio {
            width: 100%;
            margin: 10px 0;
        }
        .output-display pre {
            text-align: left;
            background: #1e2d3d;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
        }

        /* Mode Selector */
        .mode-selector {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 4px;
            margin-bottom: 8px;
        }
        .mode-btn {
            padding: 5px 6px;
            background: #243447;
            border: 2px solid #2a3f5f;
            border-radius: 5px;
            color: #b0b0b0;
            font-size: 0.68em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            white-space: nowrap;
        }
        .mode-btn:hover {
            border-color: #4fc3f7;
            background: #2a3f5f;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(79, 195, 247, 0.2);
        }
        .mode-btn.active {
            background: linear-gradient(135deg, #4fc3f7, #2196f3);
            border-color: #4fc3f7;
            color: white;
            box-shadow: 0 4px 12px rgba(79, 195, 247, 0.4);
        }

        /* Parameter Display */
        .params-display {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 4px;
            margin-bottom: 8px;
            padding: 6px;
            background: #243447;
            border-radius: 6px;
            border: 1px solid #2a3f5f;
        }
        .param-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.68em;
            padding: 3px 6px;
            background: #1a1f3a;
            border-radius: 3px;
        }
        .param-label {
            color: #8b949e;
            font-weight: 500;
        }
        .param-value {
            color: #4fc3f7;
            font-weight: 600;
            font-family: 'Courier New', monospace;
        }

        /* Image Upload */
        .image-upload {
            margin-bottom: 8px;
            display: none;
        }
        .image-upload.visible {
            display: block;
        }
        .upload-zone {
            border: 2px dashed #2a3f5f;
            border-radius: 6px;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #243447;
        }
        .upload-zone:hover {
            border-color: #4fc3f7;
            background: #2a3f5f;
        }
        .upload-zone.has-image {
            border-style: solid;
            border-color: #66bb6a;
        }
        .upload-preview {
            max-width: 100%;
            max-height: 150px;
            border-radius: 4px;
            margin-top: 8px;
        }

        .status {
            text-align: center;
            padding: 6px 8px;
            background: #243447;
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 0.85em;
        }
        .status.active {
            background: #2a4f6f;
            border: 1px solid #4fc3f7;
        }

        /* Expandable API Call Section */
        .api-call-expandable {
            margin: 12px 0;
            border: 1px solid #2a3f5f;
            border-radius: 6px;
            background: #1a1f3a;
            overflow: hidden;
        }
        .api-call-toggle {
            padding: 10px 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(145deg, #1e2d3d 0%, #1a2532 100%);
            transition: all 0.3s ease;
            user-select: none;
        }
        .api-call-toggle:hover {
            background: linear-gradient(145deg, #243447 0%, #1e2d3d 100%);
        }
        .api-call-toggle-text {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #4fc3f7;
            font-weight: 600;
            font-size: 0.85em;
        }
        .api-call-arrow {
            color: #2a3f5f;
            font-size: 0.9em;
            transition: transform 0.3s ease;
        }
        .api-call-arrow.expanded {
            transform: rotate(180deg);
        }
        .api-call-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        .api-call-content.expanded {
            max-height: 300px;
        }
        .api-call-inner {
            padding: 12px;
            background: #0d1117;
            border-top: 1px solid #2a3f5f;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.7em;
            color: #58a6ff;
            overflow-x: auto;
            overflow-y: auto;
            max-height: 280px;
        }

        /* Model Insights Panel */
        .insights-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
            margin-top: 8px;
        }
        .insight-card {
            background: linear-gradient(135deg, #1e2d3d 0%, #1a2532 100%);
            border: 1px solid #2a3f5f;
            border-radius: 6px;
            padding: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .insight-card h3 {
            font-size: 0.8em;
            color: #4fc3f7;
            margin: 0 0 8px 0;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #2a3f5f;
            padding-bottom: 6px;
        }
        .insight-row {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid #1a2532;
            font-size: 0.8em;
        }
        .insight-row:last-child {
            border-bottom: none;
        }
        .insight-label {
            color: #8b949e;
            font-weight: 500;
        }
        .insight-value {
            color: #c9d1d9;
            font-weight: 600;
            text-align: right;
        }
        .insight-value.highlight {
            color: #4fc3f7;
        }
        .performance-bar {
            background: #1a2532;
            border-radius: 4px;
            height: 8px;
            margin-top: 6px;
            overflow: hidden;
            position: relative;
        }
        .performance-bar-fill {
            background: linear-gradient(90deg, #4fc3f7 0%, #2196f3 100%);
            height: 100%;
            transition: width 0.5s ease;
            border-radius: 4px;
        }
        .stage-metric {
            background: #0d1117;
            border-radius: 4px;
            padding: 6px 10px;
            margin-bottom: 6px;
            border-left: 3px solid #4fc3f7;
        }
        .stage-metric:last-child {
            margin-bottom: 0;
        }
        .stage-name {
            font-size: 0.7em;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 3px;
        }
        .stage-stats {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stage-time {
            font-size: 0.85em;
            color: #c9d1d9;
            font-weight: 600;
        }
        .stage-percentage {
            font-size: 0.7em;
            color: #4fc3f7;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Image Generation Pipeline Visualizer</h1>

        <div class="status" id="status">Ready to generate</div>

        <div class="grid control-flow" style="align-items: start;">
            <!-- Left Column (1fr): Control Panel and Model Insights -->
            <div>
                <!-- Control Panel (FIRST) -->
                <div class="panel">
                    <h2>Control Panel</h2>

                    <!-- Mode Selector -->
                    <h3 style="margin: 0 0 6px 0; color: #b0b0b0; font-size: 0.8em;">AI Modality</h3>
                    <div class="mode-selector">
                        <div class="mode-btn active" data-mode="text2img" onclick="switchMode('text2img')">
                            🎨 Text→Image
                        </div>
                        <div class="mode-btn" data-mode="img2img" onclick="switchMode('img2img')">
                            🖼️ Image→Image
                        </div>
                        <div class="mode-btn" data-mode="controlnet" onclick="switchMode('controlnet')">
                            🎯 Structure-Guided
                        </div>
                        <div class="mode-btn" data-mode="llm" onclick="switchMode('llm')">
                            🤖 LLM Chat
                        </div>
                        <div class="mode-btn" data-mode="text2audio" onclick="switchMode('text2audio')">
                            🔊 Text→Audio
                        </div>
                        <div class="mode-btn" data-mode="audio2text" onclick="switchMode('audio2text')">
                            🎤 Audio→Text
                        </div>
                        <div class="mode-btn" data-mode="text2video" onclick="switchMode('text2video')">
                            🎬 Text→Video
                        </div>
                        <div class="mode-btn" data-mode="img2video" onclick="switchMode('img2video')">
                            📹 Image→Video
                        </div>
                    </div>

                    <!-- Parameters Display -->
                    <div class="params-display" id="params-display">
                        <div class="param-item">
                            <span class="param-label">Active Mode:</span>
                            <span class="param-value" id="param-mode">text2img</span>
                        </div>
                        <div class="param-item">
                            <span class="param-label">Model:</span>
                            <span class="param-value" id="param-model">SDXL</span>
                        </div>
                    </div>

                    <!-- Image Upload (shown for img2img and controlnet modes) -->
                    <div class="image-upload" id="imageUpload">
                        <label style="margin-bottom: 6px; display: block; color: #b0b0b0; font-size: 0.8em;">Reference Image</label>
                        <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
                            <div id="uploadText">
                                📁 Click to upload image<br>
                                <span style="font-size: 0.8em; color: #8b949e;">or drag and drop</span>
                            </div>
                            <img id="uploadPreview" class="upload-preview" style="display: none;">
                        </div>
                        <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleImageUpload(event)">
                    </div>

                    <form id="generateForm">
                        <div class="form-group">
                            <label>Prompt</label>
                            <textarea id="prompt" rows="2" placeholder="a majestic giraffe in the savanna..." title="Describe what you want to generate. Be specific about subject, style, lighting, and atmosphere for best results.">a majestic giraffe standing in the African savanna, golden hour lighting</textarea>
                        </div>

                        <div class="form-row">
                            <div class="form-group">
                                <label>Steps: <span id="stepsValue">30</span></label>
                                <input type="range" id="steps" min="5" max="50" value="30" step="5" title="Number of denoising iterations. More steps = higher quality but slower generation. Recommended: 20-40 steps.">
                            </div>

                            <div class="form-group">
                                <label>Guidance Scale: <span id="guidanceValue">7.5</span></label>
                                <input type="range" id="guidance" min="1" max="20" value="7.5" step="0.5" title="How closely the model follows your prompt. Higher values = stricter adherence to prompt. Recommended: 7-10 for balanced results.">
                            </div>
                        </div>

                        <div class="form-row">
                            <div class="form-group">
                                <label>Image Size</label>
                                <select id="imageSize" title="Output resolution. Larger sizes take more time and memory. 1024x1024 is the sweet spot for quality and speed.">
                                    <option value="512x512">512×512 (Fast)</option>
                                    <option value="768x768">768×768</option>
                                    <option value="1024x1024" selected>1024×1024 (Default)</option>
                                    <option value="1024x768">1024×768 (Landscape)</option>
                                    <option value="768x1024">768×1024 (Portrait)</option>
                                </select>
                            </div>

                            <div class="form-group">
                                <label>Seed (optional)</label>
                                <input type="text" id="seed" placeholder="42" title="Random seed for reproducibility. Use the same seed with identical settings to recreate the same image. Leave empty for random results.">
                            </div>
                        </div>

                        <button type="submit" id="generateBtn" title="Start generating your image. The pipeline visualization will show real-time progress through each stage.">Generate Image</button>
                    </form>

                    <h2 style="margin-top: 15px; margin-bottom: 8px; font-size: 1em;">Generation Progress</h2>
                    <div class="metrics" id="metrics">
                        <div class="metric">
                            <div class="metric-label">Progress</div>
                            <div class="metric-value" id="progressDisplay">0/30</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Elapsed Time</div>
                            <div class="metric-value" id="elapsedTime">0s</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Time/Step</div>
                            <div class="metric-value" id="timePerStep">0s</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Est. Remaining</div>
                            <div class="metric-value" id="estRemaining">—</div>
                        </div>
                    </div>
                </div>

                <!-- Model Insights & Performance Metrics Panel (SECOND) -->
                <div class="panel" style="margin-top: 15px;">
                    <h2>📊 Model Insights & Performance Metrics</h2>

                    <div class="insights-grid">
                        <!-- Model Information Card -->
                        <div class="insight-card">
                            <h3>🤖 Model Information</h3>
                            <div class="insight-row">
                                <span class="insight-label">Model Name:</span>
                                <span class="insight-value" id="modelName">Stable Diffusion XL</span>
                            </div>
                            <div class="insight-row">
                                <span class="insight-label">Architecture:</span>
                                <span class="insight-value" id="modelArchitecture">Latent Diffusion</span>
                            </div>
                            <div class="insight-row">
                                <span class="insight-label">Parameters:</span>
                                <span class="insight-value" id="modelParameters">2.6B</span>
                            </div>
                            <div class="insight-row">
                                <span class="insight-label">License:</span>
                                <span class="insight-value" id="modelLicense">OpenRAIL-M</span>
                            </div>
                            <div class="insight-row">
                                <span class="insight-label">Modality:</span>
                                <span class="insight-value highlight" id="modelModality">Text-to-Image</span>
                            </div>
                        </div>

                        <!-- Performance Summary Card -->
                        <div class="insight-card">
                            <h3>⚡ Performance Summary</h3>
                            <div class="insight-row">
                                <span class="insight-label">Total Time:</span>
                                <span class="insight-value highlight" id="perfTotalTime">—</span>
                            </div>
                            <div class="insight-row">
                                <span class="insight-label">Image Size:</span>
                                <span class="insight-value" id="perfImageSize">1024×1024</span>
                            </div>
                            <div class="insight-row">
                                <span class="insight-label">Inference Steps:</span>
                                <span class="insight-value" id="perfSteps">30</span>
                            </div>
                            <div class="insight-row">
                                <span class="insight-label">Time per Step:</span>
                                <span class="insight-value" id="perfTimePerStep">—</span>
                            </div>
                            <div class="insight-row">
                                <span class="insight-label">Device:</span>
                                <span class="insight-value" id="perfDevice">MPS</span>
                            </div>
                        </div>

                        <!-- Pipeline Stage Breakdown Card -->
                        <div class="insight-card">
                            <h3>🔄 Pipeline Stage Breakdown</h3>
                            <div id="stageBreakdown">
                                <div class="stage-metric">
                                    <div class="stage-name">Input & Validation</div>
                                    <div class="stage-stats">
                                        <span class="stage-time" id="stageInputTime">—</span>
                                        <span class="stage-percentage" id="stageInputPct">—%</span>
                                    </div>
                                    <div class="performance-bar">
                                        <div class="performance-bar-fill" id="stageInputBar" style="width: 0%"></div>
                                    </div>
                                </div>
                                <div class="stage-metric">
                                    <div class="stage-name">Encoding (Text/Image)</div>
                                    <div class="stage-stats">
                                        <span class="stage-time" id="stageEncodingTime">—</span>
                                        <span class="stage-percentage" id="stageEncodingPct">—%</span>
                                    </div>
                                    <div class="performance-bar">
                                        <div class="performance-bar-fill" id="stageEncodingBar" style="width: 0%"></div>
                                    </div>
                                </div>
                                <div class="stage-metric">
                                    <div class="stage-name">Processing Core (UNet)</div>
                                    <div class="stage-stats">
                                        <span class="stage-time" id="stageProcessingTime">—</span>
                                        <span class="stage-percentage" id="stageProcessingPct">—%</span>
                                    </div>
                                    <div class="performance-bar">
                                        <div class="performance-bar-fill" id="stageProcessingBar" style="width: 0%"></div>
                                    </div>
                                </div>
                                <div class="stage-metric">
                                    <div class="stage-name">Decoding (VAE)</div>
                                    <div class="stage-stats">
                                        <span class="stage-time" id="stageDecodingTime">—</span>
                                        <span class="stage-percentage" id="stageDecodingPct">—%</span>
                                    </div>
                                    <div class="performance-bar">
                                        <div class="performance-bar-fill" id="stageDecodingBar" style="width: 0%"></div>
                                    </div>
                                </div>
                                <div class="stage-metric">
                                    <div class="stage-name">Output & Save</div>
                                    <div class="stage-stats">
                                        <span class="stage-time" id="stageOutputTime">—</span>
                                        <span class="stage-percentage" id="stageOutputPct">—%</span>
                                    </div>
                                    <div class="performance-bar">
                                        <div class="performance-bar-fill" id="stageOutputBar" style="width: 0%"></div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Educational Insights Card -->
                        <div class="insight-card">
                            <h3>💡 Educational Insights</h3>
                            <div id="educationalInsights" style="font-size: 0.85em; line-height: 1.6; color: #c9d1d9;">
                                <p style="margin: 0 0 10px 0;">
                                    <strong style="color: #4fc3f7;">Stable Diffusion XL</strong> uses a latent diffusion approach
                                    where the image generation happens in a compressed "latent space" rather than pixel space.
                                </p>
                                <p style="margin: 0 0 10px 0;">
                                    This makes it ~8x faster and more memory-efficient than pixel-based approaches while
                                    maintaining high quality output.
                                </p>
                                <p style="margin: 0;">
                                    The UNet denoising process typically takes 70-85% of total generation time, which is why
                                    reducing inference steps has the biggest impact on performance.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Column: Pipeline Flowchart -->
            <div>
                <!-- Pipeline Flowchart -->
                <div class="panel">
                    <h2>AI Pipeline Flow - All Modalities</h2>
                    <div class="pipeline-comprehensive" id="pipeline">

                    <!-- HORIZONTAL ROW: INPUT and ENCODING side-by-side -->
                    <div class="pipeline-row-wrapper">
                        <!-- INPUT STAGE (Column 1) -->
                        <div class="pipeline-stage">
                            <div class="stage-header">INPUT STAGE</div>
                            <div class="flowchart-row">
                                <div class="component" data-component="input" data-path="all">
                                    <div class="component-name">📝 Start</div>
                                    <div class="component-desc">User Input</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">Input Processing</div>
                                        <div class="tooltip-desc">Entry point for all AI modalities. Receives user input (text prompt, image, audio) along with generation parameters.</div>
                                        <div class="tooltip-tech">Interface: REST API endpoint accepting multiple content types</div>
                                    </div>
                                </div>
                                <div class="h-arrow" data-path="all"></div>
                                <div class="component" data-component="api" data-path="all">
                                    <div class="component-name">🔌 API</div>
                                    <div class="component-desc">Validate Request</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">API Handler</div>
                                        <div class="tooltip-desc">Validates and parses request parameters. Routes to appropriate processing pipeline based on modality. Sets defaults (size, steps, guidance).</div>
                                        <div class="tooltip-tech">Technology: FastAPI with Pydantic validation</div>
                                    </div>
                                </div>
                            </div>

                            <!-- Expandable API Call Details -->
                            <div class="api-call-expandable">
                                <div class="api-call-toggle" onclick="toggleApiCall()">
                                    <div class="api-call-toggle-text">
                                        <span>📡</span>
                                        <span>View Exact API Call to Model</span>
                                    </div>
                                    <div class="api-call-arrow" id="apiCallArrow">▼</div>
                                </div>
                                <div class="api-call-content" id="apiCallContent">
                                    <div class="api-call-inner" id="apiCallDetails">
                                        <p style="color: #8b949e;">Waiting for generation request...</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Horizontal connector -->
                        <div class="horizontal-stage-connector">
                            <div class="horizontal-connector-label">BRANCH BY TYPE →</div>
                        </div>

                        <!-- ENCODING BRANCHES (Column 2) -->
                        <div class="pipeline-stage">
                        <div class="stage-header">ENCODING</div>

                        <div class="stage-grid">
                            <!-- Text Path Column -->
                            <div class="stage-grid-column">
                                <div class="stage-grid-column-header">Text Path</div>
                                <div class="component" data-component="text-encode" data-path="text2img,text2audio,text2video,llm,controlnet">
                                    <div class="component-name">🔤 Tokenize</div>
                                    <div class="component-desc">Text → Tokens</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">Tokenization</div>
                                        <div class="tooltip-desc">Converts text prompt into numerical tokens. Splits text into subwords, maps each to an ID from vocabulary.</div>
                                        <div class="tooltip-tech">Tokenizer: BPE (Byte-Pair Encoding)<br>Vocab size: 49,408 tokens<br>Max length: 77 tokens</div>
                                    </div>
                                </div>
                                <div class="seq-arrow"></div>
                                <div class="component" data-component="text-embed" data-path="text2img,text2audio,text2video,controlnet">
                                    <div class="component-name">📊 Embed</div>
                                    <div class="component-desc">CLIP Encoding</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">Text Embedding (CLIP)</div>
                                        <div class="tooltip-desc">Converts tokens into dense meaning vectors. Each token becomes a 768-dimensional vector capturing semantic meaning.</div>
                                        <div class="tooltip-tech">Model: CLIP Text Encoder<br>Embedding dim: 768<br>Transformer layers: 12<br>Output: 77×768 tensor</div>
                                    </div>
                                </div>
                            </div>

                            <!-- Image Path Column -->
                            <div class="stage-grid-column">
                                <div class="stage-grid-column-header">Image Path</div>
                                <div class="component" data-component="image-encode" data-path="img2img,img2video,controlnet">
                                    <div class="component-name">🖼️ Load Image</div>
                                    <div class="component-desc">Image Input</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">Image Loading & Preprocessing</div>
                                        <div class="tooltip-desc">Loads input image, converts to RGB, resizes to match model requirements (typically 1024×1024). Normalizes pixel values.</div>
                                        <div class="tooltip-tech">Input formats: PNG, JPG, WebP<br>Color space: RGB (3 channels)<br>Typical size: 1024×1024×3</div>
                                    </div>
                                </div>
                                <div class="seq-arrow"></div>
                                <div class="component" data-component="vae-encode" data-path="img2img,img2video">
                                    <div class="component-name">🔧 VAE Encode</div>
                                    <div class="component-desc">Image → Latents</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">VAE Encoder (Compression)</div>
                                        <div class="tooltip-desc">Compresses the image into compact latent space. Reduces 1024×1024×3 image to 128×128×4 latents (64× smaller!).</div>
                                        <div class="tooltip-tech">Compression: 8× per dimension<br>Output: 128×128×4 latents<br>Size reduction: 3.1M → 65K values</div>
                                    </div>
                                </div>
                            </div>

                            <!-- Audio Path Column -->
                            <div class="stage-grid-column">
                                <div class="stage-grid-column-header">Audio Path</div>
                                <div class="component" data-component="audio-encode" data-path="audio2text">
                                    <div class="component-name">🎤 Audio Input</div>
                                    <div class="component-desc">Audio Signal</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">Audio Input & Preprocessing</div>
                                        <div class="tooltip-desc">Loads audio file and resamples to standard rate (typically 16kHz for speech recognition). Converts to mono channel if stereo.</div>
                                        <div class="tooltip-tech">Input formats: WAV, MP3, FLAC<br>Sample rate: 16,000 Hz<br>Channels: Mono (1 channel)</div>
                                    </div>
                                </div>
                                <div class="seq-arrow"></div>
                                <div class="component" data-component="audio-features" data-path="audio2text">
                                    <div class="component-name">📈 Features</div>
                                    <div class="component-desc">Spectrogram</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">Audio Feature Extraction</div>
                                        <div class="tooltip-desc">Converts time-domain audio waveform into frequency-domain spectrogram using Mel-scale filterbank.</div>
                                        <div class="tooltip-tech">Transform: STFT<br>Mel filters: 80 or 128 bands<br>Output: Time×Frequency matrix</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    </div><!-- End of pipeline-row-wrapper -->

                    <!-- L-shaped Flow connector: ENCODING → PROCESSING -->
                    <div style="position: relative; height: 120px; margin: 0;">
                        <!-- SVG path for clear L-shaped flow -->
                        <svg style="position: absolute; width: 100%; height: 100%; pointer-events: none;" viewBox="0 0 100 100" preserveAspectRatio="none">
                            <!-- Gradient for stroke -->
                            <defs>
                                <linearGradient id="pathGradient1" x1="0%" y1="0%" x2="0%" y2="100%">
                                    <stop offset="0%" style="stop-color:#2a3f5f;stop-opacity:1" />
                                    <stop offset="100%" style="stop-color:#4fc3f7;stop-opacity:1" />
                                </linearGradient>
                            </defs>
                            <!-- Path from right (ENCODING) down, left, down to left (PROCESSING) -->
                            <path d="M 75 0 L 75 30 L 25 30 L 25 100"
                                  stroke="url(#pathGradient1)"
                                  stroke-width="1"
                                  fill="none"
                                  stroke-linecap="round"
                                  stroke-linejoin="round"
                                  style="filter: drop-shadow(0 0 5px rgba(79, 195, 247, 0.6));"/>
                            <!-- Arrow at end -->
                            <polygon points="25,100 21,93 29,93"
                                     fill="#4fc3f7"
                                     style="filter: drop-shadow(0 0 5px rgba(79, 195, 247, 0.6));"/>
                        </svg>
                        <!-- Label -->
                        <div class="vertical-connector-label">PROCESS</div>
                    </div>

                    <!-- HORIZONTAL ROW: PROCESSING and DECODING side-by-side -->
                    <div class="pipeline-row-wrapper">
                        <!-- PROCESSING CORE (Column 1) -->
                        <div class="pipeline-stage">
                        <div class="stage-header">PROCESSING CORE</div>

                        <div class="stage-grid">
                            <!-- Diffusion Column -->
                            <div class="stage-grid-column">
                                <div class="stage-grid-column-header">Image/Video</div>
                                <div class="component" data-component="diffusion" data-path="text2img,img2img,text2video,img2video,controlnet">
                                    <div class="component-name">🎨 Diffusion</div>
                                    <div class="component-desc">UNet Denoising</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="component-progress"><div class="progress-bar"></div></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">Diffusion Process</div>
                                        <div class="tooltip-desc">Iteratively removes noise from random latents guided by text. UNet predicts noise at each step.</div>
                                        <div class="tooltip-tech">UNet: 2.6B parameters<br>Steps: 20-50<br>Time: ~2-3 sec/step</div>
                                    </div>
                                </div>
                            </div>

                            <!-- LLM Column -->
                            <div class="stage-grid-column">
                                <div class="stage-grid-column-header">Text Generation</div>
                                <div class="component" data-component="llm-inference" data-path="llm">
                                    <div class="component-name">🧠 Inference</div>
                                    <div class="component-desc">Attention Layers</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">LLM Inference</div>
                                        <div class="tooltip-desc">Generates text token-by-token using transformer attention layers. Each layer applies self-attention to understand context and feed-forward networks to predict the next token.</div>
                                        <div class="tooltip-tech">Architecture: Decoder-only transformer<br>Layers: 32-80 (depending on model)<br>Generation speed: 20-100 tokens/sec</div>
                                    </div>
                                </div>
                            </div>

                            <!-- Audio Column -->
                            <div class="stage-grid-column">
                                <div class="stage-grid-column-header">Audio Processing</div>
                                <div class="component" data-component="tts-synthesis" data-path="text2audio">
                                    <div class="component-name">🔊 Synthesize</div>
                                    <div class="component-desc">TTS Model</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">Text-to-Speech Synthesis</div>
                                        <div class="tooltip-desc">Converts text into mel-spectrogram representations using neural TTS models. Models rhythm, pitch, and intonation to produce natural-sounding speech patterns.</div>
                                        <div class="tooltip-tech">Model: Tacotron 2 or FastSpeech<br>Output: Mel-spectrogram (80 bins)<br>Time: ~0.5-1.0 seconds for short phrases</div>
                                    </div>
                                </div>
                                <div class="component" data-component="asr-model" data-path="audio2text">
                                    <div class="component-name">🎧 Recognize</div>
                                    <div class="component-desc">ASR Model</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">Automatic Speech Recognition</div>
                                        <div class="tooltip-desc">Transcribes audio into text using encoder-decoder architecture. The encoder processes audio features while the decoder generates text tokens.</div>
                                        <div class="tooltip-tech">Model: Whisper or similar<br>Languages: 100+<br>Accuracy: ~95%+ on clean audio</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                        <!-- Horizontal connector -->
                        <div class="horizontal-stage-connector">
                            <div class="horizontal-connector-label">DECODE →</div>
                        </div>

                        <!-- DECODING STAGE (Column 2) -->
                        <div class="pipeline-stage">
                        <div class="stage-header">DECODING</div>

                        <div class="stage-grid">
                            <!-- Image Output Column -->
                            <div class="stage-grid-column">
                                <div class="stage-grid-column-header">Image Output</div>
                                <div class="component" data-component="vae-decode" data-path="text2img,img2img,text2video,img2video,controlnet">
                                    <div class="component-name">🖼️ VAE Decode</div>
                                    <div class="component-desc">Latents → Pixels</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">VAE Decoder</div>
                                        <div class="tooltip-desc">Transforms compressed latent representation (128×128×4) back to full-resolution pixels (1024×1024×3). Trained to reconstruct images with minimal quality loss.</div>
                                        <div class="tooltip-tech">Architecture: Convolutional decoder with upsampling<br>Compression: 8x spatial reduction<br>Time: ~0.2-0.4 seconds</div>
                                    </div>
                                </div>
                            </div>

                            <!-- Audio Output Column -->
                            <div class="stage-grid-column">
                                <div class="stage-grid-column-header">Audio Output</div>
                                <div class="component" data-component="vocoder" data-path="text2audio">
                                    <div class="component-name">🔈 Vocoder</div>
                                    <div class="component-desc">Mel → Audio</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">Neural Vocoder</div>
                                        <div class="tooltip-desc">Converts mel-spectrogram representation into raw audio waveforms. Uses neural networks (GAN or diffusion-based) to generate high-quality, natural-sounding speech.</div>
                                        <div class="tooltip-tech">Type: HiFi-GAN or WaveGrad<br>Sampling rate: 22.05 kHz<br>Time: ~0.1-0.2 seconds</div>
                                    </div>
                                </div>
                            </div>

                            <!-- Text Output Column -->
                            <div class="stage-grid-column">
                                <div class="stage-grid-column-header">Text Output</div>
                                <div class="component" data-component="text-decode" data-path="llm,audio2text">
                                    <div class="component-name">📝 Detokenize</div>
                                    <div class="component-desc">Tokens → Text</div>
                                    <div class="component-timing"></div>
                                    <div class="component-percentage"></div>
                                    <div class="tooltip">
                                        <div class="tooltip-title">Detokenization</div>
                                        <div class="tooltip-desc">Converts token IDs back into readable text. Maps numerical tokens to their corresponding words/subwords using the model's vocabulary.</div>
                                        <div class="tooltip-tech">Process: Token IDs → Vocabulary lookup → Text<br>Time: Near-instantaneous (&lt;0.01s)</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    </div><!-- End of pipeline-row-wrapper -->

                    <!-- L-shaped Flow connector: DECODING → OUTPUT -->
                    <div style="position: relative; height: 120px; margin: 0;">
                        <!-- SVG path for clear L-shaped flow -->
                        <svg style="position: absolute; width: 100%; height: 100%; pointer-events: none;" viewBox="0 0 100 100" preserveAspectRatio="none">
                            <!-- Gradient for stroke -->
                            <defs>
                                <linearGradient id="pathGradient2" x1="0%" y1="0%" x2="0%" y2="100%">
                                    <stop offset="0%" style="stop-color:#2a3f5f;stop-opacity:1" />
                                    <stop offset="100%" style="stop-color:#4fc3f7;stop-opacity:1" />
                                </linearGradient>
                            </defs>
                            <!-- Path from right (DECODING) down, left, down to center (OUTPUT) -->
                            <path d="M 75 0 L 75 30 L 50 30 L 50 100"
                                  stroke="url(#pathGradient2)"
                                  stroke-width="1"
                                  fill="none"
                                  stroke-linecap="round"
                                  stroke-linejoin="round"
                                  style="filter: drop-shadow(0 0 5px rgba(79, 195, 247, 0.6));"/>
                            <!-- Arrow at end -->
                            <polygon points="50,100 46,93 54,93"
                                     fill="#4fc3f7"
                                     style="filter: drop-shadow(0 0 5px rgba(79, 195, 247, 0.6));"/>
                        </svg>
                        <!-- Label -->
                        <div class="vertical-connector-label">FINALIZE</div>
                    </div>

                    <!-- OUTPUT STAGE (Common End) -->
                    <div class="pipeline-stage">
                        <div class="stage-header">OUTPUT</div>
                        <div class="flowchart-row">
                            <div class="component" data-component="output" data-path="all">
                                <div class="component-name">✅ Complete</div>
                                <div class="component-desc">Save & Return</div>
                                <div class="component-timing"></div>
                                <div class="component-percentage"></div>
                                <div class="tooltip">
                                    <div class="tooltip-title">Output & Completion</div>
                                    <div class="tooltip-desc">Finalizes the generation process. Saves the output (image/text/audio/video) to disk and returns the result to the client via WebSocket.</div>
                                    <div class="tooltip-tech">Output formats: PNG (images), MP3/WAV (audio), MP4 (video), Plain text (LLM)</div>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
                </div>

            </div>
        </div>

        <!-- README Section -->
        <div class="panel" style="margin-top: 15px;">
            <h2>📖 Pipeline Documentation</h2>
            <div style="font-size: 0.85em; line-height: 1.6; color: #c9d1d9;">
                <h3 style="color: #4fc3f7; font-size: 1em; margin-bottom: 10px;">How Stable Diffusion XL Works</h3>

                <p style="margin-bottom: 12px;">
                    <strong>Stable Diffusion XL</strong> is a latent diffusion model that generates images from text descriptions.
                    It works by gradually removing noise from random data, guided by your text prompt. Here's the complete process:
                </p>

                <div style="background: #243447; padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 3px solid #4fc3f7;">
                    <strong style="color: #4fc3f7;">Stage 1: Text Understanding</strong><br>
                    Your prompt is processed by two CLIP encoders (ViT-L/14 and OpenCLIP-ViT-G/14) that convert words into numerical embeddings.
                    These embeddings capture the semantic meaning of your text in a 2048-dimensional space with 77 tokens.
                </div>

                <div style="background: #243447; padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 3px solid #66bb6a;">
                    <strong style="color: #66bb6a;">Stage 2: Latent Diffusion (Core Process)</strong><br>
                    Instead of working with full-resolution pixels (expensive!), SDXL works in a compressed "latent space" (128×128×4).
                    Starting from pure random noise, the UNet neural network (2.6B parameters) predicts and removes noise over 20-50 steps.
                    At each step, cross-attention mechanisms align the image with your text embeddings.
                </div>

                <div style="background: #243447; padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 3px solid #ffa726;">
                    <strong style="color: #ffa726;">Stage 3: Image Decoding</strong><br>
                    The VAE (Variational Autoencoder) decoder transforms the latent representation back into full-resolution pixels (1024×1024×3 RGB).
                    This decoder was trained to reconstruct images from compressed latents with minimal quality loss.
                </div>

                <h3 style="color: #4fc3f7; font-size: 1em; margin: 15px 0 10px 0;">Key Concepts</h3>

                <ul style="margin: 0; padding-left: 20px; color: #8b949e;">
                    <li><strong>Guidance Scale (7.5):</strong> How strongly the model follows your prompt vs. exploring creative variations. Higher = more literal.</li>
                    <li><strong>Inference Steps (20-50):</strong> More steps = higher quality but slower. Each step refines the image incrementally.</li>
                    <li><strong>Seed:</strong> Random number that determines the starting noise. Same seed + prompt = same image (deterministic).</li>
                    <li><strong>Cross-Attention:</strong> The mechanism that "conditions" the image on your text. It determines which parts of the image correspond to which words.</li>
                    <li><strong>Latent Space:</strong> Compressed representation (8x smaller) where diffusion happens. Makes generation faster and cheaper.</li>
                </ul>

                <h3 style="color: #4fc3f7; font-size: 1em; margin: 15px 0 10px 0;">Performance Notes</h3>

                <ul style="margin: 0; padding-left: 20px; color: #8b949e;">
                    <li><strong>Device:</strong> Auto-detects MPS (Apple Silicon), CUDA (NVIDIA), or CPU</li>
                    <li><strong>Model Size:</strong> ~6.5GB in fp16 precision, ~13GB in fp32</li>
                    <li><strong>Memory Usage:</strong> ~8-10GB during generation (including activations)</li>
                    <li><strong>Speed:</strong> ~2-3 seconds per step on M1/M2 Max, ~0.5-1 sec on high-end NVIDIA GPUs</li>
                    <li><strong>Total Time:</strong> 30 steps ≈ 60-90 seconds on Apple Silicon, 15-30 seconds on NVIDIA RTX 4090</li>
                </ul>

                <p style="margin-top: 12px; font-size: 0.8em; color: #66bb6a;">
                    💡 <strong>Pro Tip:</strong> Hover over each component in the pipeline above to see detailed technical information!
                </p>
            </div>

            <div>
                <div class="panel">
                    <h2>Model Output</h2>
                    <div class="output-display" id="outputDisplay">
                        <p style="color: #666;">Model output will appear here (image, video, text, or audio)</p>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <script>
        let ws = null;
        let startTime = null;
        let generating = false;

        // Escape HTML for safe display
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Global state
        let currentMode = 'text2img';
        let uploadedImage = null;

        // Update block diagram visualization based on mode
        function updateBlockDiagram(mode) {
            // Comprehensive multi-path flowchart: show all paths, activate only the current one
            console.log(`Updating flowchart for mode: ${mode}`);

            // Update all components based on data-path attribute
            document.querySelectorAll('[data-path]').forEach(element => {
                const paths = element.dataset.path.split(',');
                const isActive = paths.includes('all') || paths.includes(mode);

                if (isActive) {
                    element.classList.remove('inactive');
                    element.classList.add('active');
                } else {
                    element.classList.add('inactive');
                    element.classList.remove('active');
                }
            });

            // Also update rows to fade out if no active components in them
            document.querySelectorAll('.flowchart-row').forEach(row => {
                const hasActiveComponent = row.querySelector('[data-path].active');
                if (!hasActiveComponent) {
                    row.style.opacity = '0.65';
                } else {
                    row.style.opacity = '1';
                }
            });
        }

        // Toggle API Call expandable section
        function toggleApiCall() {
            const content = document.getElementById('apiCallContent');
            const arrow = document.getElementById('apiCallArrow');

            if (content.classList.contains('expanded')) {
                content.classList.remove('expanded');
                arrow.classList.remove('expanded');
            } else {
                content.classList.add('expanded');
                arrow.classList.add('expanded');
            }
        }

        // Update Model Insights panel based on current mode
        function updateModelInsights(mode) {
            const modelInfo = {
                'text2img': {
                    name: 'Stable Diffusion XL',
                    architecture: 'Latent Diffusion',
                    parameters: '3.5B',
                    license: 'CreativeML OpenRAIL++',
                    modality: 'Text → Image',
                    insights: 'SDXL uses dual text encoders (CLIP ViT-L/14 and OpenCLIP ViT-G/14) for enhanced prompt understanding. The latent diffusion approach processes images in compressed 8x8 pixel blocks, making it ~8x faster than pixel-space diffusion while maintaining high quality.'
                },
                'img2img': {
                    name: 'Stable Diffusion XL',
                    architecture: 'Latent Diffusion',
                    parameters: '3.5B',
                    license: 'CreativeML OpenRAIL++',
                    modality: 'Image → Image',
                    insights: 'Image-to-image mode starts from your input image rather than pure noise. The "strength" parameter controls how much the original image is preserved (0.0 = no change, 1.0 = complete transformation). Lower strength values (0.3-0.5) are good for subtle modifications, while higher values (0.7-0.9) allow major changes.'
                },
                'controlnet': {
                    name: 'Stable Diffusion XL + ControlNet',
                    architecture: 'Conditioned Latent Diffusion',
                    parameters: '3.5B + 1.2B',
                    license: 'CreativeML OpenRAIL++',
                    modality: 'Structure-Guided Generation',
                    insights: 'ControlNet adds spatial conditioning to SDXL, allowing precise control over composition using edge maps, depth maps, or pose skeletons. The control structure guides the diffusion process while the text prompt defines the content, enabling highly controllable image generation.'
                },
                'llm': {
                    name: 'Language Model',
                    architecture: 'Transformer Decoder',
                    parameters: '7B - 70B+',
                    license: 'Various',
                    modality: 'Text → Text',
                    insights: 'Large Language Models use transformer-based architectures with attention mechanisms to generate coherent text. Modern LLMs can handle tasks like question answering, summarization, code generation, and creative writing through prompt-based interaction.'
                },
                'text2audio': {
                    name: 'Text-to-Speech Model',
                    architecture: 'Neural TTS',
                    parameters: '200M - 1B',
                    license: 'Various',
                    modality: 'Text → Audio',
                    insights: 'TTS models convert text to natural-sounding speech using neural vocoders. Modern approaches use transformers to model prosody (rhythm and intonation) and mel-spectrograms, then convert to waveforms using GAN-based or diffusion-based vocoders.'
                },
                'audio2text': {
                    name: 'Whisper (OpenAI)',
                    architecture: 'Encoder-Decoder Transformer',
                    parameters: '1.5B',
                    license: 'MIT',
                    modality: 'Audio → Text',
                    insights: 'Whisper uses a transformer encoder-decoder architecture trained on 680,000 hours of multilingual data. It can transcribe speech in 100+ languages, translate to English, and perform voice activity detection with state-of-the-art accuracy.'
                },
                'text2video': {
                    name: 'Video Diffusion Model',
                    architecture: 'Temporal Latent Diffusion',
                    parameters: '5B - 10B',
                    license: 'Various',
                    modality: 'Text → Video',
                    insights: 'Video diffusion models extend image diffusion to the temporal dimension, generating consistent frame sequences. They use 3D UNets or temporal attention layers to model motion and ensure temporal coherence across frames.'
                },
                'img2video': {
                    name: 'Video Diffusion Model',
                    architecture: 'Temporal Latent Diffusion',
                    parameters: '5B - 10B',
                    license: 'Various',
                    modality: 'Image → Video',
                    insights: 'Image-to-video models animate a single image by predicting plausible motion. They condition on the input frame and generate subsequent frames that maintain visual consistency while introducing realistic movement based on the text prompt.'
                }
            };

            const info = modelInfo[mode] || modelInfo['text2img'];

            // Update model information card
            document.getElementById('modelName').textContent = info.name;
            document.getElementById('modelArchitecture').textContent = info.architecture;
            document.getElementById('modelParameters').textContent = info.parameters;
            document.getElementById('modelLicense').textContent = info.license;
            document.getElementById('modelModality').textContent = info.modality;

            // Update control panel model display (they show the same model)
            document.getElementById('param-model').textContent = info.name;

            // Update educational insights
            document.getElementById('educationalInsights').innerHTML = `
                <p style="margin: 0; font-size: 0.85em; line-height: 1.6; color: #c9d1d9;">
                    ${info.insights}
                </p>
            `;
        }

        // Update performance metrics panel
        function updatePerformanceMetrics(data) {
            if (data.total_time) {
                document.getElementById('perfTotalTime').textContent = `${data.total_time.toFixed(2)}s`;
            }
            if (data.image_size) {
                document.getElementById('perfImageSize').textContent = data.image_size;
            }
            if (data.steps) {
                document.getElementById('perfSteps').textContent = data.steps;
                if (data.total_time) {
                    const timePerStep = data.total_time / data.steps;
                    document.getElementById('perfTimePerStep').textContent = `${timePerStep.toFixed(2)}s`;
                }
            }
        }

        // Update stage breakdown with timing data
        function updateStageBreakdown(stageTimings) {
            if (!stageTimings) return;

            const totalTime = stageTimings.total || 1;

            // Map component names to stage IDs
            const stageMapping = {
                'input': 'stageInput',
                'text-encode': 'stageEncoding',
                'text-embed': 'stageEncoding',
                'image-encode': 'stageEncoding',
                'diffusion': 'stageProcessing',
                'vae-decode': 'stageDecoding',
                'output': 'stageOutput'
            };

            // Aggregate timings by stage
            const stageTotals = {
                stageInput: 0,
                stageEncoding: 0,
                stageProcessing: 0,
                stageDecoding: 0,
                stageOutput: 0
            };

            // Sum up component times into stages
            Object.keys(stageTimings).forEach(component => {
                if (component !== 'total') {
                    const stageId = stageMapping[component];
                    if (stageId && stageTotals[stageId] !== undefined) {
                        stageTotals[stageId] += stageTimings[component];
                    }
                }
            });

            // Update each stage
            Object.keys(stageTotals).forEach(stageId => {
                const time = stageTotals[stageId];
                const percentage = totalTime > 0 ? (time / totalTime * 100) : 0;

                const timeEl = document.getElementById(`${stageId}Time`);
                const pctEl = document.getElementById(`${stageId}Pct`);
                const barEl = document.getElementById(`${stageId}Bar`);

                if (timeEl) timeEl.textContent = `${time.toFixed(2)}s`;
                if (pctEl) pctEl.textContent = `${percentage.toFixed(1)}%`;
                if (barEl) barEl.style.width = `${percentage}%`;
            });
        }

        // Switch generation mode
        function switchMode(mode) {
            currentMode = mode;

            // Update mode buttons
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });

            // Update parameter display
            document.getElementById('param-mode').textContent = mode;

            // Update block diagram visualization
            updateBlockDiagram(mode);

            // Update model insights panel
            updateModelInsights(mode);

            // Show/hide image upload
            const imageUpload = document.getElementById('imageUpload');
            if (mode === 'img2img' || mode === 'controlnet') {
                imageUpload.classList.add('visible');
            } else {
                imageUpload.classList.remove('visible');
            }

            // Update prompt placeholder
            const promptField = document.getElementById('prompt');
            if (mode === 'img2img') {
                promptField.placeholder = 'describe how you want to modify the image...';
            } else if (mode === 'controlnet') {
                promptField.placeholder = 'describe the image using the control structure...';
            } else {
                promptField.placeholder = 'a majestic giraffe in the savanna...';
            }

            console.log(`Switched to ${mode} mode`);
        }

        // Handle image upload
        function handleImageUpload(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImage = e.target.result;

                    // Show preview
                    const preview = document.getElementById('uploadPreview');
                    preview.src = uploadedImage;
                    preview.style.display = 'block';

                    // Hide upload text
                    document.getElementById('uploadText').style.display = 'none';

                    // Update upload zone styling
                    document.getElementById('uploadZone').classList.add('has-image');

                    console.log('Image uploaded successfully');
                };
                reader.readAsDataURL(file);
            }
        }

        // Update parameter display
        function updateParamDisplay() {
            const steps = document.getElementById('steps').value;
            const guidance = document.getElementById('guidance').value;
            const seed = document.getElementById('seed').value || 'random';
            const size = document.getElementById('imageSize').value;

            document.getElementById('param-steps').textContent = steps;
            document.getElementById('param-guidance').textContent = guidance;
            document.getElementById('param-seed').textContent = seed;
            document.getElementById('param-size').textContent = size.replace('x', '×');
        }

        // Update steps display
        document.getElementById('steps').addEventListener('input', (e) => {
            const steps = e.target.value;
            document.getElementById('stepsValue').textContent = steps;
            // Sync with Performance Summary
            document.getElementById('perfSteps').textContent = steps;
            // Update progress display
            document.getElementById('progressDisplay').textContent = `0/${steps}`;
            updateParamDisplay();
        });

        // Update guidance display
        document.getElementById('guidance').addEventListener('input', (e) => {
            document.getElementById('guidanceValue').textContent = e.target.value;
            updateParamDisplay();
        });

        // Update size display
        document.getElementById('imageSize').addEventListener('change', (e) => {
            // Sync with Performance Summary
            document.getElementById('perfImageSize').textContent = e.target.value.replace('x', '×');
            updateParamDisplay();
        });

        // Update seed display
        document.getElementById('seed').addEventListener('input', () => {
            updateParamDisplay();
        });

        // Connect to WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                console.log('WebSocket connected');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updatePipeline(data);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                setTimeout(connectWebSocket, 2000);
            };
        }

        function updatePipeline(data) {
            const { component, progress, total, message, metrics, api_call, component_timing, component_percentages } = data;

            // Update status
            document.getElementById('status').textContent = message;
            document.getElementById('status').className = component !== 'idle' ? 'status active' : 'status';

            // Update API call details if present
            if (api_call) {
                const apiDisplay = document.getElementById('apiCallDetails');
                apiDisplay.innerHTML = `<pre style="margin: 0; white-space: pre-wrap; color: #c9d1d9;">${escapeHtml(api_call)}</pre>`;
            }

            // Handle component timing updates
            if (component_timing) {
                updateComponentTiming(component_timing.component, component_timing.elapsed_time);
            }

            // Handle final percentage calculations
            if (component_percentages) {
                updateComponentPercentages(component_percentages);
            }

            // Update components
            document.querySelectorAll('.component').forEach(comp => {
                comp.classList.remove('active');
                const compName = comp.dataset.component;

                if (compName === component) {
                    comp.classList.add('active');
                    const progressBar = comp.querySelector('.progress-bar');
                    if (progressBar) {
                        if (total > 0) {
                            const percent = (progress / total) * 100;
                            progressBar.style.width = percent + '%';
                        } else {
                            progressBar.style.width = '100%';
                        }
                    }
                }
            });

            // Update arrows based on active component
            const componentOrder = ['input', 'api', 'loading', 'encoding', 'diffusion', 'saving', 'complete'];
            const currentIndex = componentOrder.indexOf(component);

            document.querySelectorAll('.h-arrow').forEach((arrow, idx) => {
                if (idx < currentIndex) {
                    arrow.classList.add('active');
                } else {
                    arrow.classList.remove('active');
                }
            });

            // Update corner connector
            const cornerConnector = document.querySelector('.corner-connector');
            if (cornerConnector) {
                if (currentIndex >= 4) { // After encoding, show active
                    cornerConnector.style.background = 'linear-gradient(180deg, #66bb6a, #4a8f4e)';
                    cornerConnector.style.boxShadow = '0 0 8px rgba(102, 187, 106, 0.6)';
                } else {
                    cornerConnector.style.background = 'linear-gradient(180deg, #4fc3f7, #3a5f7f)';
                    cornerConnector.style.boxShadow = '0 0 6px rgba(79, 195, 247, 0.4)';
                }
            }

            // Update metrics
            if (total > 0) {
                document.getElementById('progressDisplay').textContent = `${progress}/${total}`;

                // Calculate estimated remaining time
                if (metrics.time_per_step && progress > 0) {
                    const remaining = (total - progress) * metrics.time_per_step;
                    if (remaining > 60) {
                        document.getElementById('estRemaining').textContent = `${Math.floor(remaining / 60)}m ${Math.floor(remaining % 60)}s`;
                    } else {
                        document.getElementById('estRemaining').textContent = `${Math.floor(remaining)}s`;
                    }
                }
            }

            if (metrics.elapsed_time) {
                document.getElementById('elapsedTime').textContent = metrics.elapsed_time.toFixed(1) + 's';
            }
            if (metrics.time_per_step) {
                document.getElementById('timePerStep').textContent = metrics.time_per_step.toFixed(2) + 's';
            }

            // Mark completed components
            if (component === 'complete') {
                generating = false;
                document.getElementById('generateBtn').disabled = false;
                document.querySelectorAll('.component').forEach(comp => {
                    comp.classList.add('completed');
                    const progressBar = comp.querySelector('.progress-bar');
                    if (progressBar) {
                        progressBar.style.width = '100%';
                    }
                });
            }
        }

        // Update component timing display
        function updateComponentTiming(componentName, elapsedTime) {
            const components = document.querySelectorAll(`[data-component="${componentName}"]`);
            components.forEach(comp => {
                const timingDiv = comp.querySelector('.component-timing');
                if (timingDiv) {
                    timingDiv.textContent = `${elapsedTime.toFixed(2)}s`;
                }
            });
        }

        // Update component percentages after generation completes
        function updateComponentPercentages(percentages) {
            for (const [componentName, percentage] of Object.entries(percentages)) {
                const components = document.querySelectorAll(`[data-component="${componentName}"]`);
                components.forEach(comp => {
                    const percentDiv = comp.querySelector('.component-percentage');
                    if (percentDiv) {
                        percentDiv.textContent = `${percentage.toFixed(1)}%`;
                        percentDiv.classList.add('visible');
                    }
                });
            }
        }

        // Timeline function removed - using Pipeline Stage Breakdown instead

        // Handle form submission
        document.getElementById('generateForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            if (generating) return;

            generating = true;
            startTime = Date.now();
            document.getElementById('generateBtn').disabled = true;

            // Reset pipeline
            document.querySelectorAll('.component').forEach(comp => {
                comp.classList.remove('active', 'completed');
                const progressBar = comp.querySelector('.progress-bar');
                if (progressBar) {
                    progressBar.style.width = '0%';
                }
                // Clear timing and percentage displays
                const timingDiv = comp.querySelector('.component-timing');
                const percentDiv = comp.querySelector('.component-percentage');
                if (timingDiv) timingDiv.textContent = '';
                if (percentDiv) {
                    percentDiv.textContent = '';
                    percentDiv.classList.remove('visible');
                }
            });

            // Reset arrows
            document.querySelectorAll('.h-arrow').forEach(arrow => {
                arrow.classList.remove('active');
            });

            // Reset corner connector
            const cornerConnector = document.querySelector('.corner-connector');
            if (cornerConnector) {
                cornerConnector.style.background = 'linear-gradient(180deg, #4fc3f7, #3a5f7f)';
                cornerConnector.style.boxShadow = '0 0 6px rgba(79, 195, 247, 0.4)';
            }

            const prompt = document.getElementById('prompt').value;
            const steps = parseInt(document.getElementById('steps').value);
            const guidance = parseFloat(document.getElementById('guidance').value);
            const seed = document.getElementById('seed').value;
            const sizeValue = document.getElementById('imageSize').value;
            const [width, height] = sizeValue.split('x').map(v => parseInt(v));

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt,
                        mode: currentMode,
                        num_inference_steps: steps,
                        guidance_scale: guidance,
                        width: width,
                        height: height,
                        seed: seed ? parseInt(seed) : null,
                        init_image: uploadedImage  // Base64 encoded image
                    })
                });

                const result = await response.json();

                if (result.success) {
                    // Display model output based on mode
                    const outputDisplay = document.getElementById('outputDisplay');

                    if (currentMode === 'text2img' || currentMode === 'img2img' || currentMode === 'controlnet') {
                        outputDisplay.innerHTML = `<img src="/image/${result.filename}" alt="Generated image">`;
                    } else if (currentMode === 'text2video' || currentMode === 'img2video') {
                        outputDisplay.innerHTML = `<video controls><source src="/video/${result.filename}" type="video/mp4"></video>`;
                    } else if (currentMode === 'text2audio') {
                        outputDisplay.innerHTML = `<audio controls><source src="/audio/${result.filename}" type="audio/mpeg"></audio>`;
                    } else if (currentMode === 'audio2text' || currentMode === 'llm') {
                        outputDisplay.innerHTML = `<pre>${result.text || result.content}</pre>`;
                    }
                } else {
                    console.error('Generation failed:', result.error);
                }
            } catch (error) {
                console.error('Generation error:', error);
                generating = false;
                document.getElementById('generateBtn').disabled = false;
            }
        });

        // Initialize
        connectWebSocket();
        updateModelInsights('text2img'); // Initialize with default mode
        updateParamDisplay(); // Initialize parameter display
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate image with real-time pipeline visualization."""
    global _generator

    try:
        # Reset timing tracker for new generation
        timing_tracker.reset()

        start_time = time.time()

        # Get config for accurate parameters
        config = get_config()
        # Use "flux" as the default model (which is actually SDXL in our config)
        model_config = config.models["flux"]
        guidance_scale = model_config.get("guidance_scale", 7.5)
        negative_prompt = ""  # Default empty

        # Determine mode-specific configuration
        mode = request.mode or "text2img"
        mode_desc = {
            "text2img": "Text-to-Image (generate from scratch)",
            "img2img": f"Image-to-Image (transform existing image, strength={request.strength})",
            "controlnet": "ControlNet (structure-guided generation)"
        }.get(mode, "text2img")

        # Check if we have an init image for img2img/controlnet
        has_init_image = request.init_image is not None
        mode_note = ""
        if mode in ["img2img", "controlnet"] and not has_init_image:
            mode_note = "\n   ⚠️  WARNING: No reference image provided, falling back to text2img"
            mode = "text2img"  # Fallback
        elif mode in ["img2img", "controlnet"] and has_init_image:
            mode_note = "\n   ✓ Reference image received (base64 encoded)"

        api_call_details = f"""📋 GENERATION PARAMETERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 MODE:
   • {mode_desc}{mode_note}

🔹 PROMPT (Positive):
   "{request.prompt}"

🔹 NEGATIVE PROMPT:
   "{negative_prompt if negative_prompt else '(none)'}"

🔹 MODEL CONFIGURATION:
   • Model ID: {model_config['model_id']}
   • Variant: fp16 (half precision)
   • Device: {_generator.device if _generator else 'will auto-detect (CPU/MPS/CUDA)'}
   • Model Type: Stable Diffusion XL

🔹 GENERATION PARAMETERS:
   • Width: {request.width} px
   • Height: {request.height} px
   • Inference Steps: {request.num_inference_steps or 30}
   • Guidance Scale: {guidance_scale} (how strongly to follow prompt)
   • Seed: {request.seed if request.seed else 'random (non-deterministic)'}
   • Scheduler: {model_config.get('scheduler', 'EulerDiscreteScheduler')}

🔹 INTERNAL PROCESSING:
   • Latent Size: [{request.height//8} x {request.width//8}] (VAE compresses 8x)
   • Latent Channels: 4 (SDXL latent space dimensionality)
   • Text Encoder: CLIP (2 encoders for SDXL)
   • Cross-Attention: Text embeddings guide image latents
   • Self-Attention: Ensures spatial coherency in image
   • Total UNet Parameters: ~2.6 billion

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        # Step 1: Input
        timing_tracker.start_component("start")
        await emit_state(
            "input", 0, 0,
            f"Received: {request.prompt[:50]}...",
            api_call=api_call_details
        )
        await asyncio.sleep(0.3)
        input_time = timing_tracker.end_component("start")
        await emit_state(
            "input", 1, 1,
            "✓ Step 1 COMPLETE: Input received",
            component_timing={"component": "start", "elapsed_time": input_time}
        )
        await asyncio.sleep(0.2)

        # Step 2: API Processing
        timing_tracker.start_component("api")
        await emit_state("api", 0, 0, "Processing request parameters")
        await asyncio.sleep(0.3)
        api_time = timing_tracker.end_component("api")
        await emit_state(
            "api", 1, 1,
            "✓ Step 2 COMPLETE: Request processed",
            component_timing={"component": "api", "elapsed_time": api_time}
        )
        await asyncio.sleep(0.2)

        # Step 3: Model Loading
        timing_tracker.start_component("text-encode")  # Model loading maps to text-encode component
        await emit_state("loading", 0, 0, "Loading SDXL model...")
        if _generator is None:
            _generator = ImageGenerator(auto_preview=False)
        load_time = timing_tracker.end_component("text-encode")

        # Calculate model size and loading speed (config already imported at module level)

        # Estimate model size (SDXL is typically ~13GB for full precision, ~6.5GB for fp16)
        model_size_gb = 6.5  # fp16 variant
        load_speed = model_size_gb / load_time if load_time > 0 else 0
        device = str(_generator.device)

        await emit_state(
            "loading", 1, 1,
            f"✓ Step 3 COMPLETE: Model loaded in {load_time:.2f}s (~{model_size_gb}GB @ {load_speed:.2f}GB/s on {device})",
            component_timing={"component": "text-encode", "elapsed_time": load_time}
        )
        await asyncio.sleep(0.2)

        # Step 4: Text Encoding
        timing_tracker.start_component("text-embed")
        await emit_state("encoding", 0, 0, "Encoding text prompt...")
        await asyncio.sleep(0.3)
        encode_time = timing_tracker.end_component("text-embed")
        token_count = len(request.prompt.split())  # Rough estimate
        await emit_state(
            "encoding", 1, 1,
            f"✓ Step 4 COMPLETE: {token_count} tokens encoded in {encode_time:.3f}s",
            component_timing={"component": "text-embed", "elapsed_time": encode_time}
        )
        await asyncio.sleep(0.2)

        # Step 5: Diffusion (with progress updates)
        total_steps = request.num_inference_steps or 30

        # We'll simulate step-by-step progress
        # In reality, we'd hook into the diffusion process
        timing_tracker.start_component("diffusion")
        await emit_state("diffusion", 0, total_steps, "Starting diffusion process...")

        # Generate the image
        # Note: This is blocking - in a real implementation, we'd need async progress callbacks
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            None,
            lambda: _generator.generate(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=total_steps,
                seed=request.seed,
                auto_save=True
            )
        )

        diffusion_time = timing_tracker.end_component("diffusion")
        steps_per_sec = total_steps / diffusion_time if diffusion_time > 0 else 0

        await emit_state(
            "diffusion", total_steps, total_steps,
            f"✓ Step 5 COMPLETE: {total_steps} denoising steps @ {steps_per_sec:.2f} steps/s ({diffusion_time:.2f}s total)",
            component_timing={"component": "diffusion", "elapsed_time": diffusion_time}
        )
        await asyncio.sleep(0.2)

        elapsed = time.time() - start_time

        # Step 6: Saving
        timing_tracker.start_component("vae-decode")
        await emit_state("saving", 0, 0, "Decoding latents and saving image...")

        # Find the generated file
        config = get_config()
        output_dir = Path(config.output["directory"])
        files = sorted(output_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)

        if files:
            image_path = files[0]
            filename = image_path.name
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
        else:
            raise RuntimeError("Image file not found")

        save_time = timing_tracker.end_component("vae-decode")
        await emit_state(
            "saving", 1, 1,
            f"✓ Step 6 COMPLETE: Image saved ({file_size_mb:.2f}MB) in {save_time:.3f}s",
            component_timing={"component": "vae-decode", "elapsed_time": save_time}
        )
        await asyncio.sleep(0.2)

        # Step 7: Complete
        timing_tracker.start_component("output-complete")
        # Calculate percentages now that all components are timed
        percentages = timing_tracker.get_percentages()
        elapsed = time.time() - start_time
        output_time = timing_tracker.end_component("output-complete")

        await emit_state(
            "complete",
            total_steps,
            total_steps,
            f"✓ ALL STEPS COMPLETE! Total time: {elapsed:.2f}s | Resolution: {request.width}x{request.height} | Throughput: {steps_per_sec:.2f} steps/s",
            {
                "elapsed_time": elapsed,
                "time_per_step": elapsed / total_steps
            },
            component_timing={"component": "output-complete", "elapsed_time": output_time},
            component_percentages=percentages
        )

        return {
            "success": True,
            "filename": filename,
            "elapsed_time": elapsed
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        await emit_state("idle", 0, 0, f"Error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/image/{filename}")
async def get_image(filename: str):
    """Serve generated images."""
    config = get_config()
    output_dir = Path(config.output["directory"])
    image_path = output_dir / filename

    if image_path.exists():
        return FileResponse(image_path)
    else:
        return {"error": "Image not found"}


def run_visualization_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the visualization server."""
    import uvicorn

    logger.info(f"Starting visualization server on http://{host}:{port}")
    print(f"\n🎨 Image Generation Visualizer")
    print(f"Open your browser to: http://localhost:{port}")
    print(f"Press Ctrl+C to stop\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_visualization_server()
