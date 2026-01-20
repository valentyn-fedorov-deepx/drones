"""Lightweight real-time drone detection + tracking package.

This module provides a minimal, self-contained stack for drone detection and
tracking in real time, without SAM, silhouette classification or complex
orchestration. It is intentionally simple and focused on:

- a thin YOLO-based detector;
- a lightweight tracker;
- a synchronous pipeline that decides when to run detection.
"""
