import asyncio
import socket
import shutil
from av import VideoFrame
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.exceptions import InvalidStateError
from aiortc.contrib.media import MediaPlayer
from deployment.networking_utils import get_local_ip
from deployment.webrtc_streaming.video_track import NumpyVideoTrack
from deployment.remote_camera_management.server import CameraServer

DEFAULT_SAVE_PATH = "/mnt/larger_disk/recordings"
DEFAULT_PORT = 8080

class CameraServerSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CameraServer(DEFAULT_SAVE_PATH, as_standalone=False)
        return cls._instance

# Store active peer connections
pcs = set()

async def index(request):
    """Serve the HTML client page"""
    content = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Stream</title>
</head>
<body>
    <h1>Video Stream</h1>
    <video id="video" autoplay playsinline></video>
    <br>
    <label>Save Path:
        <input type="text" id="save-path" value="/mnt/larger_disk/recordings">
    </label>
    <button id="set-save-path">Set Save Path</button>
    <br><br>

    <button id="restart-stream">Reconnect Camera</button>
    <br><br>

    <button id="start-rec">Start Recording</button>
    <button id="pause-rec" disabled>Pause Recording</button>
    <button id="stop-rec" disabled>Stop Recording</button>
    <p>Free Space: <span id="free-space">--</span> GB</p>

    <script>
        const video = document.getElementById('video');
        const btnSetSavePath = document.getElementById('set-save-path');
        const btnRestartStream = document.getElementById('restart-stream');
        const btnStartRec = document.getElementById('start-rec');
        const btnPauseRec = document.getElementById('pause-rec');
        const btnStopRec = document.getElementById('stop-rec');

        let pc = null;

        // ---- Button State Management ----
        function setRecordingButtonsState({start, pause, stop}) {
            btnStartRec.disabled = !start;
            btnPauseRec.disabled = !pause;
            btnStopRec.disabled = !stop;
        }

        // ---- Stream Init ----
        async function initStream() {
            const response = await fetch('/get-offer');
            if (!response.ok) return console.error("Failed to get offer");

            const offerData = await response.json();
            pc = new RTCPeerConnection();
            pc.addEventListener('track', evt => {
                if (evt.track.kind === 'video') {
                    video.srcObject = evt.streams[0];
                }
            });

            await pc.setRemoteDescription(offerData);
            const answer = await pc.createAnswer();
            await pc.setLocalDescription(answer);

            await fetch('/set-answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type
                })
            });
        }

        async function updateFreeSpace() {
            const res = await fetch('/free-space');
            if (res.ok) {
                const data = await res.json();
                document.getElementById('free-space').textContent = data.free_gb;
            }
        }

        // ---- Event Handlers ----
        btnSetSavePath.addEventListener('click', async () => {
            const path = document.getElementById('save-path').value;
            if (!path) return alert("Please enter a path");

            const response = await fetch('/set-save-path', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ path })
            });

            if (response.ok) {
                alert("Save path updated!");
                updateFreeSpace();
            } else {
                alert("Failed to update path");
            }
        });

        btnRestartStream.addEventListener('click', () => {
            if (pc) pc.close();
            video.srcObject = null;
            initStream();
        });

        btnStartRec.addEventListener('click', async () => {
            await fetch('/start-recording', { method: 'POST' });
            setRecordingButtonsState({ start: false, pause: true, stop: true });
        });

        btnPauseRec.addEventListener('click', async () => {
            await fetch('/pause-recording', { method: 'POST' });
            setRecordingButtonsState({ start: true, pause: false, stop: true });
        });

        btnStopRec.addEventListener('click', async () => {
            await fetch('/stop-recording', { method: 'POST' });
            setRecordingButtonsState({ start: true, pause: false, stop: false });
        });

        pc.addEventListener('track', evt => {
        // Some browsers deliver an empty `streams` array.
        if (evt.streams && evt.streams[0]) {
            video.srcObject = evt.streams[0];
        } else {
            const ms = new MediaStream();
            ms.addTrack(evt.track);
            video.srcObject = ms;
        }
        });
        pc.addEventListener('iceconnectionstatechange', () => {
            console.log('ICE state:', pc.iceConnectionState);
        });
        pc.addEventListener('connectionstatechange', () => {
            console.log('PC state:', pc.connectionState);
        });

        // ---- On page load ----
        window.addEventListener('load', () => {
            setRecordingButtonsState({ start: true, pause: false, stop: false });
            initStream();
            updateFreeSpace();
            setInterval(updateFreeSpace, 1000);
        });
    </script>
</body>
</html>
    """
    return web.Response(text=content, content_type="text/html")

def _make_pc(app):
    pc = RTCPeerConnection()
    app['pcs'].add(pc)

    @pc.on("connectionstatechange")
    async def _on_conn_state():
        print("PC state:", pc.connectionState)
        if pc.connectionState == "failed":
            try:
                await pc.close()
            finally:
                app['pcs'].discard(pc)

    return pc

# --- Stream control ---
async def ensure_stream_started(app):
    state = app['webrtc']

    if state.get('pc') and state['pc'].connectionState != "closed":
        return # already running

    last_frame_count = -1
    last_sub_folder = None
    if state.get('video_track'):
        last_frame_count = state['video_track'].frame_count
        last_sub_folder = state['video_track'].sub_folder

    if state.get('pc'):
        await state['pc'].close()
        app['pcs'].discard(state['pc'])

    video_track = NumpyVideoTrack(
        CameraServerSingleton.get_instance(),
        frame_count=last_frame_count,
        sub_folder=last_sub_folder
    )
    state['video_track'] = video_track

    pc = _make_pc(app)
    pc.addTrack(video_track)
    state['pc'] = pc


async def start_stream(request):
    state = request.app['webrtc']

    old_track = state.get('video_track')
    last_frame_count = -1
    last_sub_folder = ""
    if old_track:
        last_frame_count = old_track.frame_count
        last_sub_folder = old_track.sub_folder

    # Close & discard old pc if any
    if state.get('pc'):
        await state['pc'].close()
        request.app['pcs'].discard(state['pc'])
        state['pc'] = None

    # Create fresh track with preserved frame count
    video_track = NumpyVideoTrack(
        CameraServerSingleton.get_instance(),
        frame_count=last_frame_count,
        sub_folder=last_sub_folder
    )
    state['video_track'] = video_track

    pc = _make_pc(request.app)
    pc.addTrack(video_track)
    state['pc'] = pc

    return web.Response(text="Stream started")

async def restart_stream(request):
    await ensure_stream_started(request.app)
    return web.Response(text="Stream restarted")

# --- Recording control ---
async def start_recording(request):
    vt = request.app['webrtc'].get('video_track')
    if vt:
        vt.start_recording()
        return web.Response(text="Recording started")
    return web.Response(status=404, text="No video track")

async def pause_recording(request):
    vt = request.app['webrtc'].get('video_track')
    if vt:
        vt.stop_recording()
        return web.Response(text="Recording paused")
    return web.Response(status=404, text="No video track")

async def stop_recording(request):
    vt = request.app['webrtc'].get('video_track')
    if vt:
        vt.stop_recording()
        vt.schedule_new_recording()
        return web.Response(text="Recording stopped")
    return web.Response(status=404, text="No video track")


# --- WebRTC handshake ---
async def get_offer(request):
    await ensure_stream_started(request.app)

    pc = request.app['webrtc']['pc']
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

async def set_answer(request):
    state = request.app['webrtc']
    pc = state.get('pc')
    if not pc:
        return web.Response(status=404, text="No active stream")

    # Bail if the pc is already closed
    if getattr(pc, "connectionState", None) == "closed" or pc.signalingState == "closed":
        return web.Response(status=410, text="Connection is closed; restart required")

    data = await request.json()
    answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    try:
        await pc.setRemoteDescription(answer)
    except InvalidStateError:
        return web.Response(status=410, text="PC was closed before setting answer")

    return web.Response(text="Answer set")

async def set_save_path(request):
    data = await request.json()
    new_path = data.get("path")

    if not new_path:
        return web.Response(status=400, text="Path required")

    CameraServerSingleton.get_instance().set_save_path(new_path)

    request.app['save_path'] = new_path
    print(f"SAVE_PATH updated to: {new_path}")
    return web.Response(text="OK")

def get_save_path(app):
    return app.get('save_path', DEFAULT_SAVE_PATH)

async def on_startup(app):
    app['save_path'] = "/default/path"

async def cleanup_pcs(app):
    coros = [pc.close() for pc in list(app['pcs'])]
    await asyncio.gather(*coros, return_exceptions=True)
    app['pcs'].clear()

async def get_free_space(request):
    save_path = get_save_path(request.app)
    total, used, free = shutil.disk_usage(save_path)
    return web.json_response({
        "free_gb": round(free / (1024**3), 2),
        "path": save_path
    })

def create_app():
    app = web.Application()
    app.on_startup.append(on_startup)

    # Pre-create mutable containers to avoid DeprecationWarning
    app['webrtc'] = {'pc': None, 'video_track': None}
    app['pcs'] = set()

    app.router.add_get("/", index)
    app.router.add_post('/restart-stream', restart_stream)
    app.router.add_post('/start-recording', start_recording)
    app.router.add_post('/pause-recording', pause_recording)
    app.router.add_post('/stop-recording', stop_recording)
    app.router.add_get('/get-offer', get_offer)
    app.router.add_post('/set-answer', set_answer)
    app.router.add_post("/set-save-path", set_save_path)
    app.router.add_get("/free-space", get_free_space)

    app.on_shutdown.append(cleanup_pcs)
    return app


if __name__ == "__main__":
    ip = get_local_ip()
    app = create_app()
    print(f"Starting WebRTC server on http://{ip}:{DEFAULT_PORT}")
    print("Open this URL in a web browser to view the stream")
    web.run_app(app, host=ip, port=DEFAULT_PORT)
