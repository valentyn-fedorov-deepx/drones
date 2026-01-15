# gl_launcher.py
import os, sys

# 1) Знести усе, що тягне GLES/EGL/Wayland/софтверний GL
for k in [
    "LIBGL_ALWAYS_SOFTWARE", "MESA_LOADER_DRIVER_OVERRIDE", "QT_OPENGL",
    "QT_QUICK_BACKEND", "QSG_RHI_BACKEND", "QT_ANGLE_PLATFORM",
    "WAYLAND_DISPLAY"
]:
    os.environ.pop(k, None)

# 2) Примусово X11/XWayland + desktop GL через GLX, без RHI для QWidget
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "d3d12"

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_OPENGL"] = "desktop"
os.environ["QT_XCB_GL_INTEGRATION"] = "glx"
os.environ["QT_WIDGETS_RHI"] = "0"

print(">> ENV sanity:", {k:v for k,v in os.environ.items()
      if any(x in k for x in ["QT_", "LIBGL", "MESA", "WAYLAND", "EGL", "QSG"])})

# 3) Далі — Qt
from PyQt6.QtGui import QSurfaceFormat, QOpenGLContext
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

# Desktop OpenGL 3.3 Core — до QApplication!
fmt = QSurfaceFormat()
fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
fmt.setVersion(3, 3)
fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
QSurfaceFormat.setDefaultFormat(fmt)

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

class GLTest(QOpenGLWidget):
    def initializeGL(self):
        ctx = QOpenGLContext.currentContext()
        f = ctx.format()
        print("=== OpenGL context created ===")
        print("RenderableType:", f.renderableType())  # 0=Default, 1=OpenGL, 2=OpenGLES
        print("Version:", f.majorVersion(), f.minorVersion())
        print("Profile:", f.profile())                # 0=NoProfile,1=Core,2=Compatibility
        print("SwapBehavior:", f.swapBehavior())
    def paintGL(self): pass

app = QApplication(sys.argv)
print("QPA platform =", app.platformName())  # очікуємо 'xcb'

w = QMainWindow()
w.setCentralWidget(GLTest())
w.resize(640, 480)
w.show()
sys.exit(app.exec())
