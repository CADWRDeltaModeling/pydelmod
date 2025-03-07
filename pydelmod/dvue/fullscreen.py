import panel as pn
import param
from panel.reactive import ReactiveHTML

pn.extension("gridstack")

CSS = """
:not(:root):fullscreen::backdrop {
    background: white;
}
.fullscreen-button {
  position: absolute;
  top: 0px;
  right: 0px;
  width: 24px;
  height: 24px;
  z-index: 10000;
  opacity: 0;
  transition-delay: 0.5s;
  transition: 0.5s;
  cursor: pointer;
}

.fullscreen-button:hover {
  transition: 0.5s;
  opacity: 1;
}

.fullscreen-button:focus {
  opacity: 1;
}
.pn-container, .object-container {
    height: 100%;
    width: 100%;
}
"""


class FullScreen(ReactiveHTML):
    object = param.Parameter()

    def __init__(self, object, **params):
        super().__init__(object=object, **params)

    _template = """
<div id="pn-container" class="pn-container">
    <span id="button" class="fullscreen-button" onclick="${script('maximize')}">
        <svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 18 18">
    <path d="M4.5 11H3v4h4v-1.5H4.5V11zM3 7h1.5V4.5H7V3H3v4zm10.5 6.5H11V15h4v-4h-1.5v2.5zM11 3v1.5h2.5V7H15V3h-4z"></path>
        </svg>
    </span>
    <div id="object_el" class="object-container">${object}</div>
</div>
"""
    _stylesheets = [CSS]
    _scripts = {
        "maximize": """
function isFullScreen() {
  return (
    document.fullscreenElement ||
    document.webkitFullscreenElement ||
    document.mozFullScreenElement ||
    document.msFullscreenElement
  )
}
function exitFullScreen() {
  if (document.exitFullscreen) {
    document.exitFullscreen()
  } else if (document.mozCancelFullScreen) {
    document.mozCancelFullScreen()
  } else if (document.webkitExitFullscreen) {
    document.webkitExitFullscreen()
  } else if (document.msExitFullscreen) {
    document.msExitFullscreen()
  }
}
function requestFullScreen(element) {
  if (element.requestFullscreen) {
    element.requestFullscreen()
  } else if (element.mozRequestFullScreen) {
    element.mozRequestFullScreen()
  } else if (element.webkitRequestFullscreen) {
    element.webkitRequestFullscreen(Element.ALLOW_KEYBOARD_INPUT)
  } else if (element.msRequestFullscreen) {
    element.msRequestFullscreen()
  }
}

function toggleFullScreen() {
  if (isFullScreen()) {
    exitFullScreen()
  } else {
    requestFullScreen(button.parentElement)
  }
}
toggleFullScreen()
"""
    }
