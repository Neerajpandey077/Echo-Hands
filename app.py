"""
ECHO HANDS — Compact Real-Time Sign Language Communication Dashboard
====================================================================
Features:
- 100% Native Working Buttons (Space, Back, Undo, Speak/Enter, Clear)
- Zero-Shift Camera Layout: Holds the previous frame in memory during button actions
  so the camera never collapses, never blinks, and never shifts elements below!
- ONE Unified Message Screen: Words and spaces appear continuously in a single display.
- Complete 26-Letter ASL Finger Guide: Step-by-step finger positions for non-signers.
- 26 Indian Languages Selector: Full regional expansion pipeline ready for judges.
- Clean English display with zero vertical scrolling.
"""

import os
import sys
import time
import cv2
import importlib
import streamlit as st
import streamlit.components.v1 as components

_base_dir = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_base_dir, ".cache", "matplotlib"))
os.environ.setdefault("STREAMLIT_CONFIG_DIR", os.path.join(_base_dir, ".streamlit"))

import hand_detector, recognizer, word_builder as _wb_mod, tts as _tts_mod
importlib.reload(hand_detector); importlib.reload(recognizer)
importlib.reload(_wb_mod);       importlib.reload(_tts_mod)

from hand_detector import HandDetector, HAND_CONNECTIONS
from recognizer   import recognize_sign, SignStabilizer
from word_builder import WordBuilder
from tts          import SpeechManager, generate_browser_speech_html, format_natural_word

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECHO HANDS — Sign Language Communication",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS for Rock-Solid Single-Screen Layout (No Shift, No Scroll) ─────
st.markdown("""
<style>
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0.2rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 99% !important;
}
header[data-testid="stHeader"] { display: none; }
footer { display: none; }

/* Keep camera container locked at exact fixed height to prevent layout shifts */
[data-testid="stImage"], [data-testid="stImage"] img {
    height: 360px !important;
    max-height: 360px !important;
    width: 100% !important;
    object-fit: cover !important;
    border-radius: 8px !important;
    background: #0f172a !important;
}

/* Big, convenient action buttons */
div.stButton > button {
    height: 46px !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    transition: all 0.12s ease-in-out !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35) !important;
}

.finger-badge {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 8px;
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.finger-letter {
    background: #2563eb;
    color: white;
    font-weight: 900;
    font-size: 1.05rem;
    width: 26px;
    height: 26px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

# ─── 26 Indian Languages List (For Judges & Multi-Lingual Expansion) ──────────
INDIAN_LANGUAGES = [
    "English (Active)",
    "Hindi (हिन्दी)",
    "Bengali (বাংলা)",
    "Telugu (తెలుగు)",
    "Marathi (मराठी)",
    "Tamil (தமிழ்)",
    "Urdu (اردو)",
    "Gujarati (ગુજરાતી)",
    "Kannada (ಕನ್ನಡ)",
    "Malayalam (മലയാളം)",
    "Odia (ଓଡ଼ିଆ)",
    "Punjabi (ਪੰਜਾਬੀ)",
    "Assamese (অসমীয়া)",
    "Maithili (मैथिली)",
    "Santali (संथाली)",
    "Kashmiri (कश्मीरी)",
    "Nepali (नेपाली)",
    "Konkani (कोंकणी)",
    "Sindhi (सिन्धी)",
    "Dogri (डोगरी)",
    "Manipuri (মৈতৈलोन्)",
    "Bodo (बड़ो)",
    "Sanskrit (संस्कृतम्)",
    "Bhojpuri (भोजपुरी)",
    "Marwari (मारवाड़ी)",
    "Chhattisgarhi (छत्तीसगढ़ी)",
]

# ─── Complete 26-Letter A–Z Finger Association Chart ──────────────────────────
ASL_FINGER_GUIDE = {
    "A": ("✊ Fist", "Thumb upright resting alongside index finger"),
    "B": ("✋ Flat Hand", "4 fingers straight up together, thumb tucked across palm"),
    "C": ("⭕ Open Arch", "All fingers curved in a smooth 'C' shaped arch"),
    "D": ("☝️ Index Up", "Index finger straight up, other 3 fingers curled touching thumb"),
    "E": ("✊ Curled Fist", "All 4 fingers curled tightly, thumb folded flat under fingertips"),
    "F": ("👌 OK Sign", "Index & thumb tips touching in circle, middle+ring+pinky straight up"),
    "G": ("👉 Sideways Point", "Index extended horizontally, thumb parallel, other 3 curled"),
    "H": ("👉 2 Sideways", "Index & middle fingers extended horizontally together"),
    "I": ("🤙 Pinky Up", "Pinky straight up, other 4 fingers curled into a fist"),
    "J": ("💫 Dynamic Pinky", "Pinky up tracing a downward hook 'J' curve in the air"),
    "K": ("✌️ Two + Thumb", "Index straight up, middle tilted forward, thumb nestled between them"),
    "L": ("📐 L-Shape", "Thumb and index finger extended at a 90° right angle"),
    "M": ("✊ 3-Tuck", "Closed fist with thumb tucked under 3 fingers (index, middle, ring)"),
    "N": ("✊ 2-Tuck", "Closed fist with thumb tucked under 2 fingers (index, middle)"),
    "O": ("⭕ Closed Loop", "All fingertips touching thumb tip forming a closed circle"),
    "P": ("👇 K Down", "Same handshape as K, pointing downwards towards floor"),
    "Q": ("👇 G Down", "Same handshape as G, pointing downwards towards floor"),
    "R": ("🤞 Crossed", "Index and middle fingers extended and crossed over each other"),
    "S": ("✊ Fist Wrap", "Closed fist with thumb wrapped across the front knuckles"),
    "T": ("✊ 1-Tuck", "Closed fist with thumb tucked under index finger only"),
    "U": ("✌️ Close Pair", "Index and middle fingers straight up held tightly together"),
    "V": ("✌️ Peace Sign", "Index and middle fingers straight up spread apart in a 'V'"),
    "W": ("🖖 3 Fingers Up", "Index, middle, and ring fingers straight up, pinky curled"),
    "X": ("🪝 Finger Hook", "Index finger bent into a hook shape, other fingers curled"),
    "Y": ("🤙 Shaka Sign", "Thumb and pinky extended wide outwards, middle 3 curled"),
    "Z": ("⚡ Dynamic Zigzag", "Index finger tracing a 'Z' zigzag stroke in the air"),
}

# ─── Cached Detector Instance (Instant Rerun — Zero Delay) ────────────────────
@st.cache_resource
def get_hand_detector():
    """Cached singleton of HandDetector to prevent 1.7-second reload freezing on button clicks."""
    return HandDetector(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def create_standby_frame():
    """Generates an initial 480x360 placeholder frame so the camera container never collapses."""
    import numpy as np
    canvas = np.zeros((360, 480, 3), dtype=np.uint8)
    canvas[:] = (15, 23, 42)
    cv2.putText(canvas, "ECHO HANDS", (140, 160), cv2.FONT_HERSHEY_DUPLEX, 0.9, (241, 245, 249), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Ready to Sign", (175, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Click [ Camera] toggle above to start", (105, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# ─── Session State Initialization ─────────────────────────────────────────────
def _init_state():
    if "wb" not in st.session_state:
        st.session_state.wb = WordBuilder(hold_seconds=0.7)
    if "stabilizer" not in st.session_state:
        st.session_state.stabilizer = SignStabilizer(window_size=5, consensus_ratio=0.6)
    if "sm" not in st.session_state:
        st.session_state.sm = SpeechManager(min_stable_seconds=0.8)
    if "preferred_cam" not in st.session_state:
        st.session_state.preferred_cam = 0
    if "speech_rate" not in st.session_state:
        st.session_state.speech_rate = 0.95
    if "selected_lang" not in st.session_state:
        st.session_state.selected_lang = INDIAN_LANGUAGES[0]
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "last_frame" not in st.session_state or st.session_state.last_frame is None:
        st.session_state.last_frame = create_standby_frame()
    if "pending_speak" not in st.session_state:
        st.session_state.pending_speak = None

_init_state()

# ─── Top Bar: Title + Camera Toggle + 26 Languages Selector ───────────────────
top_l, top_r = st.columns([1.5, 1.5])

with top_l:
    st.markdown("<h3 style='margin:0;padding:0;'>🤟 ECHO HANDS <span style='font-size:0.95rem;color:#94a3b8;font-weight:normal;'>| Sign Language Communication</span></h3>", unsafe_allow_html=True)

with top_r:
    h_col1, h_col2, h_col3 = st.columns([1.1, 1.0, 1.8])
    run_camera = h_col1.toggle("▶ Camera", value=False, key="cam_toggle")
    voice_on   = h_col2.checkbox("🔊 Voice", value=st.session_state.sm.enabled, key="voice_chk")
    st.session_state.sm.enabled = voice_on

    selected_lang = h_col3.selectbox(
        "Languages (26 Indian Languages)",
        INDIAN_LANGUAGES,
        index=0,
        help="Select language. English is active; 25 regional languages represent our Phase 2 multi-lingual rollout."
    )
    st.session_state.selected_lang = selected_lang

# ─── Two-Column Viewport: Left (Camera) | Right (Single Text Screen & Keys) ───
col_cam, col_panel = st.columns([1.05, 1.05], gap="medium")

wb:   WordBuilder    = st.session_state.wb
stab: SignStabilizer = st.session_state.stabilizer
sm:   SpeechManager  = st.session_state.sm
rate: float          = st.session_state.speech_rate

# ─── RIGHT COLUMN: ONE Single Unified Screen & 100% Reliable Action Keys ──────
with col_panel:
    # 1. ONE SINGLE UNIFIED DISPLAY SCREEN
    display_screen_placeholder = st.empty()

    # 2. CONVENIENT ACTION KEYS (Native Streamlit Buttons — Guaranteed 100% Working)
    k_col1, k_col2, k_col3 = st.columns([1.4, 1.1, 1.1])
    space_btn = k_col1.button("␣ Space",      key="btn_space", width="stretch", help="Insert space between words")
    back_btn  = k_col2.button("⌫ Back",       key="btn_back",  width="stretch", help="Delete last letter or space")
    undo_btn  = k_col3.button("↩ Undo Space", key="btn_undo",  width="stretch", help="Restore previous word for editing")

    k_col4, k_col5 = st.columns([1.5, 1.0])
    speak_btn = k_col4.button("↵ Speak / Enter", key="btn_speak", type="primary", width="stretch", help="Speak the entire message")
    clear_btn = k_col5.button("🗑 Clear",        key="btn_clear", width="stretch", help="Clear all text")

    # PROCESS BUTTON ACTIONS IMMEDIATELY
    if space_btn:
        completed = wb.word.strip()
        wb.add_space()
        sm.reset()
        if completed and sm.enabled:
            st.session_state.pending_speak = completed

    if back_btn:
        wb.backspace()

    if undo_btn:
        wb.undo_space()
        sm.reset()

    if clear_btn:
        wb.clear()
        stab.reset()
        sm.reset()

    if speak_btn:
        full_text = wb.get_display_text().strip()
        if full_text:
            st.session_state.pending_speak = full_text

    speech_slot = st.empty()

    # 3. COMPLETE A–Z FINGER GUIDE (For non-signers & judges)
    st.markdown("##### 📖 A–Z Sign Language Finger Guide")
    g_tabs = st.tabs(["A – G", "H – N", "O – T", "U – Z"])

    letter_groups = [
        ("A – G", ["A", "B", "C", "D", "E", "F", "G"]),
        ("H – N", ["H", "I", "J", "K", "L", "M", "N"]),
        ("O – T", ["O", "P", "Q", "R", "S", "T"]),
        ("U – Z", ["U", "V", "W", "X", "Y", "Z"]),
    ]

    for tab_idx, (t_name, letters) in enumerate(letter_groups):
        with g_tabs[tab_idx]:
            cols = st.columns(2)
            for i, letter in enumerate(letters):
                c_idx = i % 2
                pose, desc = ASL_FINGER_GUIDE[letter]
                with cols[c_idx]:
                    st.markdown(
                        f"<div class='finger-badge'>"
                        f"<div class='finger-letter'>{letter}</div>"
                        f"<div style='font-size:0.88rem;'><b>{pose}</b><br><span style='color:#94a3b8;'>{desc}</span></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

# ─── Helper: Render Single Unified Display Screen ────────────────────────────
def update_unified_display(active_letter="", hold_pct=0, conf=0.0):
    full_text = wb.get_display_text()

    if full_text:
        display_html = f"<span style='color:#facc15;font-weight:bold;font-size:1.6rem;letter-spacing:1px;'>{full_text}</span>"
    else:
        display_html = "<span style='color:#64748b;font-style:italic;font-size:1.2rem;'>Waiting for hand gestures...</span>"

    # Candidate Letter & Hold Progress Bar
    if active_letter and hold_pct > 0:
        cand_html = (
            f"<div style='margin-top:6px;display:flex;align-items:center;gap:10px;'>"
            f"<span style='color:#38bdf8;font-weight:bold;font-size:1.05rem;'>Signing: <b>{active_letter}</b> ({int(conf*100)}%)</span>"
            f"<div style='flex:1;background:#1e293b;border-radius:4px;height:10px;overflow:hidden;'>"
            f"<div style='background:#22c55e;height:100%;width:{hold_pct}%;transition:width 0.1s;'></div>"
            f"</div>"
            f"<span style='color:#22c55e;font-size:0.85rem;font-weight:bold;'>{hold_pct}%</span>"
            f"</div>"
        )
    else:
        cand_html = "<div style='height:24px;'></div>"

    display_screen_placeholder.markdown(
        f"<div style='background:#0f172a;border:2px solid #334155;border-radius:10px;padding:12px 16px;margin-bottom:8px;'>"
        f"<div style='font-size:0.82rem;color:#94a3b8;font-weight:bold;text-transform:uppercase;letter-spacing:1px;'>💬 Message Screen</div>"
        f"<div style='min-height:48px;display:flex;align-items:center;'>{display_html}<span style='color:#38bdf8;animation:blink 1s infinite;'> |</span></div>"
        f"{cand_html}"
        f"</div>",
        unsafe_allow_html=True
    )

# Render initial display
update_unified_display()

# ─── Play Pending Speech (If Spoken Action was Triggered) ─────────────────────
if st.session_state.pending_speak:
    _txt = st.session_state.pending_speak
    try:
        with speech_slot:
            components.html(generate_browser_speech_html(_txt, voice_lang="en-US", rate=rate), height=0)
    except Exception:
        pass
    st.session_state.pending_speak = None

# ─── LEFT COLUMN: Camera Feed (1/2 Screen — Zero Collapse / Zero Shift) ───────
with col_cam:
    cam_slot = st.empty()
    cam_slot.image(st.session_state.last_frame, channels="RGB", width="stretch")

# ─── Camera Open Helper ───────────────────────────────────────────────────────
def get_or_open_camera():
    if st.session_state.cap is not None and st.session_state.cap.isOpened():
        return st.session_state.cap

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
    idx = st.session_state.preferred_cam

    for try_idx in ([idx] + [i for i in [0, 1, 2] if i != idx]):
        try:
            c = cv2.VideoCapture(try_idx, backend) if backend else cv2.VideoCapture(try_idx)
            if c is not None and c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                c.set(cv2.CAP_PROP_FPS, 30)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                st.session_state.cap = c
                return c
        except Exception:
            continue

    c = cv2.VideoCapture(0)
    st.session_state.cap = c
    return c

# ─── Live Camera Loop ─────────────────────────────────────────────────────────
if run_camera:
    cap = get_or_open_camera()
    detector = get_hand_detector()

    if not cap.isOpened():
        cam_slot.error("⚠️ Camera could not be opened. Check device permissions.")
    else:
        # Quick buffer flush to eliminate stale capture lag
        try:
            cap.grab()
        except Exception:
            pass

        last_ui_update = 0.0
        prev_display_state = None

        while run_camera:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.015)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            hands = detector.find_hands(frame)

            detected_letter = ""
            confidence      = 0.0
            raw_letter      = ""
            reason          = ""
            handedness      = "Right"

            if hands:
                lm = hands[0]
                try:
                    detector.draw_landmarks(frame, lm)
                except Exception:
                    for p1, p2 in HAND_CONNECTIONS:
                        if p1 < len(lm) and p2 < len(lm):
                            cv2.line(frame, (int(lm[p1][1]),int(lm[p1][2])),
                                     (int(lm[p2][1]),int(lm[p2][2])), (0,255,0), 2)
                    for p in lm:
                        cv2.circle(frame, (int(p[1]),int(p[2])), 4, (0,0,255), -1)

                handedness = detector.get_handedness(0)
                raw_letter, raw_conf, reason, _ = recognize_sign(lm, handedness=handedness)
                stable_letter, stable_conf = stab.update(raw_letter, raw_conf)

                detected_letter = stable_letter
                confidence      = stable_conf if stable_letter else raw_conf

                wb.update(detected_letter)
                hold_pct = int(wb.progress * 100)

                # Camera HUD Overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (w,52), (15,23,42), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                if detected_letter:
                    hud_text = f"SIGN: {detected_letter}  ({int(confidence*100)}%)"
                    hud_col  = (60, 240, 80)
                elif raw_letter:
                    hud_text = f"HOLD: {raw_letter} ..."
                    hud_col  = (60, 220, 240)
                else:
                    hud_text = "SIGN: [Transitioning]"
                    hud_col  = (160, 160, 160)

                cv2.putText(frame, hud_text, (14, 36),
                            cv2.FONT_HERSHEY_DUPLEX, 0.82, hud_col, 2, cv2.LINE_AA)

                if hold_pct > 0:
                    bw, bh = 130, 14
                    bx, by = w - bw - 14, 18
                    cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (30,41,59), -1)
                    cv2.rectangle(frame, (bx,by), (bx+int(bw*hold_pct/100), by+bh), (34,197,94), -1)
                    cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (148,163,184), 1)

            else:
                stab.reset(); wb.update("")
                hold_pct = 0
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (w,48), (15,23,42), -1)
                cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
                cv2.putText(frame, "Show your hand to the camera",
                            (14, 32), cv2.FONT_HERSHEY_DUPLEX, 0.68, (100,116,139), 1, cv2.LINE_AA)

            # Display video & remember last frame to prevent any layout collapse
            disp = cv2.resize(frame, (480, 360))
            rgb_disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            st.session_state.last_frame = rgb_disp
            cam_slot.image(rgb_disp, channels="RGB", width="stretch")

            # Efficient UI update: only send DOM diffs when content or progress changes
            now_t = time.time()
            active_char = detected_letter or raw_letter
            current_state = (wb.get_display_text(), active_char, hold_pct // 5)
            if current_state != prev_display_state or (now_t - last_ui_update > 0.25):
                last_ui_update = now_t
                prev_display_state = current_state
                update_unified_display(active_letter=active_char, hold_pct=hold_pct, conf=confidence)

            # Auto-speak building word after stability window
            curr = wb.word.strip()
            if curr and sm.enabled and sm.process_word(curr):
                try:
                    with speech_slot:
                        components.html(generate_browser_speech_html(curr, voice_lang="en-US", rate=rate), height=0)
                except Exception:
                    pass

            time.sleep(0.015)

else:
    # Camera stopped
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None

    # Keep standby frame visible so layout never collapses
    cam_slot.image(st.session_state.last_frame, channels="RGB", width="stretch")

