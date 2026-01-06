import os
import streamlit as st

from src.preprocessing import normalize
from src.model import load_bundle
from src.router import route_ticket
from src.calibration import load_calibration_params


MODEL_PATH = "artifacts/category_baseline.joblib"
CALIB_PATH = "artifacts/calibration.txt"

st.set_page_config(page_title="Support Ticket Router", layout="wide")
st.title("LLM-Ready Support Ticket Router (Baseline Demo)")
st.caption("Ticket → Category + Priority + Queue + Human-in-the-loop routing")

with st.sidebar:
    st.header("Settings")
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found. Run: `python scripts/train_baseline.py`")
    show_topk = st.checkbox("Show top-3 categories", value=True)

    # Calibration toggle
    use_calibration = st.checkbox(
        "Use calibrated confidence (if available)",
        value=True,
        help="Uses temperature scaling + tuned threshold from artifacts/calibration.txt",
    )

    # LLM fallback toggle
    enable_llm = st.checkbox(
        "Enable LLM fallback (if configured)",
        value=False,
        help="Only used when ML says 'needs human review'. If LLM isn't configured, system will silently fall back to ML.",
    )

    # Defaults (will be overwritten if calibration is loaded)
    T = None
    tuned_threshold = None
    target_precision = None

    if use_calibration and os.path.exists(CALIB_PATH):
        params = load_calibration_params(CALIB_PATH)
        T = params.get("T")
        tuned_threshold = params.get("threshold")
        target_precision = params.get("target_precision")

        st.success("Calibration loaded ✅")
        st.write(f"- Temperature (T): **{T:.3f}**")
        st.write(f"- Tuned threshold: **{tuned_threshold:.3f}**")
        if target_precision is not None:
            st.write(f"- Target precision: **{target_precision:.2f}**")
    elif use_calibration and not os.path.exists(CALIB_PATH):
        st.info("Calibration file not found. Run: `python -m scripts.calibrate_and_tune`")
    else:
        st.warning("Calibration disabled (using raw probabilities).")

ticket = st.text_area(
    "Paste a support ticket:",
    height=170,
    placeholder="Example: My ads stopped serving after I updated billing...",
)

col1, col2 = st.columns([1, 1])

run_disabled = (not os.path.exists(MODEL_PATH)) or (not ticket.strip())

if st.button("Route ticket", type="primary", disabled=not os.path.exists(MODEL_PATH) or not ticket.strip()):
    bundle = load_bundle(MODEL_PATH)
    cleaned = normalize(ticket)

    # Route using calibrated confidence if available
    result = route_ticket(
        cleaned, 
        bundle,
        T=T if (use_calibration and T is not None) else None,
        threshold=tuned_threshold if (use_calibration and tuned_threshold is not None) else None,
    )

    with col1:
        st.subheader("Routing decision")
        st.metric("Category", result.category)
        st.metric("Queue", result.queue)
        st.metric("Priority", result.priority)
        st.write("**Priority rationale:**", result.priority_reason)

        # Show LLM metadata if your RouteResult includes it
        if getattr(result, "llm_used", False):
            st.info("LLM fallback used ✅")
            llm_rationale = getattr(result, "llm_rationale", None)
            if llm_rationale:
                st.write("**LLM rationale:**")
                for r in llm_rationale:
                    st.write(f"- {r}")

    with col2:
        st.subheader("Risk & Confidence")

        # Show confidence and threshold used
        if use_calibration and (T is not None) and (tuned_threshold is not None):
            st.write(f"**Calibrated** confidence (T={T:.3f})")
            st.write(f"Threshold: **{tuned_threshold:.3f}**")
        else:
            st.write("**Raw** model confidence")
            st.write("Threshold: **default** (see src/config.py)")

        st.metric("Confidence", f"{result.confidence:.3f}")
        if result.needs_human_review:
            st.error("Send to Human Review ✅ (low confidence / high-risk)")
        else:
            st.success("Auto-route ✅ (meets confidence threshold)")

        if show_topk:
            st.write("**Top-3 categories:**")
            for item in result.top3:
                st.write(f"- {item['label']}: {item['prob']:.2f}")

    st.divider()
    st.write("**Normalized ticket text:**")
    st.code(cleaned)
