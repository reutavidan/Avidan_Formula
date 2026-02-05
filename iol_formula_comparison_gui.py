"""
AVIDAN FORMULA - Pediatric IOL Power Calculator
Two-stage AI approach:
  Stage 1: Predict optimal target refraction based on developmental profile
  Stage 2: Calculate IOL power to achieve that target

Author: Dr. Reut Avidan
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
import json

# Import the XGBoost model
try:
    from xgboost_iol_model import PediatricIOLXGBoost, Config
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# ============================================================
# STAGE 1: AVIDAN TARGET REFRACTION
# ============================================================

def calculate_avidan_target(age_months, syndrome, parental_myopia, siblings_myopic, laterality, axial_length,
                            prematurity="Unknown", rop_history="Unknown", nystagmus="Unknown"):
    """
    AVIDAN TARGET: Personalized target refraction based on developmental risk factors.

    This is NOT the Enyedi nomogram - it's a risk-stratified approach that accounts
    for individual myopia risk factors beyond just age.
    """

    # Base target by age (foundation)
    if age_months < 6:
        base = 8.0
    elif age_months < 12:
        base = 6.0
    elif age_months < 18:
        base = 5.5
    elif age_months < 24:
        base = 5.0
    elif age_months < 36:
        base = 4.0
    elif age_months < 48:
        base = 3.0
    elif age_months < 60:
        base = 2.5
    elif age_months < 84:
        base = 2.0
    elif age_months < 120:
        base = 1.5
    else:
        base = 1.0

    # Risk modifiers
    modifier = 0.0
    risk_factors = []

    # Syndrome-based adjustment (high myopia syndromes need more buffer)
    if syndrome in ["Stickler", "Marfan"]:
        modifier += 2.0  # High myopia risk
        risk_factors.append(f"+2.0 D ({syndrome} - high myopia risk)")
    elif syndrome == "Homocystinuria":
        modifier += 1.5
        risk_factors.append("+1.5 D (Homocystinuria)")
    elif syndrome == "Lowe":
        modifier += 1.0
        risk_factors.append("+1.0 D (Lowe syndrome)")
    elif syndrome == "Down":
        modifier -= 0.5  # Often less myopic shift
        risk_factors.append("-0.5 D (Down syndrome - typically less myopic shift)")
    elif syndrome == "Norrie":
        modifier += 1.0
        risk_factors.append("+1.0 D (Norrie disease)")

    # Family history adjustment
    if parental_myopia == "Both parents":
        modifier += 1.5
        risk_factors.append("+1.5 D (both parents myopic)")
    elif parental_myopia == "One parent":
        modifier += 0.75
        risk_factors.append("+0.75 D (one parent myopic)")

    # Siblings
    if siblings_myopic == "Yes":
        modifier += 0.5
        risk_factors.append("+0.5 D (myopic siblings)")

    # Axial length consideration (longer eyes = higher myopia risk)
    # Age-adjusted AL threshold
    expected_al = 16.0 + (age_months / 12) * 0.4  # Rough growth estimate
    if axial_length > expected_al + 1.5:
        modifier += 0.5
        risk_factors.append("+0.5 D (long eye for age)")
    elif axial_length < expected_al - 1.5:
        modifier -= 0.25
        risk_factors.append("-0.25 D (short eye for age)")

    # Laterality adjustment
    if "Unilateral" in laterality:
        modifier -= 0.5  # Unilateral cases often target closer to fellow eye
        risk_factors.append("-0.5 D (unilateral - targeting closer to fellow eye)")

    # Prematurity adjustment (preterm infants have higher myopia risk)
    if "Very Preterm" in prematurity or "Extremely Preterm" in prematurity:
        modifier += 1.0
        risk_factors.append("+1.0 D (very/extremely preterm - high myopia risk)")
    elif "Moderate Preterm" in prematurity:
        modifier += 0.5
        risk_factors.append("+0.5 D (moderate preterm)")

    # ROP history (treated ROP has higher myopia risk)
    if "treated" in rop_history.lower():
        modifier += 1.0
        risk_factors.append("+1.0 D (ROP treated - high myopia risk)")
    elif "Stage 1-2" in rop_history:
        modifier += 0.25
        risk_factors.append("+0.25 D (mild ROP history)")

    # Nystagmus (may benefit from less hyperopic target for better VA)
    if nystagmus.startswith("Yes"):
        modifier -= 0.5
        risk_factors.append("-0.5 D (nystagmus - targeting closer to emmetropia)")

    # Calculate final target
    target = base + modifier

    # Bounds
    target = max(0.5, min(target, 10.0))

    return round(target, 2), base, risk_factors

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    """Load the Avidan Formula XGBoost model"""
    if not MODEL_AVAILABLE:
        return None

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()

    model_paths = [
        script_dir / 'pediatric_iol/outputs/models/xgboost_pediatric_iol',
        script_dir / 'models/xgboost_pediatric_iol',
        Path('pediatric_iol/outputs/models/xgboost_pediatric_iol'),
        Path('models/xgboost_pediatric_iol'),
    ]

    for path in model_paths:
        if (path / 'xgboost_model.json').exists():
            try:
                model = PediatricIOLXGBoost(Config())
                model.load_model(path)
                return model
            except:
                continue
    return None

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AVIDAN FORMULA",
    page_icon="",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-power {
        font-size: 3.5rem;
        font-weight: bold;
    }
    .result-label {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .target-box {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .target-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .patient-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .risk-factor {
        background: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown('<p class="main-header">AVIDAN FORMULA</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Pediatric IOL Calculator</p>', unsafe_allow_html=True)

# Load model
model = load_model()

if not model:
    st.error("Model not loaded. Please ensure the XGBoost model files are in place.")
    st.stop()

# ============================================================
# PATIENT IDENTIFICATION
# ============================================================

st.markdown("---")

anonymous_mode = st.checkbox("Anonymous Mode (for testing)", value=True)

if not anonymous_mode:
    col_id1, col_id2 = st.columns(2)
    with col_id1:
        patient_name = st.text_input("Patient Name")
    with col_id2:
        patient_id = st.text_input("Patient ID / MRN")
else:
    patient_name = "Anonymous"
    patient_id = f"TEST-{datetime.now().strftime('%H%M%S')}"

# ============================================================
# BIOMETRY INPUT
# ============================================================

st.markdown("---")
st.subheader("Biometry")

col1, col2 = st.columns(2)

with col1:
    age_unit = st.radio("Age unit", ["Months", "Years"], horizontal=True)

    if age_unit == "Months":
        age_months = st.number_input(
            "Age at Surgery (months)",
            min_value=1,
            max_value=180,
            value=24,
            help="Patient age in months"
        )
    else:
        age_years_input = st.number_input(
            "Age at Surgery (years)",
            min_value=0.5,
            max_value=15.0,
            value=2.0,
            step=0.5,
            format="%.1f",
            help="Patient age in years"
        )
        age_months = int(age_years_input * 12)

    sex = st.selectbox("Sex", ["Male", "Female", "Unknown"])

with col2:
    st.write("")  # Spacer
    st.write("")
    st.caption(f"Age: {age_months} months ({age_months/12:.1f} years)")

# ============================================================
# AXIAL LENGTH
# ============================================================
st.markdown("---")
st.markdown("##### Axial Length")

al_col1, al_col2 = st.columns([1, 2])

with al_col1:
    al_modality = st.selectbox(
        "AL Method",
        ["IOLMaster/Lenstar", "A-scan (Contact)", "A-scan (Immersion)",
         "Ultrasound (unspecified)", "MRI-based", "Not Available"],
        key="al_method"
    )

with al_col2:
    if al_modality == "Not Available":
        al_available = False
        axial_length = None
        st.info("AL not available - will use age-based estimate")
    else:
        al_available = True
        axial_length = st.number_input(
            f"Axial Length - {al_modality} (mm)",
            min_value=14.0,
            max_value=28.0,
            value=20.5,
            step=0.01,
            format="%.2f",
            key="al_value"
        )
        # A-scan correction note
        if "Contact" in al_modality:
            st.caption("Contact A-scan may read 0.1-0.3mm shorter than immersion")
        elif "Immersion" in al_modality:
            st.caption("Immersion A-scan is gold standard for pediatric eyes")

# ============================================================
# KERATOMETRY
# ============================================================
st.markdown("---")
st.markdown("##### Keratometry")

k_col1, k_col2 = st.columns([1, 2])

with k_col1:
    k_modality = st.selectbox(
        "K Method",
        ["IOLMaster/Lenstar", "Handheld Autorefractor", "Retinoscopy (estimated)",
         "Manual Keratometer", "Placido Topography", "Not Available"],
        key="k_method"
    )

with k_col2:
    if k_modality == "Not Available":
        k_available = False
        k1 = None
        k2 = None
        st.info("K not available - will use age-based estimate")
    elif k_modality == "Retinoscopy (estimated)":
        k_available = True
        st.caption("Enter estimated K from retinoscopy or use typical values for age")
        retino_col1, retino_col2 = st.columns(2)
        with retino_col1:
            # For retinoscopy, often only get sphere/cylinder, estimate K
            retino_sphere = st.number_input(
                "Retinoscopy Sphere (D)",
                min_value=-20.0,
                max_value=20.0,
                value=2.0,
                step=0.25,
                format="%.2f"
            )
        with retino_col2:
            retino_cyl = st.number_input(
                "Retinoscopy Cylinder (D)",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.25,
                format="%.2f"
            )
        # Estimate K from retinoscopy - this is approximate
        # Using typical pediatric cornea as baseline
        if age_months < 12:
            base_k = 47.0
        elif age_months < 24:
            base_k = 45.0
        elif age_months < 48:
            base_k = 44.0
        else:
            base_k = 43.5
        k1 = base_k - abs(retino_cyl) / 2
        k2 = base_k + abs(retino_cyl) / 2
        st.caption(f"Estimated K: {k1:.2f} / {k2:.2f} D (based on age + cylinder)")
    else:
        k_available = True
        k_input_col1, k_input_col2 = st.columns(2)
        with k_input_col1:
            k1 = st.number_input(
                "K1 - Flat (D)",
                min_value=35.0,
                max_value=55.0,
                value=43.5,
                step=0.25,
                format="%.2f"
            )
        with k_input_col2:
            k2 = st.number_input(
                "K2 - Steep (D)",
                min_value=35.0,
                max_value=55.0,
                value=44.5,
                step=0.25,
                format="%.2f"
            )
        if k_modality == "Handheld Autorefractor":
            st.caption("Handheld devices (Retinomax, PlusOptix) - verify with multiple readings")

# ============================================================
# ACD & OTHER MEASUREMENTS
# ============================================================
st.markdown("---")
st.markdown("##### Other Measurements")

other_col1, other_col2, other_col3 = st.columns(3)

with other_col1:
    acd_modality = st.selectbox(
        "ACD Method",
        ["IOLMaster/Lenstar", "A-scan/Ultrasound", "UBM", "AS-OCT", "Not Available"],
        key="acd_method"
    )
    if acd_modality == "Not Available":
        acd_available = False
        acd = None
    else:
        acd_available = True
        acd = st.number_input(
            "ACD (mm)",
            min_value=1.5,
            max_value=5.0,
            value=3.0,
            step=0.01,
            format="%.2f"
        )

with other_col2:
    lt_modality = st.selectbox(
        "Lens Thickness Method",
        ["IOLMaster/Lenstar", "A-scan/Ultrasound", "UBM", "Not Available"],
        key="lt_method"
    )
    if lt_modality == "Not Available":
        lt_available = False
        lens_thickness = None
    else:
        lt_available = True
        lens_thickness = st.number_input(
            "Lens Thickness (mm)",
            min_value=2.0,
            max_value=6.0,
            value=4.0,
            step=0.1,
            format="%.1f"
        )

with other_col3:
    wtw_modality = st.selectbox(
        "WTW Method",
        ["IOLMaster/Lenstar", "Caliper", "Orbscan/Pentacam", "Estimated", "Not Available"],
        key="wtw_method"
    )
    if wtw_modality == "Not Available":
        wtw_available = False
        wtw = None
    else:
        wtw_available = True
        wtw = st.number_input(
            "WTW (mm)",
            min_value=8.0,
            max_value=14.0,
            value=11.0,
            step=0.1,
            format="%.1f"
        )

# Age-based estimates for missing values (pediatric norms)
def estimate_al_by_age(age_months):
    """Estimate AL based on age - pediatric growth curves"""
    if age_months < 6:
        return 17.0
    elif age_months < 12:
        return 18.0
    elif age_months < 24:
        return 19.5
    elif age_months < 36:
        return 20.5
    elif age_months < 48:
        return 21.0
    elif age_months < 72:
        return 21.5
    elif age_months < 120:
        return 22.0
    else:
        return 23.0

def estimate_k_by_age(age_months):
    """Estimate K based on age - pediatric norms"""
    if age_months < 6:
        return 49.0
    elif age_months < 12:
        return 47.0
    elif age_months < 24:
        return 45.0
    elif age_months < 48:
        return 44.0
    else:
        return 43.5

# Apply estimates if needed
if axial_length is None:
    axial_length = estimate_al_by_age(age_months)
    st.info(f"Using estimated AL: {axial_length} mm (based on age {age_months} months)")

if k1 is None or k2 is None:
    k_est = estimate_k_by_age(age_months)
    k1 = k_est
    k2 = k_est
    st.info(f"Using estimated K: {k_est} D (based on age {age_months} months)")

if acd is None:
    acd = 2.5 if age_months < 24 else 3.0
    st.info(f"Using estimated ACD: {acd} mm")

if lens_thickness is None:
    lens_thickness = 4.0
    st.info(f"Using estimated LT: {lens_thickness} mm")

if wtw is None:
    wtw = 10.0 if age_months < 12 else 11.0
    st.info(f"Using estimated WTW: {wtw} mm")

# K average
k_avg = (k1 + k2) / 2
st.caption(f"K Average: {k_avg:.2f} D")

# ============================================================
# CLINICAL FACTORS
# ============================================================

st.markdown("---")
st.subheader("Clinical Factors")

col3, col4 = st.columns(2)

with col3:
    laterality = st.selectbox(
        "Laterality",
        ["Bilateral", "Unilateral_Right", "Unilateral_Left", "Unknown"]
    )

    etiology = st.selectbox(
        "Cataract Etiology",
        ["Congenital", "Developmental", "Traumatic", "Persistent Fetal Vasculature (PFV)",
         "Rubella", "Metabolic", "Radiation", "Other", "Unknown"]
    )

    morphology = st.selectbox(
        "Cataract Morphology",
        ["Total/Dense", "Nuclear", "Lamellar/Zonular", "Anterior Polar", "Posterior Polar",
         "PSC", "Cortical", "PHPV/PFV", "Membranous", "Other", "Unknown"]
    )

with col4:
    syndrome = st.selectbox(
        "Syndrome",
        ["None", "Stickler", "Marfan", "Down", "Homocystinuria", "Lowe",
         "Norrie", "Hallermann-Streiff", "Rubinstein-Taybi", "Nance-Horan",
         "Galactosemia", "Other", "Unknown"]
    )

    parental_myopia = st.selectbox(
        "Parental Myopia",
        ["None", "One parent", "Both parents", "Unknown"]
    )

    siblings_myopic = st.selectbox(
        "Siblings Myopic",
        ["No", "Yes", "No siblings", "Unknown"]
    )

# Additional pediatric-specific factors
st.markdown("---")
st.subheader("Additional Factors")

add_col1, add_col2 = st.columns(2)

with add_col1:
    prematurity = st.selectbox(
        "Prematurity",
        ["Term (≥37 weeks)", "Late Preterm (34-36 weeks)", "Moderate Preterm (32-33 weeks)",
         "Very Preterm (28-31 weeks)", "Extremely Preterm (<28 weeks)", "Unknown"]
    )

    rop_history = st.selectbox(
        "ROP History",
        ["No ROP", "ROP Stage 1-2 (no treatment)", "ROP treated (laser/injection)",
         "Not applicable", "Unknown"]
    )

with add_col2:
    previous_surgery = st.selectbox(
        "Previous Eye Surgery",
        ["None", "Glaucoma surgery", "Vitrectomy", "Corneal surgery", "Other", "Unknown"]
    )

    nystagmus = st.selectbox(
        "Nystagmus",
        ["No", "Yes - Sensory", "Yes - Congenital", "Yes - Other", "Unknown"]
    )

# Fellow eye info for unilateral cases
if "Unilateral" in laterality:
    st.markdown("---")
    st.subheader("Fellow Eye Information")
    fellow_col1, fellow_col2 = st.columns(2)

    with fellow_col1:
        fellow_refraction_available = st.checkbox("Fellow eye refraction available", value=False)
        if fellow_refraction_available:
            fellow_refraction = st.number_input(
                "Fellow Eye Refraction (SE, D)",
                min_value=-15.0,
                max_value=15.0,
                value=0.0,
                step=0.25,
                format="%.2f"
            )
        else:
            fellow_refraction = None

    with fellow_col2:
        fellow_al_available = st.checkbox("Fellow eye AL available", value=False)
        if fellow_al_available:
            fellow_al = st.number_input(
                "Fellow Eye AL (mm)",
                min_value=14.0,
                max_value=28.0,
                value=22.0,
                step=0.01,
                format="%.2f"
            )
        else:
            fellow_al = None

# ============================================================
# SURGICAL PLANNING (Optional)
# ============================================================

st.markdown("---")
with st.expander("Surgical Planning (Optional)"):
    col5, col6 = st.columns(2)

    with col5:
        iol_model = st.selectbox(
            "IOL Model",
            ["Unknown", "SA60AT", "MA60AC", "SN60WF", "Other"]
        )

    with col6:
        fixation = st.selectbox(
            "Fixation Location",
            ["In-the-bag", "Sulcus", "Unknown"]
        )

# ============================================================
# CALCULATE
# ============================================================

st.markdown("---")

if st.button("Calculate IOL Power", type="primary", use_container_width=True):

    # ========== STAGE 1: Calculate personalized target ==========
    avidan_target, base_target, risk_factors = calculate_avidan_target(
        age_months, syndrome, parental_myopia, siblings_myopic, laterality, axial_length,
        prematurity=prematurity,
        rop_history=rop_history,
        nystagmus=nystagmus
    )

    # ========== STAGE 2: Predict IOL power ==========

    # Map parental myopia
    if parental_myopia == "Both parents":
        father_myopia, mother_myopia, family_score = 1, 1, 8
        parental_category = "Both"
    elif parental_myopia == "One parent":
        father_myopia, mother_myopia, family_score = 1, 0, 4
        parental_category = "One"
    else:
        father_myopia, mother_myopia, family_score = 0, 0, 0
        parental_category = "None"

    # Map siblings
    siblings = 1 if siblings_myopic == "Yes" else 0

    # Prepare input data with the calculated target
    input_data = pd.DataFrame([{
        # Demographics
        'age_months': age_months,
        'sex': sex,
        'race': 'Unknown',
        'laterality': laterality,

        # Biometry
        'axial_length': axial_length,
        'keratometry_k1': k1,
        'keratometry_k2': k2,
        'keratometry_avg': k_avg,
        'anterior_chamber_depth': acd,
        'lens_thickness': lens_thickness,
        'white_to_white': wtw,

        # Clinical
        'syndrome': syndrome,
        'etiology': etiology,
        'cataract_morphology': morphology,

        # Family history
        'father_myopia': father_myopia,
        'mother_myopia': mother_myopia,
        'siblings_myopic': siblings,
        'family_myopia_score': family_score,
        'parental_myopia_category': parental_category,

        # Surgical
        'iol_model': iol_model,
        'fixation_location': fixation,

        # STAGE 1 OUTPUT -> STAGE 2 INPUT
        'target_refraction': avidan_target,

        # ID
        'patient_id': patient_id
    }])

    try:
        # Run prediction
        X, feature_names = model.feature_engineer.prepare_features(input_data, fit=False)
        prediction = model.predict(X)
        iol_power = float(prediction[0])

        # IOL power limits (commercially available pediatric IOLs)
        IOL_MIN = 10.0  # Minimum commonly available
        IOL_MAX = 34.0  # Maximum commonly available (some go to 40D)

        # Validate result
        if iol_power < IOL_MIN:
            st.warning(f"Calculated power ({iol_power:.1f} D) is below minimum available IOL ({IOL_MIN} D).")
            st.info(f"Consider: This eye may not be suitable for primary IOL implantation, or use lowest available power ({IOL_MIN} D).")
            iol_power = IOL_MIN
        elif iol_power > IOL_MAX:
            st.warning(f"Calculated power ({iol_power:.1f} D) exceeds maximum commonly available IOL ({IOL_MAX} D).")
            st.info(f"Consider: Contact IOL manufacturer for high-power options, or defer primary implantation.")
            # Cap at max but show the calculated value
            calculated_power = iol_power
            iol_power = IOL_MAX
        else:
            # Display results
            st.markdown("---")

            # Stage 1 Result: Target
            st.markdown("#### Stage 1: Personalized Target")
            st.markdown(f"""
            <div class="target-box">
                <div class="result-label">AVIDAN TARGET</div>
                <div class="target-value">+{avidan_target:.1f} D</div>
            </div>
            """, unsafe_allow_html=True)

            # Show risk factor breakdown
            with st.expander(f"Target Calculation: Base +{base_target:.1f} D + Risk Adjustments"):
                if risk_factors:
                    for rf in risk_factors:
                        st.markdown(f'<div class="risk-factor">{rf}</div>', unsafe_allow_html=True)
                else:
                    st.write("No additional risk modifiers applied.")

            st.markdown("---")

            # Stage 2 Result: IOL Power
            st.markdown("#### Stage 2: IOL Power")
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">AVIDAN FORMULA</div>
                <div class="result-power">{iol_power:.1f} D</div>
                <div class="result-label">to achieve +{avidan_target:.1f} D target</div>
            </div>
            """, unsafe_allow_html=True)


            # ============================================================
            # SAVE RESULTS
            # ============================================================
            st.markdown("---")
            st.markdown("#### Save Results")

            # Create results record
            results_record = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "patient_id": patient_id,
                "patient_name": patient_name if not anonymous_mode else "Anonymous",
                "age_months": age_months,
                "age_years": round(age_months / 12, 1),
                "sex": sex,
                "laterality": laterality,
                # Biometry
                "axial_length_mm": axial_length,
                "al_modality": al_modality,
                "al_measured": al_available,
                "k1_flat_D": k1,
                "k2_steep_D": k2,
                "k_avg_D": round(k_avg, 2),
                "k_modality": k_modality,
                "k_measured": k_available,
                "k_from_retinoscopy": k_modality == "Retinoscopy (estimated)",
                "acd_mm": acd,
                "acd_modality": acd_modality,
                "acd_measured": acd_available,
                "lens_thickness_mm": lens_thickness,
                "lt_modality": lt_modality,
                "lt_measured": lt_available,
                "wtw_mm": wtw,
                "wtw_modality": wtw_modality if 'wtw_modality' in dir() else "Unknown",
                "wtw_measured": wtw_available,
                # Clinical
                "syndrome": syndrome,
                "etiology": etiology,
                "morphology": morphology,
                "parental_myopia": parental_myopia,
                "siblings_myopic": siblings_myopic,
                # Additional factors
                "prematurity": prematurity,
                "rop_history": rop_history,
                "previous_surgery": previous_surgery,
                "nystagmus": nystagmus,
                # Results
                "avidan_target_D": avidan_target,
                "avidan_iol_D": round(iol_power, 1),
            }

            # CSV format for easy spreadsheet import
            csv_header = ",".join(results_record.keys())
            csv_row = ",".join([str(v) for v in results_record.values()])
            csv_content = f"{csv_header}\n{csv_row}"

            # Download buttons
            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                st.download_button(
                    label="Download as CSV",
                    data=csv_content,
                    file_name=f"avidan_result_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download this calculation as a CSV file"
                )

            with col_dl2:
                json_content = json.dumps(results_record, indent=2)
                st.download_button(
                    label="Download as JSON",
                    data=json_content,
                    file_name=f"avidan_result_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download this calculation as a JSON file"
                )

            # Copy-paste summary
            with st.expander("Copy-Paste Summary"):
                summary_text = f"""AVIDAN FORMULA RESULT
=====================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Patient ID: {patient_id}

BIOMETRY:
- Age: {age_months} months ({age_months/12:.1f} years)
- AL: {axial_length} mm | K: {k_avg:.2f} D | ACD: {acd} mm

CLINICAL:
- Laterality: {laterality}
- Syndrome: {syndrome}
- Parental Myopia: {parental_myopia}

RESULTS:
- AVIDAN Target: +{avidan_target} D
- AVIDAN IOL: {iol_power:.1f} D
"""
                st.code(summary_text, language=None)

            # Patient summary
            if not anonymous_mode:
                st.markdown(f"""
                <div class="patient-info">
                <strong>Patient:</strong> {patient_name} | <strong>ID:</strong> {patient_id}<br>
                <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}
                </div>
                """, unsafe_allow_html=True)

            # Details expander
            with st.expander("View All Calculation Details"):
                st.markdown("**Input Parameters:**")

                details_col1, details_col2 = st.columns(2)

                with details_col1:
                    st.write(f"- Age: {age_months} months ({age_months/12:.1f} years)")
                    st.write(f"- Axial Length: {axial_length:.2f} mm")
                    st.write(f"- K1 (Flat): {k1:.2f} D")
                    st.write(f"- K2 (Steep): {k2:.2f} D")
                    st.write(f"- K Average: {k_avg:.2f} D")
                    st.write(f"- ACD: {acd:.2f} mm")

                with details_col2:
                    st.write(f"- Lens Thickness: {lens_thickness:.1f} mm")
                    st.write(f"- WTW: {wtw:.1f} mm")
                    st.write(f"- Sex: {sex}")
                    st.write(f"- Laterality: {laterality}")
                    st.write(f"- Etiology: {etiology}")
                    st.write(f"- Syndrome: {syndrome}")
                    st.write(f"- Parental Myopia: {parental_myopia}")

                st.markdown("---")
                st.markdown(f"**AVIDAN Target:** +{avidan_target:.1f} D")
                st.markdown(f"**Recommended IOL Power:** {iol_power:.1f} D")

    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        with st.expander("Debug Info"):
            st.write("Input data columns:", list(input_data.columns))
            st.write("Number of input columns:", len(input_data.columns))
            st.dataframe(input_data.T)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("AVIDAN FORMULA | Pediatric IOL Power Calculator | Dr. Reut Avidan")
st.caption("Two-stage approach: Risk-stratified target + ML-based IOL prediction")
st.caption("For research and clinical decision support. Always verify with clinical judgment.")
