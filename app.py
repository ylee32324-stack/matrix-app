import streamlit as st
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title=“행렬 변환 실험실”, page_icon=“🔢”, layout=“wide”)

st.markdown(”””

<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&family=JetBrains+Mono:wght@400;700&display=swap');
  html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
  .stApp { background: #0d1117; color: #e6edf3; }
  .hero {
      background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 60%, #141b2d 100%);
      border: 1px solid #30363d; border-radius: 16px;
      padding: 2.4rem 2.8rem 2rem; margin-bottom: 2rem; position: relative; overflow: hidden;
  }
  .hero::before {
      content: "[[M]]"; font-family: 'JetBrains Mono', monospace;
      font-size: 9rem; color: rgba(88,166,255,0.05);
      position: absolute; right: 1.5rem; top: -1rem; pointer-events: none;
  }
  .hero h1 {
      font-size: 2.2rem; font-weight: 700;
      background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5d6ff);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 .5rem;
  }
  .hero p { color: #8b949e; margin: 0; font-size: .95rem; line-height: 1.7; }
  .formula-row { display: flex; gap: .75rem; flex-wrap: wrap; margin-top: 1.2rem; }
  .formula-badge {
      background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: .45rem .9rem; font-family: 'JetBrains Mono', monospace;
      font-size: .82rem; color: #79c0ff;
  }
  .formula-badge span { color: #8b949e; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1.5rem 1.8rem; margin-bottom: 1.2rem; }
  .card-title { font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: #58a6ff; margin-bottom: 1rem; }
  .matrix-label { font-family: 'JetBrains Mono', monospace; font-size: .85rem; color: #8b949e; margin-bottom: .3rem; }
  .stButton > button {
      width: 100%; background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
      border: none !important; border-radius: 10px !important; color: #fff !important;
      font-family: 'Noto Sans KR', sans-serif !important; font-size: 1rem !important;
      font-weight: 700 !important; padding: .75rem !important;
  }
  .img-label { text-align: center; font-size: .8rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; padding: .4rem 0; border-radius: 6px; margin-bottom: .5rem; }
  .img-label.original { background: #1c2128; color: #8b949e; }
  .img-label.transformed { background: #1f2d1f; color: #56d364; }
  .matrix-display { font-family: 'JetBrains Mono', monospace; font-size: 1rem; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: .8rem 1.2rem; color: #a5d6ff; margin-top: .5rem; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>

“””, unsafe_allow_html=True)

def apply_matrix_transform_pil(image: Image.Image, M: np.ndarray) -> Image.Image:
“”“PIL + NumPy만 사용한 2×2 행렬 선형변환 (역방향 매핑)”””
img = image.convert(“RGBA”)
w, h = img.size
src = np.array(img, dtype=np.uint8)
out = np.zeros_like(src)

```
cx, cy = w / 2.0, h / 2.0

try:
    M_inv = np.linalg.inv(M)
except np.linalg.LinAlgError:
    return image

ys, xs = np.mgrid[0:h, 0:w]
xs_c = xs - cx
ys_c = ys - cy

src_x = M_inv[0, 0] * xs_c + M_inv[0, 1] * ys_c + cx
src_y = M_inv[1, 0] * xs_c + M_inv[1, 1] * ys_c + cy

src_xi = np.round(src_x).astype(int)
src_yi = np.round(src_y).astype(int)
valid = (src_xi >= 0) & (src_xi < w) & (src_yi >= 0) & (src_yi < h)

out[ys[valid], xs[valid]] = src[src_yi[valid], src_xi[valid]]

result = Image.fromarray(out, "RGBA")
bg = Image.new("RGBA", result.size, (20, 20, 30, 255))
bg.paste(result, mask=result.split()[3])
return bg.convert("RGB")
```

if “a” not in st.session_state:
st.session_state.update({“a”: 1.0, “b”: 0.0, “c”: 0.0, “d”: 1.0})

PRESETS = {
“단위행렬”: (1.0, 0.0, 0.0, 1.0),
“좌우반전”: (-1.0, 0.0, 0.0, 1.0),
“상하반전”: (1.0, 0.0, 0.0, -1.0),
“45° 회전”: (0.707, -0.707, 0.707, 0.707),
“90° 회전”: (0.0, -1.0, 1.0, 0.0),
“2배 확대”: (2.0, 0.0, 0.0, 2.0),
“X축 전단”: (1.0, 0.5, 0.0, 1.0),
“Y축 전단”: (1.0, 0.0, 0.5, 1.0),
}

st.markdown(”””

<div class="hero">
  <h1>🔢 행렬 변환 실험실</h1>
  <p>2×2 행렬을 이미지에 직접 적용해 선형변환의 기하학적 의미를 시각적으로 탐구하세요.<br>
     픽셀 좌표 <b>(x, y)</b>에 행렬을 곱하면 새로운 좌표 <b>(x′, y′)</b>가 만들어집니다.</p>
  <div class="formula-row">
    <div class="formula-badge">[ a  b ] · [x] = [ax+by]<br><span>[ c  d ]   [y]   [cx+dy]</span></div>
    <div class="formula-badge">회전: <span>cos θ  -sin θ</span><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span>sin θ   cos θ</span></div>
    <div class="formula-badge">전단: <span>1  k</span><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span>0  1</span></div>
    <div class="formula-badge">반사: <span>-1  0</span><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span> 0  1</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 2.2], gap=“large”)

with left_col:
st.markdown(’<div class="card"><div class="card-title">📁 이미지 업로드</div>’, unsafe_allow_html=True)
uploaded = st.file_uploader(“JPG / PNG / WEBP”, type=[“jpg”, “jpeg”, “png”, “webp”], label_visibility=“collapsed”)
st.markdown(’</div>’, unsafe_allow_html=True)

```
st.markdown('<div class="card"><div class="card-title">⚡ 빠른 프리셋</div>', unsafe_allow_html=True)
preset_cols = st.columns(2)
for i, name in enumerate(PRESETS):
    with preset_cols[i % 2]:
        if st.button(name, key=f"preset_{name}"):
            a2, b2, c2, d2 = PRESETS[name]
            st.session_state.a = a2; st.session_state.b = b2
            st.session_state.c = c2; st.session_state.d = d2
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">🔢 행렬 성분 입력</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#8b949e;font-size:.85rem;margin-bottom:.8rem;">M = [ a  b ] / [ c  d ]</p>', unsafe_allow_html=True)
r1, r2 = st.columns(2)
with r1:
    st.markdown('<div class="matrix-label">a (1행 1열)</div>', unsafe_allow_html=True)
    a = st.number_input("a", value=st.session_state.a, step=0.1, format="%.3f", label_visibility="collapsed", key="input_a")
with r2:
    st.markdown('<div class="matrix-label">b (1행 2열)</div>', unsafe_allow_html=True)
    b = st.number_input("b", value=st.session_state.b, step=0.1, format="%.3f", label_visibility="collapsed", key="input_b")
r3, r4 = st.columns(2)
with r3:
    st.markdown('<div class="matrix-label">c (2행 1열)</div>', unsafe_allow_html=True)
    c = st.number_input("c", value=st.session_state.c, step=0.1, format="%.3f", label_visibility="collapsed", key="input_c")
with r4:
    st.markdown('<div class="matrix-label">d (2행 2열)</div>', unsafe_allow_html=True)
    d = st.number_input("d", value=st.session_state.d, step=0.1, format="%.3f", label_visibility="collapsed", key="input_d")

det = a * d - b * c
det_color = "#56d364" if abs(det) > 1e-6 else "#f85149"
st.markdown(f"""
<div class="matrix-display">
  ⎡ {a:+.3f}  {b:+.3f} ⎤<br>
  ⎣ {c:+.3f}  {d:+.3f} ⎦<br>
  <span style="font-size:.75rem;color:#8b949e;">행렬식(det) = </span>
  <span style="color:{det_color};font-weight:700;">{det:.4f}</span>
</div>
""", unsafe_allow_html=True)

if abs(det) < 1e-6:
    st.warning("⚠️ 행렬식 = 0 : 역행렬이 없어 변환 불가 (차원 붕괴)")
st.markdown('</div>', unsafe_allow_html=True)

transform_clicked = st.button("✨ 변환하기", type="primary")
```

with right_col:
if uploaded is None:
st.markdown(”””
<div style="background:#161b22;border:2px dashed #30363d;border-radius:16px;
padding:5rem 2rem;text-align:center;color:#484f58;margin-top:1rem;">
<div style="font-size:3.5rem;margin-bottom:1rem;">🖼️</div>
<div style="font-size:1.1rem;font-weight:700;color:#6e7681;">이미지를 업로드해 주세요</div>
<div style="font-size:.85rem;margin-top:.5rem;">JPG, PNG, WEBP 형식 지원</div>
</div>
“””, unsafe_allow_html=True)
else:
original_img = Image.open(uploaded)
MAX_SIZE = 500
ow, oh = original_img.size
if max(ow, oh) > MAX_SIZE:
scale = MAX_SIZE / max(ow, oh)
original_img = original_img.resize((int(ow*scale), int(oh*scale)), Image.LANCZOS)

```
    if transform_clicked and abs(det) > 1e-6:
        M = np.array([[a, b], [c, d]], dtype=np.float64)
        with st.spinner("변환 중… 잠시만 기다려 주세요 ⏳"):
            transformed_img = apply_matrix_transform_pil(original_img, M)
        st.session_state["transformed"] = transformed_img
        st.session_state["last_M"] = M.copy()

    img_a, img_b_col = st.columns(2, gap="medium")
    with img_a:
        st.markdown('<div class="img-label original">📷 원본 이미지</div>', unsafe_allow_html=True)
        st.image(original_img, use_container_width=True)
        st.markdown(f'<div style="text-align:center;color:#484f58;font-size:.75rem;">{original_img.size[0]} × {original_img.size[1]} px</div>', unsafe_allow_html=True)

    with img_b_col:
        st.markdown('<div class="img-label transformed">✨ 변환된 이미지</div>', unsafe_allow_html=True)
        if "transformed" in st.session_state:
            st.image(st.session_state["transformed"], use_container_width=True)
            M_d = st.session_state["last_M"]
            st.markdown(f'<div style="text-align:center;color:#484f58;font-size:.75rem;">M=[{M_d[0,0]:.2f},{M_d[0,1]:.2f};{M_d[1,0]:.2f},{M_d[1,1]:.2f}]</div>', unsafe_allow_html=True)
            buf = io.BytesIO()
            st.session_state["transformed"].save(buf, format="PNG")
            st.download_button("⬇️ 변환 이미지 저장", data=buf.getvalue(), file_name="transformed.png", mime="image/png")
        else:
            st.markdown("""
            <div style="background:#0d1117;border:2px dashed #30363d;border-radius:12px;
                        padding:4rem 1rem;text-align:center;color:#484f58;">
              <div style="font-size:2rem;">⬅️</div>
              <div style="font-size:.9rem;margin-top:.5rem;">'변환하기' 버튼을 눌러<br>결과를 확인하세요</div>
            </div>
            """, unsafe_allow_html=True)
```

st.markdown(”—”)
c1, c2, c3 = st.columns(3)
with c1:
st.markdown(”**📐 회전 행렬**\n`\nθ=45° → [0.707,-0.707]\n          [0.707, 0.707]\nθ=90° → [0,-1]\n          [1, 0]\n`”)
with c2:
st.markdown(”**↔️ 반사 행렬**\n`\n좌우 → [-1, 0]\n        [ 0, 1]\n상하 → [1,  0]\n        [0, -1]\n`”)
with c3:
st.markdown(”**📏 전단(Shear)**\n`\nX축 → [1, k]\n       [0, 1]\nY축 → [1, 0]\n       [k, 1]\n`”)