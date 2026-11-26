"""Spec2MRM local Streamlit prototype."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import time

import streamlit as st

import qa
from engine import (ANCHOR_TEMPLATE, EXAMPLE, MAX_ROWS, POLARITIES, TARGET_TEMPLATE, demo_available,
                    generate, library_credits, public_site, supported_columns, targets_from_csv, targets_from_text)
import fiora_predict  # importable after engine (engine adds the toolkit root to sys.path)

HERE = Path(__file__).resolve().parent
LOGO_LIGHT = HERE / 'assets' / 'spec2mrm_logo.jpg'  # original logo, white background
CONTACT_EMAIL = 'mh.nafis29@gmail.com'
MOLECULES = HERE / 'assets' / 'molecules.json'     # made by build_molecules.py
PASTE_PLACEHOLDER = ('Caffeine, Cn1c(=O)c2c(ncn2C)n(C)c1=O\n'
                     'PFOA\tOC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F\n'
                     'Acetaminophen, CC(=O)Nc1ccc(O)cc1, Positive')
RT_CREDIT = (
    '**Retention-time model: 2-step**, developed by F. Kretschmer, E.-M. Harrieder, M. Witting and S. Böcker. '
    '*Times are changing but order matters: transferable prediction of small-molecule liquid chromatography '
    'retention times.* Nature Methods (2026). '
    '[doi:10.1038/s41592-026-03243-2](https://doi.org/10.1038/s41592-026-03243-2) · '
    'Code and pretrained weights: [boecker-lab/2-step](https://github.com/boecker-lab/2-step) (AGPL-3.0) · '
    'Column database: [RepoRT](https://github.com/michaelwitting/RepoRT) (CC BY-SA 4.0). '
    'Spec2MRM uses their published model unchanged.')

st.set_page_config(page_title='Spec2MRM | Spectra to MRM workflow', page_icon='⚗️', layout='wide')


def secret(name):
    """Value from the host's secret settings (Streamlit secrets), else the environment."""
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file on this host
        value = None
    return str(value or os.environ.get(name, '')).strip()


# The Q&A database URL lives in the secrets; qa.py reads it from the environment.
if secret('SPEC2MRM_QA_URL'):
    os.environ['SPEC2MRM_QA_URL'] = secret('SPEC2MRM_QA_URL')
PUBLIC = public_site()
ADMIN_HASH = secret('SPEC2MRM_ADMIN_SHA256').lower()   # SHA-256 of the developer's admin password

st.markdown("""
<style>
html { background: #000000 !important; }
body { background: transparent !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {
    background: transparent !important; }
[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: none; }
.stApp [data-testid="stSidebar"] p, .stApp [data-testid="stSidebar"] li, .stApp [data-testid="stSidebar"] span,
.stApp [data-testid="stSidebar"] label, .stApp [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"],
.stApp [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #000000 !important; }
.stApp [data-testid="stSidebar"] hr { border-color: #DADFDD !important; background: #DADFDD !important; }
.stApp [data-testid="stSidebar"] button { background: #F3F6F4 !important; color: #000000 !important;
    border: 1px solid #CBD5D0 !important; }
.stApp [data-testid="stSidebar"] button * { color: #000000 !important; }
.stApp [data-testid="stSidebar"] button[aria-checked="true"],
.stApp [data-testid="stSidebar"] button[aria-checked="true"] * {
    background: #4E9F3D !important; color: #FFFFFF !important; border-color: #4E9F3D !important; }
.stApp [data-testid="stSidebarCollapseButton"] button { background: transparent !important; border: none !important; }
[data-testid="stSidebarHeader"] { height: 0.5rem !important; min-height: 0 !important; padding-top: 0.4rem !important; }
[data-testid="stSidebar"] [data-testid="stImage"] img { border-radius: 10px; }
[data-testid="stSidebarUserContent"] { padding-top: 0.6rem !important; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(0, 0, 0, 0.55); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p, [data-testid="stWidgetLabel"] p,
[data-testid="stExpander"] summary p { color: #FFFFFF !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: 1rem !important; opacity: 1 !important; }
.stApp textarea::placeholder, .stApp input::placeholder { color: #A8B0B8 !important; opacity: 1; }
[data-testid="stWidgetLabel"] p { font-size: 1.05rem !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: #8FD47F !important; }
.stApp h1, .stApp h2, .stApp h3 { color: #FFFFFF !important; }
.stApp a { color: #8FD47F !important; }
.stApp textarea, .stApp input { color: #FFFFFF !important; }
.s2m-tagline { font-size: 1.3rem; margin: 0.2rem 0 0.4rem 0; }
.s2m-credit, .s2m-credit p { font-size: 0.85rem !important; line-height: 1.45; color: #FFFFFF !important; }
.s2m-credit { border-top: 1px solid #2F343A; margin-top: 2.5rem; padding-top: 0.8rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def molecule_data():
    return MOLECULES.read_text(encoding='utf-8') if MOLECULES.exists() else '[]'


def molecule_background():
    """Rotating 3D structures (PubChem conformers) spread evenly over the page.

    Each molecule has its own spot on a jittered grid covering the whole
    window and dances slowly around it while rotating. About 30% of them are
    "followers": when the cursor comes near they swarm around it, and return
    to their spot when it leaves. The script runs in the main page once per
    tab, so Streamlit reruns do not restart it. Purely decorative; it ignores
    clicks and sends nothing anywhere."""
    st.iframe("""
<script>
(function () {
  const doc = window.parent.document, win = window.parent;
  win.__s2mMolecules = __DATA__;
  if (win.__s2mBg3dV5) return;
  win.__s2mBg3dV5 = true;
  win.__s2mBg3dV2 = win.__s2mBg3dV3 = win.__s2mBg3dV4 = true;
  for (const id of ['s2m-spectra-bg', 's2m-molecules-bg', 's2m-molecule-label']) { const old = doc.getElementById(id); if (old) old.remove(); }
  win.__s2mBg3d = true;  // stop an older version of this script from starting
  const code = `(function () {
    const mols = (window.__s2mMolecules || []).map(m => {
      const n = m.atoms.length; let cx = 0, cy = 0, cz = 0;
      for (const a of m.atoms) { cx += a[1]; cy += a[2]; cz += a[3]; }
      cx /= n; cy /= n; cz /= n;
      const atoms = m.atoms.map(a => ({el: a[0], x: a[1] - cx, y: a[2] - cy, z: a[3] - cz}));
      const radius = Math.max(...atoms.map(a => Math.hypot(a.x, a.y, a.z))) + 0.6;
      return {name: m.name, cls: m.cls, atoms, bonds: m.bonds, radius};
    });
    if (!mols.length) return;
    const canvas = document.createElement('canvas');
    canvas.id = 's2m-molecules-bg';
    canvas.setAttribute('aria-hidden', 'true');
    Object.assign(canvas.style, {position: 'fixed', inset: '0', width: '100vw', height: '100vh',
                                 zIndex: '-1', pointerEvents: 'none'});
    document.body.prepend(canvas);
    const ctx = canvas.getContext('2d');
    const STYLE = {C: ['#C3C8CF', 0.38], H: ['#FFFFFF', 0.24], O: ['#FF4B4B', 0.37], N: ['#4F86FF', 0.38],
                   F: ['#8CE356', 0.35], Cl: ['#2FD15A', 0.46], S: ['#FFD43B', 0.48], P: ['#FF9A2E', 0.48],
                   Br: ['#C8503C', 0.52], I: ['#9B4BD6', 0.56], Na: ['#AB5CF2', 0.5]};
    const style = el => STYLE[el] || ['#E08AF0', 0.42];
    const sprites = {};
    function sprite(el) {
      if (sprites[el]) return sprites[el];
      const c = document.createElement('canvas'); c.width = c.height = 64;
      const g = c.getContext('2d'), col = style(el)[0];
      const grad = g.createRadialGradient(23, 21, 3, 32, 32, 31);
      grad.addColorStop(0, '#FFFFFF'); grad.addColorStop(0.28, col); grad.addColorStop(1, '#0A0A0A');
      g.fillStyle = grad; g.beginPath(); g.arc(32, 32, 31, 0, Math.PI * 2); g.fill();
      return (sprites[el] = c);
    }
    const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const FOLLOW_SHARE = 0.3, FOLLOW_RANGE = 520;
    let W = 0, H = 0, items = [], deck = [], cell = 200;
    const mouse = {x: -1e4, y: -1e4, vx: 0, vy: 0, active: false};
    function nextMol() {
      if (!deck.length) deck = mols.slice().sort(() => Math.random() - 0.5);
      return deck.pop();
    }
    function make(i) {
      const mol = nextMol();
      return {mol, follower: false, x: 0, y: 0, hx: 0, hy: 0, glow: 0, size: 0.8 + Math.random() * 0.4,
              f1: 0.00009 + Math.random() * 0.00012, f2: 0.00007 + Math.random() * 0.00012,
              p1: Math.random() * 6.28, p2: Math.random() * 6.28, orbit: Math.random() * 6.28,
              ring: 70 + Math.random() * 110,
              ax: Math.random() * 6.28, ay: Math.random() * 6.28, az: Math.random() * 6.28,
              wx: (Math.random() - 0.5) * 0.005, wy: (Math.random() < 0.5 ? -1 : 1) * (0.002 + Math.random() * 0.004),
              wz: (Math.random() - 0.5) * 0.002};
    }
    function layout() {
      // Jittered grid over the whole window: no large empty areas.
      const count = items.length, cols = Math.max(1, Math.round(Math.sqrt(count * W / H)));
      const rows = Math.ceil(count / cols), cw = W / cols, ch = H / rows;
      cell = Math.min(cw, ch);
      const order = items.map((_, i) => i).sort(() => Math.random() - 0.5);
      order.forEach((idx, k) => {
        const s = items[idx], c = k % cols, r = Math.floor(k / cols);
        s.hx = (c + 0.5 + (Math.random() - 0.5) * 0.5) * cw;
        s.hy = (r + 0.5 + (Math.random() - 0.5) * 0.5) * ch;
        if (!s.x && !s.y) { s.x = s.hx; s.y = s.hy; }
        s.scale = Math.min(20, cell * 0.42 * s.size / s.mol.radius);
      });
      const shuffled = items.slice().sort(() => Math.random() - 0.5);
      shuffled.forEach((s, k) => { s.follower = k < Math.round(items.length * FOLLOW_SHARE); });
    }
    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      W = window.innerWidth; H = window.innerHeight;
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.max(12, Math.min(40, Math.round(W * H / 52000)));
      while (items.length < count) items.push(make(items.length));
      items.length = count;
      layout();
    }
    function rotate(a, s) {
      const cx = Math.cos(s.ax), sx = Math.sin(s.ax), cy = Math.cos(s.ay), sy = Math.sin(s.ay),
            cz = Math.cos(s.az), sz = Math.sin(s.az);
      let x = a.x, y = a.y * cx - a.z * sx, z = a.y * sx + a.z * cx;
      let x2 = x * cy + z * sy; z = -x * sy + z * cy; x = x2;
      x2 = x * cz - y * sz; y = x * sz + y * cz; x = x2;
      return {x, y, z};
    }
    function drawMolecule(s) {
      const R = s.mol.radius;
      const pts = s.mol.atoms.map(a => {
        const r = rotate(a, s), f = 9 / (9 - r.z / R * 2.2);
        return {x: s.x + r.x * s.scale * f, y: s.y + r.y * s.scale * f, z: r.z, f, el: a.el};
      });
      const base = 0.32 + 0.6 * s.glow;
      const order = [];
      s.mol.bonds.forEach(b => order.push({z: (pts[b[0]].z + pts[b[1]].z) / 2 - 0.01, bond: b}));
      pts.forEach((p, i) => order.push({z: p.z, atom: i}));
      order.sort((p, q) => p.z - q.z);
      ctx.lineCap = 'round';
      for (const item of order) {
        const depth = 0.55 + 0.45 * (item.z / R + 1) / 2;
        ctx.globalAlpha = Math.max(0, Math.min(1, base * depth));
        if (item.bond) {
          const p = pts[item.bond[0]], q = pts[item.bond[1]], mx = (p.x + q.x) / 2, my = (p.y + q.y) / 2;
          const w = Math.max(1, s.scale * 0.13), n = item.bond[2];
          const dx = q.x - p.x, dy = q.y - p.y, len = Math.hypot(dx, dy) || 1;
          const ox = -dy / len * w * 1.1, oy = dx / len * w * 1.1;
          const offsets = n === 2 ? [-0.5, 0.5] : n === 3 ? [-1, 0, 1] : [0];
          ctx.lineWidth = n > 1 ? w * 0.7 : w;
          for (const k of offsets) {
            ctx.strokeStyle = style(p.el)[0];
            ctx.beginPath(); ctx.moveTo(p.x + ox * k, p.y + oy * k); ctx.lineTo(mx + ox * k, my + oy * k); ctx.stroke();
            ctx.strokeStyle = style(q.el)[0];
            ctx.beginPath(); ctx.moveTo(mx + ox * k, my + oy * k); ctx.lineTo(q.x + ox * k, q.y + oy * k); ctx.stroke();
          }
        } else {
          const p = pts[item.atom], r = style(p.el)[1] * s.scale * p.f;
          ctx.drawImage(sprite(p.el), p.x - r, p.y - r, r * 2, r * 2);
        }
      }
      if (s.glow > 0.35) {
        ctx.globalAlpha = Math.min(1, (s.glow - 0.35) * 2.2);
        ctx.fillStyle = '#FFFFFF'; ctx.textAlign = 'center';
        ctx.font = '600 14px "Source Sans Pro", sans-serif';
        const bottom = s.y + R * s.scale + 16;
        ctx.fillText(s.mol.name, s.x, bottom);
        ctx.font = '12px "Source Sans Pro", sans-serif'; ctx.fillStyle = '#9FD8A0';
        ctx.fillText(s.mol.cls, s.x, bottom + 15);
      }
    }
    function step(t) {
      for (const s of items) {
        // Slow dance around the molecule's own spot.
        const amp = cell * 0.55;
        let tx = s.hx + amp * Math.sin(t * s.f1 + s.p1) + amp * 0.35 * Math.sin(t * s.f2 * 2.3 + s.p2);
        let ty = s.hy + amp * Math.cos(t * s.f2 + s.p2) + amp * 0.35 * Math.cos(t * s.f1 * 1.7 + s.p1);
        const d = Math.hypot(mouse.x - s.x, mouse.y - s.y);
        const chasing = s.follower && mouse.active && Math.hypot(mouse.x - s.hx, mouse.y - s.hy) < FOLLOW_RANGE;
        if (chasing) {
          s.orbit += 0.004;
          tx = mouse.x + Math.cos(s.orbit) * s.ring;
          ty = mouse.y + Math.sin(s.orbit) * s.ring * 0.8;
        }
        const ease = chasing ? 0.035 : 0.012;
        s.x += (tx - s.x) * ease; s.y += (ty - s.y) * ease;
        const close = mouse.active && d < 200;
        if (close) { s.wy += mouse.vx * 0.00003 * (1 - d / 200); s.wx -= mouse.vy * 0.00003 * (1 - d / 200); }
        s.glow += ((close ? 1 - d / 200 : chasing ? 0.25 : 0) - s.glow) * 0.06;
        s.wx *= 0.997; s.wy *= 0.997; s.wz *= 0.997;
        if (Math.abs(s.wy) < 0.002) s.wy += 0.0003 * Math.sign(s.wy || 1);
        s.ax += s.wx; s.ay += s.wy; s.az += s.wz;
      }
      mouse.vx *= 0.8; mouse.vy *= 0.8;
    }
    function frame(t) {
      if (!still) step(t || 0);
      ctx.clearRect(0, 0, W, H);
      for (const s of items) drawMolecule(s);
      ctx.globalAlpha = 1;
      if (!still) requestAnimationFrame(frame);
    }
    let resizeTimer = null;
    window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(resize, 150); });
    window.addEventListener('pointermove', e => {
      if (mouse.active) { mouse.vx = e.clientX - mouse.x; mouse.vy = e.clientY - mouse.y; }
      mouse.x = e.clientX; mouse.y = e.clientY; mouse.active = true;
    }, {passive: true});
    document.addEventListener('pointerleave', () => { mouse.active = false; });
    resize(); requestAnimationFrame(frame);
  })();`;
  const script = doc.createElement('script');
  script.textContent = code;
  doc.body.appendChild(script);
})();
</script>
""".replace('__DATA__', molecule_data()), height=1)


def page_header():
    st.image(str(LOGO_LIGHT), width=300)


def escape_md(text):
    """User text shown literally: no links, images, HTML or formatting."""
    return re.sub(r'([\\`*_{}\[\]()#+\-.!|<>~$])', r'\\\1', text).replace('\n', '  \n')


def rate_ok():
    now = time.time()
    recent = [t for t in st.session_state.get('qa_posts', []) if now - t < 3600]
    st.session_state['qa_posts'] = recent
    if len(recent) >= 10:
        raise ValueError('You have posted 10 times in the last hour. Please try again later.')
    recent.append(now)


def admin_sign_in():
    """Public site: the developer signs in with the admin password to answer and delete."""
    if st.session_state.get('_admin'):
        if st.button('Sign out of developer mode'):
            st.session_state['_admin'] = False
            st.rerun()
        return
    if not ADMIN_HASH:
        return
    with st.expander('Developer sign-in'):
        tries = st.session_state.get('_admin_tries', 0)
        if tries >= 5:
            st.error('Too many attempts. Reload the page to try again later.')
            return
        with st.form('admin_login', clear_on_submit=True):
            password = st.text_input('Admin password', type='password', max_chars=200)
            if st.form_submit_button('Sign in'):
                if hmac.compare_digest(hashlib.sha256(password.encode('utf-8')).hexdigest(), ADMIN_HASH):
                    st.session_state['_admin'] = True
                    st.rerun()
                st.session_state['_admin_tries'] = tries + 1
                time.sleep(1)
                st.error('That password is not correct.')


def render_qa(developer):
    st.header('Questions & answers')
    if PUBLIC and not qa.online():
        st.warning('The Q&A board is not connected to its online database yet, so posts may disappear when the '
                   f'server restarts. Email [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL}) for anything important.')
    st.markdown('Ask anything about Spec2MRM. The developer answers here, and other users can answer too. '
                'Posts are visible to everyone using this app, so please do not include confidential data.')
    if PUBLIC:
        admin_sign_in()
    if developer:
        st.info('You are on the developer copy of the app. Your answers are marked “Developer answer”, '
                'and you can delete posts.')
    with st.form('qa_ask', clear_on_submit=True):
        name = st.text_input('Your name (optional)', max_chars=qa.MAX_NAME)
        body = st.text_area('Your question', max_chars=qa.MAX_QUESTION, height=120)
        if st.form_submit_button('Post question', type='primary'):
            try:
                rate_ok()
                qa.ask(name, body)
                st.success('Your question is posted.')
            except ValueError as error:
                st.error(str(error))
    search = st.text_input('Search questions and answers', key='qa_search')
    items = qa.board(search)
    if not items:
        st.info('No questions match.' if search.strip() else 'No questions yet. Be the first to ask.')
    for q in items:
        with st.container(border=True):
            st.markdown(f'**Q.** {escape_md(q["body"])}')
            st.caption(f'Asked by {escape_md(q["author"])} · {q["created"]}')
            for a in q['answers']:
                st.divider()
                if a['developer']:
                    st.badge('Developer answer', color='green')
                st.markdown(escape_md(a['body']))
                st.caption(f'Answered by {escape_md(a["author"])} · {a["created"]}')
                if developer and st.button('Delete this answer', key=f'qa_del_a_{a["id"]}'):
                    qa.delete_answer(a['id'])
                    st.rerun()
            label = f'Write an answer ({len(q["answers"])} so far)'
            with st.expander(label):
                with st.form(f'qa_answer_{q["id"]}', clear_on_submit=True):
                    who = st.text_input('Your name (optional)', max_chars=qa.MAX_NAME, key=f'qa_name_{q["id"]}',
                                        placeholder='Developer' if developer else 'Anonymous')
                    text = st.text_area('Your answer', max_chars=qa.MAX_ANSWER, key=f'qa_text_{q["id"]}')
                    if st.form_submit_button('Post answer'):
                        try:
                            rate_ok()
                            qa.answer(q['id'], who, text, developer=developer)
                            st.rerun()
                        except ValueError as error:
                            st.error(str(error))
            if developer and st.button('Delete this question', key=f'qa_del_q_{q["id"]}'):
                qa.delete_question(q['id'])
                st.rerun()


def require_trial_access():
    if PUBLIC:
        return False  # the public site is open to everyone
    expected = os.environ.get('SPEC2MRM_TRIAL_CODE_HASH')
    network_trial = bool(expected) or st.get_option('server.address') not in ('127.0.0.1', '::1', 'localhost')
    if not network_trial:
        return False
    if not expected:
        st.error('Network access is locked. Start this instance with the department trial launcher.')
        st.stop()
    if not st.session_state.get('_department_trial_access'):
        molecule_background()
        page_header()
        st.badge('DEPARTMENT TRIAL', color='green')
        st.markdown('Enter the access code from your department host.')
        with st.form('department_trial_login', clear_on_submit=True):
            code = st.text_input('Access code', type='password', max_chars=128)
            unlocked = st.form_submit_button('Open Spec2MRM', type='primary')
        if unlocked:
            if hmac.compare_digest(hashlib.sha256(code.encode('utf-8')).hexdigest(), expected):
                st.session_state['_department_trial_access'] = True
                st.rerun()
            st.error('The access code did not match.')
        st.stop()
    return True


department_trial = require_trial_access()


@st.cache_data(show_spinner=False)
def columns_for_rt():
    return supported_columns()


def download(label, data, name):
    st.download_button(label, data, file_name=name, mime='text/csv', on_click='ignore')


def profile_form(pol, columns):
    with st.expander(f'{pol} LC method', expanded=True):
        left, right = st.columns(2)
        with left:
            label = st.text_input('Method name', key=f'{pol}_name', max_chars=80,
                                  placeholder=f'My {pol.lower()} reversed-phase method')
            name = st.selectbox('Reversed-phase column', columns, index=None,
                                placeholder='Select a supported column', key=f'{pol}_column')
            ph = st.number_input('Aqueous mobile-phase pH', min_value=0.0, max_value=14.0,
                                 value=None, step=0.05, key=f'{pol}_ph')
            run = st.number_input('Run length (min)', min_value=0.1, max_value=600.0,
                                  value=None, key=f'{pol}_run')
        with right:
            length = st.number_input('Column length (mm)', min_value=1.0, max_value=1000.0, value=None, key=f'{pol}_length')
            diameter = st.number_input('Column internal diameter (mm)', min_value=0.1, max_value=100.0,
                                       value=None, step=0.1, key=f'{pol}_id')
            particle = st.number_input('Particle size (µm)', min_value=0.1, max_value=50.0,
                                       value=None, step=0.1, key=f'{pol}_particle')
            flow = st.number_input('Flow (mL/min)', min_value=0.001, max_value=20.0,
                                   value=None, step=0.05, format='%.3f', key=f'{pol}_flow')
        temperature, dead = st.columns(2)
        temp = temperature.number_input('Column temperature (°C)', min_value=0.0, max_value=100.0,
                                        value=None, key=f'{pol}_temp')
        t0 = dead.number_input('Measured dead time (min, optional)', min_value=0.001, max_value=100.0,
                               value=None, format='%.3f', key=f'{pol}_t0')
        st.caption('If dead time is blank, Spec2MRM estimates it from the column dimensions and flow. '
                   'Anchors at or below twice the dead time are excluded.')
        notes = st.text_area('Solvents and gradient (for your records)', key=f'{pol}_notes', max_chars=1000,
                             placeholder='Mobile phases, gradient steps, additives…')
        anchor = st.file_uploader(f'{pol} measured anchors', type=['csv'], key=f'{pol}_anchor', max_upload_size=1)
        st.caption('CSV columns: smiles, rt_min. Use measured RTs from this LC setup. '
                   'At least three distinct retained anchors are required; include standards spanning your run.')
    method = dict(method_id=label.strip(), column_name=name, ph=ph, run_time_min=run,
                  column_length_mm=length, column_id_mm=diameter, particle_size_um=particle,
                  flow_ml_min=flow, column_temperature_c=temp, dead_time_min=t0, void_factor=2.0,
                  ph_source='User-provided aqueous mobile-phase pH', mobile_phase_and_gradient=notes.strip())
    ready = bool(label.strip() and anchor is not None and
                 all(x is not None for x in (name, ph, run, length, diameter, particle, flow, temp)))
    return (method, anchor.getvalue()) if ready else None


def render_results(result):
    transitions, misses, rt = (result[key] for key in ('transitions', 'misses', 'rt'))
    st.header('3 · Review and download')
    a, b, c, d = st.columns(4)
    a.metric('Compounds submitted', result['target_count'])
    b.metric('Transitions generated', len(transitions))
    c.metric('Compounds without transitions', len(misses))
    d.metric('RT estimates', int(rt.predicted_rt_min.notna().sum()) if not rt.empty else 0)
    if result['rt_error']:
        st.warning(f"RT prediction could not finish: {result['rt_error']} Any generated MRM transitions are still available.")
    if result.get('fiora_error'):
        st.info(f"In-silico prediction was not run: {result['fiora_error']} Library and rule transitions are unaffected.")
    tabs = st.tabs(['Transitions', 'RT estimates', 'Unmatched compounds'])
    with tabs[0]:
        if transitions.empty:
            st.info('No transitions were generated. Review the unmatched compounds tab.')
        else:
            source = transitions.fragment_source.fillna('')
            rule = source.str.startswith('class rule')
            fiora = source.str.startswith('in-silico')
            n_lib = transitions.loc[~rule & ~fiora, ['inchikey', 'polarity']].drop_duplicates().shape[0]
            n_rule = transitions.loc[rule, ['inchikey', 'polarity']].drop_duplicates().shape[0]
            n_fiora = transitions.loc[fiora, ['inchikey', 'polarity']].drop_duplicates().shape[0]
            st.markdown(f'**{n_lib}** compound/mode pairs from experimental library spectra · '
                        f'**{n_rule}** from PFAS/QAC class rules · **{n_fiora}** predicted in silico (FIORA)')
            if n_fiora:
                st.warning('FIORA-predicted transitions are low-confidence starting points. In our benchmark on 187 '
                           "compounds measured on the lab's Agilent 6470, FIORA's top predicted ion was one of the "
                           f'monitored product ions for {fiora_predict.TOP1_HIT_ALL:.0%} of compounds '
                           f'({fiora_predict.TOP1_HIT_UNSEEN:.0%} for compounds absent from public libraries). '
                           'Confirm them with a standard or by product-ion scan before quantitative use.')
            if n_rule:
                st.caption('Rule-predicted transitions use well-established PFAS and QAC fragmentation. On the '
                           "lab's Agilent 6470 they reproduced 55 of 58 measured quantifier ions and 43 of 46 "
                           'qualifiers (a consistency check on the data the rules were built from, not an '
                           'independent validation). Confirm them with a standard when one is available.')
            st.caption('Model-predicted CE values are Agilent starting settings. Verify them on your instrument.')
            transitions = transitions.assign(source=source.str.replace('experimental library consensus', 'Library')
                                             .str.replace('class rule: ', 'Rule: ')
                                             .str.replace('in-silico: FIORA', 'Predicted: FIORA'))
            cols = ['name', 'polarity', 'role', 'precursor_ion', 'precursor_mz', 'product_mz', 'collision_energy_eV',
                    'source', 'fragment_confidence', 'library_spectra', 'predicted_rt_min', 'rt_status']
            st.dataframe(transitions[[c for c in cols if c in transitions]], hide_index=True, width='stretch',
                         column_config={'name': 'Compound', 'precursor_ion': 'Precursor ion',
                                        'precursor_mz': st.column_config.NumberColumn(
                                            'Q1 m/z', format='%.4f',
                                            help='Calculated from the structure for the precursor ion shown.'),
                                        'product_mz': st.column_config.NumberColumn(
                                            'Q3 m/z', format='%.4f',
                                            help='Consensus product ion. Ions closer than 0.7 Da are merged, '
                                                 'since Q3 cannot separate them.'),
                                        'library_spectra': st.column_config.NumberColumn(
                                            'Library spectra', help='Independent spectra after removing '
                                                                    'duplicates mirrored between libraries.'),
                                        'collision_energy_eV': 'Starting CE', 'source': 'Source',
                                        'fragment_confidence': 'Confidence',
                                        'predicted_rt_min': 'Predicted RT (min)'})
            left, right = st.columns(2)
            with left:
                download('Download unscheduled MassHunter CSV', result['masshunter'], 'spec2mrm_masshunter_unscheduled.csv')
            with right:
                download('Download full results CSV', transitions.to_csv(index=False), 'spec2mrm_transitions.csv')
    with tabs[1]:
        if rt.empty:
            st.info('Select an LC profile with measured anchors to add retention times in minutes.')
        else:
            st.caption('2-step RT applies to the selected LC profile. Estimates are not scheduled acquisition windows. '
                       'Values outside the run are flagged and omitted from the RT column.')
            for job in result['manifest']['jobs']:
                st.caption(f"{job['polarity']}: {job['n_anchors_retained']} of {job['n_anchors_total']} anchors retained; "
                           f"dead time {job['dead_time_min']:.3f} min ({job['dead_time_source']}).")
            st.dataframe(rt[['name', 'polarity', 'predicted_rt_min', 'rt_roi', 'rt_method_id', 'rt_status']],
                         hide_index=True, width='stretch')
            download('Download RT estimates CSV', rt.to_csv(index=False), 'spec2mrm_rt_predictions.csv')
            st.download_button('Download RT run details', json.dumps(result['manifest'], indent=2),
                               file_name='spec2mrm_rt_run.json', mime='application/json', on_click='ignore')
    with tabs[2]:
        if misses.empty:
            st.success('Every submitted compound has at least one transition.')
        else:
            st.caption('These compounds have no library spectrum in the requested mode, no PFAS/QAC class rule, and '
                       'no usable in-silico prediction (see the reason column). Use this list for follow-up.')
            st.dataframe(misses, hide_index=True, width='stretch')
            download('Download unmatched list', misses.to_csv(index=False), 'spec2mrm_unmatched.csv')


molecule_background()

SHOW_QA = not PUBLIC or qa.online()  # public site without a Q&A database: no Q&A page

with st.sidebar:
    st.image(str(LOGO_LIGHT), width=250)
    page = (st.segmented_control('Page', ['Method builder', 'Q&A'], default='Method builder', key='page',
                                 label_visibility='collapsed') if SHOW_QA else None) or 'Method builder'
    with st.popover('Help', icon=':material/help:', width='stretch'):
        st.markdown(f"""**How to use Spec2MRM**

1. Paste or upload compound **names and SMILES**. Polarity is optional: both ion modes are
   searched and every mode with library spectra is returned.
2. Optional: add retention times. Choose your reversed-phase LC method and upload measured
   standards (`smiles, rt_min`).
3. Click **Generate MRM method**, review the transitions, and download the MassHunter CSV.

Collision energies are starting values (typically within about 10 eV); optimise them on your
instrument. PFAS and quaternary ammonium compounds without library spectra get transitions
from class rules (labelled "Rule" in the Source column). Remaining compounds get an in-silico
prediction from FIORA (labelled "Predicted: FIORA", low confidence). Compounds FIORA cannot
handle are listed as unmatched.

**Questions?** {'Open the **Q&A** page, or email' if SHOW_QA else 'Email'}
[{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL}).""")
    st.divider()
    if page == 'Method builder':
        st.markdown('1. Load compounds\n2. Choose an LC method\n3. Review and download')
    st.caption('This prototype searches the installed spectral library. CFM-ID is not run here. The unscheduled CSV uses the Agilent MassHunter column format.')
    if PUBLIC:
        st.caption(f'Up to {MAX_ROWS} rows per CSV. Your compound lists are processed for your session only and '
                   'are not stored.' + (' Q&A posts are public.' if SHOW_QA else ''))
    elif department_trial:
        st.caption(f'Up to {MAX_ROWS} rows per CSV. Computation runs on the host. '
                   'This trial uses HTTP; upload only data approved for testing.')
    else:
        st.caption(f'Up to {MAX_ROWS} rows per CSV. Files and predictions stay on this machine.')

st.badge('PUBLIC BETA' if PUBLIC else 'DEPARTMENT TRIAL' if department_trial else 'LOCAL PROTOTYPE', color='green')
if page == 'Q&A' and SHOW_QA:
    render_qa(developer=st.session_state.get('_admin', False) if PUBLIC else not department_trial)
    st.stop()

st.markdown('<p class="s2m-tagline">Generate starting MRM transitions and optional retention-time estimates '
            'from a compound name and SMILES.</p>', unsafe_allow_html=True)
st.caption('EXPERIMENTAL MS/MS LIBRARY  ·  CATBOOST COLLISION ENERGY  ·  2-STEP RETENTION TIME')

st.header('1 · Load compounds')
source = st.radio('Compound source', ['Built-in example', 'Paste compounds', 'Upload compound CSV'], horizontal=True)
parsed = None  # targets already parsed from pasted text
if source == 'Built-in example':
    raw = EXAMPLE
    with st.expander('Preview example compounds', expanded=True):
        st.dataframe(targets_from_csv(EXAMPLE)[['name', 'smiles', 'polarity']], hide_index=True, width='stretch')
elif source == 'Paste compounds':
    raw = None
    text = st.text_area('One compound per line', key='paste_text', height=160, max_chars=200_000,
                        placeholder=PASTE_PLACEHOLDER)
    st.caption('Write name and SMILES on each line, separated by a comma or a tab. You can copy two '
               'columns straight from Excel. Spectra are searched in both positive and negative mode, '
               'and every mode with library spectra is returned. To restrict a line, add Positive or Negative.')
    default_pol = st.radio('Ion mode to search', ('Both', *POLARITIES), horizontal=True, key='paste_polarity')
    if text.strip():
        try:
            parsed = targets_from_text(text, default_pol)
        except ValueError as error:
            st.error(str(error))
        else:
            preview = parsed.groupby('input_row', sort=True).agg(
                name=('name', 'first'), smiles=('smiles', 'first'),
                ion_mode=('polarity', lambda p: ' + '.join(p)))
            st.dataframe(preview, hide_index=True, width='stretch',
                         column_config={'name': 'Compound', 'smiles': 'SMILES', 'ion_mode': 'Ion mode searched'})
else:
    upload = st.file_uploader('Compound list', type=['csv'], max_upload_size=1)
    raw = upload.getvalue() if upload is not None else None
    st.caption('Required columns: name, smiles. Optional: polarity (Positive, Negative or Both). '
               'A blank or missing polarity searches both modes.')
download('Download compound template', TARGET_TEMPLATE, 'spec2mrm_compounds_template.csv')

st.header('2 · Add retention time (optional)')
modes = {'No RT · unscheduled MRM': 'none', 'Use my LC method and anchors': 'custom'}
if source == 'Built-in example' and not department_trial and demo_available():
    modes['Try the local BEH C18 demo'] = 'demo'
selection = st.radio('RT source', list(modes), horizontal=True)
mode, profiles = modes[selection], {}
incomplete = []
if mode == 'demo':
    st.info('Uses the lab’s example BEH C18 method and local standards. These RTs are specific to that setup.')
if mode != 'none':
    st.markdown(f'<div class="s2m-credit" style="margin-top:0.3rem;border:0;padding:0">\n\n{RT_CREDIT}\n\n</div>',
                unsafe_allow_html=True)
if mode == 'custom':
    st.caption('RT estimates are available for reversed-phase LC. Enable the profile for each polarity you want to map.')
    download('Download anchor template', ANCHOR_TEMPLATE, 'spec2mrm_anchors_template.csv')
    try:
        available = columns_for_rt()
        if not available:
            raise ValueError('The installed tables contain no supported columns.')
    except (OSError, ValueError) as error:
        st.warning(f'RT column list not available: {error}. Select “No RT” to continue.')
        incomplete.append('RT column list')
    else:
        for pol in POLARITIES:
            if st.checkbox(f'Configure {pol.lower()} method', key=f'use_{pol}'):
                setting = profile_form(pol, available)
                if setting:
                    profiles[pol] = setting
                else:
                    incomplete.append(f'{pol} method')
        if not profiles and not incomplete:
            incomplete.append('LC method')

insilico = st.checkbox('Predict transitions in silico (FIORA) for compounds without library spectra or class rules',
                       value=True, key='use_fiora',
                       help='Adds about 30 s per run plus a few seconds per compound. Predictions are low confidence.')
if st.button('Generate MRM method', type='primary'):
    st.session_state.pop('spec2mrm_result', None)
    try:
        if source == 'Paste compounds' and parsed is None:
            raise ValueError('Paste at least one valid compound first.')
        if source != 'Paste compounds' and raw is None:
            raise ValueError('Choose a compound CSV first.')
        if incomplete:
            raise ValueError('Complete the settings and anchor upload for: ' + ', '.join(incomplete) + '.')
        with st.spinner('Searching the library and generating predictions…'):
            targets = parsed if parsed is not None else targets_from_csv(raw)
            st.session_state['spec2mrm_result'] = generate(targets, mode, profiles, insilico=insilico)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        st.error(str(error))

if 'spec2mrm_result' in st.session_state:
    st.divider()
    st.caption('Results are from the last completed run. Generate again after changing inputs.')
    render_results(st.session_state['spec2mrm_result'])

st.markdown(f"""<div class="s2m-credit">

**Credits.** {RT_CREDIT}
Product ions come from public experimental spectral libraries (GNPS, MoNA and MassBank; see below).
Starting collision energies come from Spec2MRM's CatBoost model, trained on measured Agilent 6470 settings.
In-silico transitions: {fiora_predict.CITATION}
Background structures: PubChem 3D conformers. Contact: [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL}).

</div>""", unsafe_allow_html=True)
credits = library_credits()
if credits:
    with st.expander('Spectral libraries behind the transition table'):
        st.caption('Transitions are derived values (precursor ions, product-ion m/z and how often each ion was seen), '
                   'computed from these libraries as distributed by GNPS and MoNA. No spectra are redistributed. '
                   'Spectra per library:')
        st.markdown('  \n'.join(f'{escape_md(name)}: {count:,}' for name, count in credits))
