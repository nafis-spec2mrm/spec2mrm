"""Public-site behaviour: open access, no lab demo, public library table, admin-only Q&A rights.

Run from this folder:  python public_test.py   (works on the lab PC and in the deploy folder)
"""
import hashlib
import os
from pathlib import Path
import tempfile

os.environ['SPEC2MRM_PUBLIC'] = '1'
os.environ['SPEC2MRM_LIBRARY'] = 'public'
folder = tempfile.mkdtemp()
os.environ['SPEC2MRM_QA_DB'] = str(Path(folder) / 'qa_public_test.sqlite3')
os.environ['SPEC2MRM_ADMIN_SHA256'] = hashlib.sha256(b'correct horse battery staple').hexdigest()

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).with_name('app.py'))

app = AppTest.from_file(APP, default_timeout=120).run()
assert not app.exception, app.exception
assert not any('access code' in m.value.lower() for m in app.markdown), 'public site must not ask for a code'
assert any('PUBLIC BETA' in m.value for m in app.markdown), 'PUBLIC BETA badge missing'
assert all('BEH C18 demo' not in o for r in app.radio for o in r.options), 'lab demo must be hidden'
assert any('Spectral libraries' in e.label for e in app.expander), 'library credits missing'
print('PASS: open access, PUBLIC BETA badge, lab demo hidden, library credits shown.', flush=True)

app.radio[0].set_value('Paste compounds').run()
app.text_area(key='paste_text').input(
    'Caffeine, Cn1c(=O)c2c(ncn2C)n(C)c1=O\n'
    'PFOS, OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F\n'
    'Bis(2-butoxyethyl) phthalate, CCCCOCCOC(=O)c1ccccc1C(=O)OCCOCCCC, Positive\n').run()
app.button[0].click().run()
assert not app.exception and not app.error, [e.value for e in app.error]
t = app.session_state['spec2mrm_result']['transitions']
src = t.groupby('name').fragment_source.first()
assert src['Caffeine'] == src['PFOS'] == 'experimental library consensus', src
assert src['Bis(2-butoxyethyl) phthalate'] == 'in-silico: FIORA', src
assert t.collision_energy_eV.notna().all() and app.session_state['spec2mrm_result']['masshunter']
print('PASS: public library table, FIORA and CE work in public mode.', flush=True)

if os.environ.get('SPEC2MRM_QA_URL'):
    app = AppTest.from_file(APP, default_timeout=120)
    app.session_state['page'] = 'Q&A'
    app.run()
    app.text_area[0].input('Is negative mode supported?')
    next(b for b in app.button if b.label == 'Post question').click().run()
    assert not any(b.label == 'Delete this question' for b in app.button), 'visitors must not delete'
    form_password = next(w for w in app.text_input if w.label == 'Admin password')
    form_password.input('wrong')
    next(b for b in app.button if b.label == 'Sign in').click().run()
    assert any('not correct' in e.value for e in app.error)
    assert '_admin' not in app.session_state or not app.session_state['_admin']
    form_password = next(w for w in app.text_input if w.label == 'Admin password')
    form_password.input('correct horse battery staple')
    next(b for b in app.button if b.label == 'Sign in').click().run()
    assert app.session_state['_admin'] is True
    assert any(b.label == 'Delete this question' for b in app.button), 'admin must be able to delete'
    print('PASS: Q&A visitors cannot delete; admin password unlocks developer mode; wrong password refused.', flush=True)
else:
    app = AppTest.from_file(APP, default_timeout=120)
    app.session_state['page'] = 'Q&A'
    app.run()
    assert not app.exception, app.exception
    assert not any(b.label == 'Post question' for b in app.button), 'Q&A must be hidden without a database'
    assert not any('Q&A posts' in c.value for c in app.caption)
    print('PASS: no Q&A database set, so the Q&A page is hidden.', flush=True)

# Retention time with a user LC profile. The anchors are invented values for a
# functional test only (no lab standards are shipped with the public app).
anchors = ('smiles,rt_min\n'
           'Cn1c(=O)c2c(ncn2C)n(C)c1=O,3.1\n'           # caffeine
           'CC(=O)Nc1ccc(O)cc1,2.4\n'                   # acetaminophen
           'Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1,4.6\n'      # sulfamethoxazole
           'CC(C)Cc1ccc(C(C)C(=O)O)cc1,12.8\n'          # ibuprofen
           'O=C(O)c1ccccc1O,5.2\n'                      # salicylic acid
           'CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1,15.9\n')    # C12-BAC
app = AppTest.from_file(APP, default_timeout=300).run()
app.radio[1].set_value('Use my LC method and anchors').run()
app.checkbox(key='use_Positive').check().run()
app.file_uploader(key='Positive_anchor').upload('anchors.csv', anchors.encode(), 'text/csv')
app.text_input(key='Positive_name').set_value('Public test method')
app.selectbox(key='Positive_column').set_value('Waters ACQUITY UPLC BEH C18')
for key, value in {'ph': 3.0, 'run': 21.0, 'length': 100.0, 'id': 2.1, 'particle': 1.7,
                   'flow': 0.3, 'temp': 40.0}.items():
    app.number_input(key=f'Positive_{key}').set_value(value)
app.button[0].click().run()
assert not app.exception and not app.error, [e.value for e in app.error]
result = app.session_state['spec2mrm_result']
assert not result['rt_error'], result['rt_error']
assert result['rt'].predicted_rt_min.notna().sum() == 3, result['rt']
print('PASS: 2-step RT runs with a user LC profile.', flush=True)
