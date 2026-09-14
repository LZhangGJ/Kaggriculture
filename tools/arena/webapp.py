"""Read-only arena site. Public GitHub identity plus a fresh repo-member list."""
import base64
import hashlib
import json
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import requests
from flask import Flask, abort, make_response, redirect, request, send_file

COOKIE = '__Host-arena-session'
STATE_COOKIE = '__Host-arena-state'


def create_app(config):
    origin = config['origin'].rstrip('/')
    parsed = urlsplit(origin)
    if parsed.scheme != 'https' or parsed.path or parsed.query or parsed.fragment:
        raise ValueError('An HTTPS origin is required')
    root = Path(config['data_dir'])
    assets = Path(__file__).with_name('web_assets')
    app = Flask(__name__, static_folder=None)
    app.config.update(MAX_CONTENT_LENGTH=4096, TRUSTED_HOSTS=[parsed.netloc])
    sessions, states, starts = {}, {}, []
    lock = threading.Lock()

    def allowed(uid):
        try:
            members = json.loads((root / 'members.json').read_text())
            age = time.time() - members['checked_at']
            return (members['repository'] == config['repository'] and 0 <= age <= 300
                    and uid in members['user_ids'])
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def identity():
        with lock:
            sid = request.cookies.get(COOKIE)
            item = sessions.get(sid)
            if item and item['expires'] > time.time() and allowed(item['id']):
                return item
            sessions.pop(sid, None)
        return None

    def cookie(response, name, value, age):
        response.set_cookie(name, value, max_age=age, secure=True, httponly=True,
                            samesite='Lax', path='/')
        return response

    @app.after_request
    def headers(response):
        response.headers.update({
            'Cache-Control': 'no-store, private', 'Pragma': 'no-cache',
            'Content-Security-Policy': "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'; object-src 'none'",
            'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
            'Referrer-Policy': 'no-referrer', 'Strict-Transport-Security': 'max-age=31536000',
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
            'X-Robots-Tag': 'noindex, nofollow, noarchive',
        })
        return response

    @app.get('/healthz')
    def health():
        return 'ok\n'

    @app.get('/skin.css')
    def skin():
        return send_file(assets / 'skin.css')

    @app.get('/')
    def home():
        return send_file(assets / ('index.html' if identity() else 'login.html'))

    @app.get('/dashboard.js')
    def script():
        if not identity(): abort(401)
        return send_file(assets / 'dashboard.js')

    @app.get('/api/data')
    def data():
        if not identity(): abort(401)
        # Explicit single-file allowlist: never serve an arena directory.
        return send_file(root / 'data.json', mimetype='application/json', conditional=False)

    @app.get('/auth/login')
    def login():
        if not config.get('client_secret'): abort(503)
        now = time.time()
        with lock:
            for mapping in (states, sessions):
                for key in list(mapping):
                    if mapping[key]['expires'] <= now: del mapping[key]
            starts[:] = [t for t in starts if now-t < 60]
            if len(starts) >= 30 or len(states) >= 512 or len(sessions) >= 256: abort(429)
            starts.append(now)
            state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(64)
            states[state] = dict(verifier=verifier, expires=now+600)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
        query = urlencode(dict(client_id=config['client_id'], redirect_uri=origin+'/auth/callback',
                               state=state, scope='', code_challenge=challenge, code_challenge_method='S256'))
        return cookie(redirect('https://github.com/login/oauth/authorize?'+query), STATE_COOKIE, state, 600)

    @app.get('/auth/callback')
    def callback():
        state = request.args.get('state', '')
        if not state or not secrets.compare_digest(state, request.cookies.get(STATE_COOKIE, '')):
            abort(400)
        with lock:
            pending = states.pop(state, None)
        if not pending or pending['expires'] <= time.time(): abort(400)
        code = request.args.get('code', '')
        if not code or len(code) > 512: abort(400)
        try:
            response = requests.post('https://github.com/login/oauth/access_token', data={
                'client_id': config['client_id'], 'client_secret': config['client_secret'],
                'code': code, 'redirect_uri': origin+'/auth/callback', 'code_verifier': pending['verifier'],
            }, headers={'Accept': 'application/json'}, timeout=10, allow_redirects=False)
            response.raise_for_status()
            token = response.json()['access_token']
            response = requests.get('https://api.github.com/user', headers={
                'Authorization': 'Bearer '+token, 'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28',
            }, timeout=10, allow_redirects=False)
            response.raise_for_status()
            uid = response.json()['id']
            # Tokens are never written to disk, cookies, logs, or the session store.
        except (requests.RequestException, ValueError, KeyError):
            abort(502)
        if not allowed(uid): abort(403)
        sid = secrets.token_urlsafe(32)
        with lock:
            sessions.pop(request.cookies.get(COOKIE), None)
            sessions[sid] = dict(id=uid, expires=time.time()+28800)
        result = cookie(redirect('/'), COOKIE, sid, 28800)
        return cookie(result, STATE_COOKIE, '', 0)

    @app.post('/auth/logout')
    def logout():
        if request.headers.get('Origin') != origin: abort(403)
        with lock: sessions.pop(request.cookies.get(COOKIE), None)
        return cookie(redirect('/'), COOKIE, '', 0)

    return app


if __name__ == '__main__':
    import sys
    from waitress import serve
    cfg = json.loads(Path(sys.argv[1]).read_text())
    serve(create_app(cfg), host='127.0.0.1', port=cfg.get('port', 8766), threads=4,
          connection_limit=64, channel_timeout=20, max_request_body_size=4096,
          max_request_header_size=8192, ident='Arena')
