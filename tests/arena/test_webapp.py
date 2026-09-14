import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlsplit, parse_qs
from tools.arena.webapp import create_app, COOKIE


class WebSecurityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = dict(origin='https://arena.test', repository='owner/repo',
                        client_id='test', client_secret='test-secret', data_dir=str(self.root))
        self.members()
        (self.root/'data.json').write_text('{"private_results":42}')
        self.app = create_app(self.cfg)
        self.client = self.app.test_client()

    def tearDown(self): self.tmp.cleanup()

    def members(self, ids=None, age=0):
        (self.root/'members.json').write_text(json.dumps(dict(repository='owner/repo',
            user_ids=[7] if ids is None else ids, checked_at=time.time()-age)))

    def get(self, url, **kw): return self.client.get(url, base_url=self.cfg['origin'], buffered=True, **kw)

    def login(self, uid=7):
        r=self.get('/auth/login')
        q=parse_qs(urlsplit(r.location).query,keep_blank_values=True)
        self.assertEqual(q['scope'],[''])
        self.assertEqual(q['code_challenge_method'],['S256'])
        token=Mock();token.json.return_value={'access_token':'never-store-this'}
        user=Mock();user.json.return_value={'id':uid}
        with patch('tools.arena.webapp.requests.post',return_value=token), patch('tools.arena.webapp.requests.get',return_value=user):
            result=self.get('/auth/callback?state='+q['state'][0]+'&code=one-use-code')
        return result,q['state'][0]

    def test_anonymous_cannot_read_files_or_data(self):
        for path in ['/api/data','/dashboard.js']:
            r=self.get(path);self.assertEqual(r.status_code,401)
            self.assertNotIn(b'private_results',r.data)
        for path in ['/data.json','/members.json','/.git/config','/../members.json','/static/data.json']:
            self.assertEqual(self.get(path).status_code,404)
        self.assertNotIn(b'private_results',self.get('/').data)

    def test_login_cookie_headers_and_replay(self):
        r,state=self.login();self.assertEqual(r.status_code,302)
        cookie=r.headers.getlist('Set-Cookie')[0]
        for flag in ['Secure','HttpOnly','SameSite=Lax','Path=/']:self.assertIn(flag,cookie)
        self.assertNotIn('never-store-this',cookie)
        r=self.get('/api/data');self.assertEqual(r.status_code,200)
        self.assertEqual(r.headers['Cache-Control'],'no-store, private')
        self.assertIn("frame-ancestors 'none'",r.headers['Content-Security-Policy'])
        self.assertEqual(self.get('/auth/callback?state='+state+'&code=x').status_code,400)

    def test_nonmember_rejected(self):
        self.assertEqual(self.login(999)[0].status_code,403)
        self.assertEqual(self.get('/api/data').status_code,401)

    def test_revocation_applies_to_existing_session(self):
        self.login();self.members(ids=[])
        self.assertEqual(self.get('/api/data').status_code,401)

    def test_stale_membership_fails_closed(self):
        self.login();self.members(age=301)
        self.assertEqual(self.get('/api/data').status_code,401)

    def test_state_forgery_and_host_rejected(self):
        self.assertEqual(self.get('/auth/callback?state=forged&code=x').status_code,400)
        self.assertEqual(self.client.get('/',base_url='https://evil.test').status_code,400)

    def test_logout_requires_same_origin(self):
        self.login()
        self.assertEqual(self.client.post('/auth/logout',base_url=self.cfg['origin'],headers={'Origin':'https://evil.test'}).status_code,403)
        self.assertEqual(self.client.post('/auth/logout',base_url=self.cfg['origin'],headers={'Origin':self.cfg['origin']}).status_code,302)
        self.assertEqual(self.get('/api/data').status_code,401)

    def test_github_error_does_not_create_session(self):
        r=self.get('/auth/login');state=parse_qs(urlsplit(r.location).query)['state'][0]
        import requests
        with patch('tools.arena.webapp.requests.post',side_effect=requests.Timeout):
            self.assertEqual(self.get('/auth/callback?state='+state+'&code=x').status_code,502)
        self.assertEqual(self.get('/api/data').status_code,401)
