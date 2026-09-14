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
                        client_id='test', client_secret='test-secret', data_dir=str(self.root),upload_dir=str(self.root/'uploads'))
        (self.root/'uploads').mkdir()
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

    def test_upload_auth_origin_and_dedup(self):
        import io
        def post(origin='https://arena.test'):
            return self.client.post('/api/uploads',base_url=self.cfg['origin'],headers={'Origin':origin},
                data={'name':'My agent','file':(io.BytesIO(b'def agent(obs): return {}'),'agent.py')})
        self.assertEqual(post().status_code,401)
        self.login()
        self.assertEqual(post('https://evil.test').status_code,403)
        r=post();self.assertEqual(r.status_code,202)
        self.assertEqual(post().status_code,200)
        self.assertEqual(len(list((self.root/'uploads').glob('*.bin'))),1)
        self.assertEqual(len(self.get('/api/uploads').json['uploads']),1)

    def test_uploads_are_private_per_user(self):
        import io
        self.login()
        self.client.post('/api/uploads',base_url=self.cfg['origin'],headers={'Origin':self.cfg['origin']},
            data={'name':'Private upload','file':(io.BytesIO(b'pass'),'agent.py')})
        self.members(ids=[7,8]);self.login(8)
        self.assertEqual(self.get('/api/uploads').json['uploads'],[])

    def test_upload_larger_than_form_parser_buffer(self):
        import io
        self.login()
        r=self.client.post('/api/uploads',base_url=self.cfg['origin'],headers={'Origin':self.cfg['origin']},
            data={'name':'Large file','file':(io.BytesIO(b'#'+b'x'*200000),'agent.py')})
        self.assertEqual(r.status_code,202)

    def test_package_traversal_and_execution(self):
        import zipfile
        from tools.arena.web_uploads import package
        from tools.arena.store import file_hash
        p=self.root/'input.bin';dest=self.root/'agent.zip'
        marker=self.root/'must-not-exist'
        p.write_text('raise RuntimeError("must never execute on host")')
        m=dict(sha256=file_hash(p),format='py',interface='kaggle',name='test',version='1',user_id=7)
        manifest=package(p,m,dest)
        self.assertEqual(manifest['run'],['python','_arena_bridge.py'])
        self.assertFalse(marker.exists())
        with zipfile.ZipFile(p,'w') as z:z.writestr('../escape.py','bad')
        m.update(sha256=file_hash(p),format='zip')
        with self.assertRaises(ValueError):package(p,m,dest)

    def test_upload_enters_intake_once_and_bad_tar_is_rejected(self):
        from tools.arena import store
        from tools.arena.web_uploads import import_pending,statuses
        arena=self.root/'arena';store.init(arena)
        inbox=arena/'private/web-uploads';inbox.mkdir()
        rid='a'*32;source=inbox/(rid+'.bin');source.write_text('def agent(obs): return {}')
        meta=dict(id=rid,user_id=7,name='Valid upload',version='1',format='py',interface='kaggle',sha256=store.file_hash(source))
        store.write(inbox/(rid+'.json'),meta)
        import_pending(arena);import_pending(arena)
        self.assertEqual(len(list((arena/'agents').glob('*.json'))),1)
        self.assertEqual(statuses(arena)[0]['status'],'registered')
        rid='b'*32;source=inbox/(rid+'.bin');source.write_bytes(b'invalid tar data')
        meta.update(id=rid,format='tar.gz',sha256=store.file_hash(source));store.write(inbox/(rid+'.json'),meta)
        import_pending(arena)
        self.assertEqual(statuses(arena)[1]['status'],'rejected')
