import gzip,json,pathlib,importlib.util,ctypes,collections,time
B=pathlib.Path('/mnt/data/fix1_work');r=B/'bundle';s=importlib.util.spec_from_file_location('needs_main',r/'main.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
j=json.load(gzip.open(B/'feedback/own_traces/candidate_r3/submission_56149565_201542742_seat0.json.gz','rt'));a=m.create_agent(binary_path=B/'audit/inspect.so');a.lib.td_live_needs.argtypes=[ctypes.c_void_p];a.lib.td_live_needs.restype=ctypes.c_char_p
rows=[]
for t,o in enumerate(j['observations'][:-1]):
 out=a(o,j['configuration']);assert out==j['own_actions'][t]
 z=json.loads(a.lib.td_live_needs(a.handle));rows.append(z)
 if o['hour']==0:
  acts=j['own_actions'][t:min(t+24,719)];orders=[x for ac in acts for x in ac['market']];fert=sum(x[0]=='FERTILIZE' for ac in acts for x in [ac['farmer'],*ac['hands']]);feed=sum(x[0]=='FEED' for ac in acts for x in [ac['farmer'],*ac['hands']]);print(o['day'],'cash',o['farms'][0]['money'],'needW/F',z['daily_need'][0],z['daily_need'][8],'shedW/F',o['private']['shed']['WHEAT'],o['private']['shed']['FERTILIZER'],'ACT FEED/FERT',feed,fert,'boughtF',sum(x[2] for x in orders if x[:2]==['BUY_PRODUCT','FERTILIZER']),'first_orders',out['market'])
a.close()
with gzip.open(B/'logs/live_needs_r3.json.gz','wt') as f:json.dump(rows,f)
