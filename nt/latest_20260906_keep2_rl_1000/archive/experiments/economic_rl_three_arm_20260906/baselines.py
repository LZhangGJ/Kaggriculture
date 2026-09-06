from runtime import *
def main():
 pool=make_pool()
 for arm in ('c3auto','f3','c3j7'):
  for name,mode,path in [('keep',0,''),('random',3,''),('initial_stochastic',2,P/'training'/f'{arm}_r0/step000.bin')]:
   result=rollout(pool,arm,path,mode=mode,start=63000000,count=100,sample=9901)
   store_result(P/'baselines'/f'{arm}_{name}',result)
   assert summary(result)['status']=='PASS',summary(result)['errors']
   print('BASELINE',arm,name,summary(result)['overall'],flush=True)
if __name__=='__main__':main()
