"""Plan grammar and a bounded integer feasibility repair."""
import copy
import numpy as np


def simple_plan(kind=0):
    rows=[]
    for phase in range(5):
        row=[0]*14
        row[kind]=8 if kind<5 else 4
        row[8:]=[6,1,150,0,1,27]
        rows.append(row)
    return rows


def repair(plan):
    """Project integer targets onto field capacity and remaining maturation time.

    This is deterministic feasibility repair, not an optimal schedule solver.
    Executable commands still pass through the exact sequential ledger.
    """
    p=np.asarray(plan,dtype=np.int64).copy()
    if p.shape!=(5,14): raise ValueError('expected five stages of fourteen fields')
    lo=np.array([0]*8+[0,1,0,0,0,0]); hi=np.array([80]*8+[16,4,3000,23,1,29])
    p=np.clip(p,lo,hi)
    first=np.array([2,2,8,10,10,4,8,6])
    for stage,row in enumerate(p):
        row[:8][stage*6+first>=30]=0
        capacity=min(100,int(row[9])*25)
        while row[:8].sum()>capacity:
            row[int(row[:8].argmax())]-=1
    return p.tolist()


def random_plan(rng):
    p=[]
    for stage in range(5):
        mix=rng.multinomial(int(rng.integers(8,35)),rng.dirichlet(np.ones(8)*.5))
        p.append([*mix.tolist(),int(rng.integers(4,13)),int(rng.integers(1,4)),
                  int(rng.choice([0,100,300])),int(rng.choice([0,8,16])),1,27])
    return repair(p)


def mutate(plan,rng,large=False,operator=None):
    p=copy.deepcopy(plan)
    op=int(rng.integers(4)) if operator is None else operator
    stages=list(range(5)) if large and rng.random()<.3 else [int(rng.integers(5))]
    for stage in stages:
        if op==0:
            a,b=rng.choice(8,2,replace=False); amount=int(rng.integers(2,12))
            p[stage][a]-=amount;p[stage][b]+=amount
        elif op==1:
            p[stage][8]+=int(rng.choice([-3,-1,1,3]));p[stage][9]+=int(rng.choice([-1,0,1]))
        elif op==2:
            p[stage][10]=int(rng.choice([0,100,300,750]));p[stage][11]=int(rng.choice([0,8,16,22]))
        else:
            p[stage][:8]=random_plan(rng)[stage][:8];p[stage][13]=int(rng.integers(20,30))
    return repair(p)


def solver_repair(plan):
    """Minimum-change CP-SAT repair of stage allocation and maturation bounds.

    Cash flow and worker routes remain the feedback executor's responsibility.
    This model does not claim to solve the complete season scheduling problem.
    """
    from ortools.sat.python import cp_model
    target=np.asarray(plan,dtype=np.int64);base=np.asarray(repair(plan))
    model=cp_model.CpModel();xs=[];loss=[]
    for stage in range(5):
        row=[]
        for it in range(8):
            upper=0 if stage*6+[2,2,8,10,10,4,8,6][it]>=30 else 80
            x=model.new_int_var(0,upper,f'x{stage}_{it}');row.append(x)
            d=model.new_int_var(0,200,f'd{stage}_{it}');model.add_abs_equality(d,x-int(target[stage,it]));loss.append(d)
        model.add(sum(row)<=int(base[stage,9])*25);xs.append(row)
    model.minimize(sum(loss));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
    solver.parameters.max_time_in_seconds=.1
    status=solver.solve(model)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError('allocation repair failed')
    for stage,row in enumerate(xs):base[stage,:8]=[solver.value(x) for x in row]
    return base.tolist()
