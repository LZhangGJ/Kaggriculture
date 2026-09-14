"""Persistent Elo from continuous, seat-swapped pairs only. No double counting."""
import sqlite3
from contextlib import closing
from pathlib import Path
from collections import defaultdict
from . import schedule
from .store import digest, read, write, now


def update(root):
    root=Path(root)
    with closing(sqlite3.connect(root/'private/elo.sqlite')) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS pairs (id TEXT PRIMARY KEY)')
        db.execute('CREATE TABLE IF NOT EXISTS rounds (id TEXT PRIMARY KEY)')
        db.execute('CREATE TABLE IF NOT EXISTS ratings (contract TEXT, agent TEXT, elo REAL, games INTEGER, wins REAL, PRIMARY KEY(contract,agent))')
        db.execute('CREATE TABLE IF NOT EXISTS history (contract TEXT, agent TEXT, games INTEGER, at TEXT, elo REAL, kind TEXT, PRIMARY KEY(contract,agent,games))')
        # Existing ratings are a baseline, never a reconstructed sequence of games.
        stamp=now()
        db.execute("INSERT OR IGNORE INTO history SELECT contract,agent,games,?,elo,'baseline' FROM ratings",(stamp,))
        for path in sorted((root/'runs').glob('continuous-*/manifest.json')):
            if db.execute('SELECT 1 FROM rounds WHERE id=?',(path.parent.name,)).fetchone():continue
            m=read(path);groups=defaultdict(list)
            for aid in m['agents']:db.execute('INSERT OR IGNORE INTO ratings VALUES (?,?,1500,0,0)',(m['contract_hash'],aid))
            fully_scored=True
            for g in m['games']:groups[(g['seed'],tuple(sorted(g['agents'])))].append(g)
            for (seed,agents),games in sorted(groups.items()):
                pid=digest([m['contract_hash'],seed,agents])
                if db.execute('SELECT 1 FROM pairs WHERE id=?',(pid,)).fetchone():continue
                results=[read(path.parent/'games'/f"{g['id']}.json") for g in games]
                if len(games)!=2 or not all(r and r.get('resolved') for r in results):
                    fully_scored=False
                    continue
                for g,r in zip(games,results):schedule.validate_result(g,r)
                a,b=agents;points=0
                for g,r in zip(games,results):
                    points+=.5 if r['outcome']=='draw' else float(r['outcome']==f"win{g['agents'].index(a)}")
                contract=m['contract_hash']
                for aid in agents:db.execute('INSERT OR IGNORE INTO ratings VALUES (?,?,1500,0,0)',(contract,aid))
                ra,rb=[db.execute('SELECT elo FROM ratings WHERE contract=? AND agent=?',(contract,aid)).fetchone()[0] for aid in agents]
                delta=32*(points/2-1/(1+10**((rb-ra)/400)))
                db.execute('UPDATE ratings SET elo=elo+?, games=games+2,wins=wins+? WHERE contract=? AND agent=?',(delta,points,contract,a))
                db.execute('UPDATE ratings SET elo=elo-?, games=games+2,wins=wins+? WHERE contract=? AND agent=?',(delta,2-points,contract,b))
                db.execute('INSERT INTO pairs VALUES (?)',(pid,))
                for aid in agents:
                    db.execute("INSERT INTO history SELECT contract,agent,games,?,elo,'pair' FROM ratings WHERE contract=? AND agent=?",(stamp,contract,aid))
            if fully_scored:db.execute('INSERT OR IGNORE INTO rounds VALUES (?)',(m['id'],))
        rows=[dict(contract=c,agent=a,elo=e,games=g,score=p/g if g else None) for c,a,e,g,p in db.execute('SELECT * FROM ratings ORDER BY contract,elo DESC')]
        history=[]
        for r in rows:
            points=list(reversed(db.execute('SELECT games,at,elo,kind FROM history WHERE contract=? AND agent=? ORDER BY games DESC LIMIT 5050',(r['contract'],r['agent'])).fetchall()))
            window=[];series=[]
            retired=read(root/'agents'/(r['agent']+'.json'),{}).get('retired_at')
            for games,at,value,kind in points:
                if retired and at>retired:continue
                if kind=='pair':window.append(value);window=window[-50:]
                else:window=[]
                mean=sum(window)/50 if len(window)==50 else None
                series.append([games,at,round(value,3),round(mean,3) if mean is not None else None])
            history.append(dict(agent=r['agent'],contract=r['contract'],points=series[-5000:]))
    result=dict(updated=now(),method='Internal Elo, not a replica of Kaggle live ratings. Start 1500; K=32 per completed seat-swapped pair; draws half a point; separate table per execution contract',ratings=rows,
                history=history,history_note='Latest 5,000 recorded points per version and contract. Older points are saved dashboard snapshots, not per-game history. New points record each seat-pair update; time is processing time, so batches share a timestamp. The average requires 50 consecutive paired updates / 100 games. Full history stays stored on the server.')
    write(root/'continuous-elo.json',result)
    return result
