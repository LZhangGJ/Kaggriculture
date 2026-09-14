'use strict';
let data, view = 'elo', runId, detailId;
const $ = id => document.getElementById(id);
const fmt = n => n == null ? '—' : Number(n).toLocaleString('en-US', {maximumFractionDigits: 1});
const pct = n => n == null ? '—' : (n * 100).toFixed(1) + '%';
const date = s => s ? new Date(s).toISOString().replace('T',' ').replace(/\.\d+Z$/, ' UTC') : 'Not yet';
function element(tag, text, cls) {const x=document.createElement(tag); if(text!=null)x.textContent=text; if(cls)x.className=cls;return x;}
function agent(id) {return data.agents.find(a=>a.id===id) || {id,name:id,version:'',agent_type:'team'};}
function nameCell(id, clickable=false) {
  const a=agent(id), td=element('td');
  const label=element(clickable?'button':'span', a.name, clickable?'agent-link':'');
  if(clickable) label.onclick=()=>{detailId=id;renderDetail();};td.append(label);
  if(a.agent_type==='public')td.append(element('span','PUBLIC','badge'));
  td.append(element('small',a.version,'version'));return td;
}
function table(headers, rows) {
  const t=element('table'),h=element('tr');headers.forEach(x=>h.append(element('th',x)));
  const head=element('thead');head.append(h);t.append(head);const body=element('tbody');
  rows.forEach(cells=>{const row=element('tr');cells.forEach(c=>row.append(c instanceof Node?c:element('td',c,'num')));body.append(row);});
  t.append(body);return t;
}
function renderDetail(){
  $('detail').replaceChildren();const r=data.runs.find(r=>r.run===runId);if(!r||!detailId)return;
  $('detail').append(element('h3',agent(detailId).name+' · By opponent'));
  const rows=Object.entries(r.matrix).filter(([k])=>k.startsWith(detailId+':')).map(([k,s])=>[
    nameCell(k.split(':')[1]),`${s.wins} / ${s.games}`,pct(s.win_rate),pct(s.score),fmt(s.draws),fmt(s.cash_margin),
    `${s.seats['0'].wins} / ${s.seats['0'].games}`,`${s.seats['1'].wins} / ${s.seats['1'].games}`]);
  const box=element('div',null,'scroll');box.append(table(['Opponent','Wins / games','Strict win rate','Win + ½ draw','Draws','Cash margin','Seat 0 W/G','Seat 1 W/G'],rows));$('detail').append(box);
}
function render(){
  if(!data)return;const search=$('search').value.toLowerCase();
  const match=id=>{const a=agent(id);return (a.name+' '+a.version).toLowerCase().includes(search);};
  const active=new Set(data.roster.map(r=>r.agent));
  const elo=data.continuous_elo.ratings.filter(r=>active.has(r.agent));
  $('freshness').textContent='Published '+date(data.updated);
  const age=(Date.now()-new Date(data.updated).getTime())/60000;
  $('notice').replaceChildren();
  if(age>15)$('notice').append(element('p',`Results are ${Math.round(age)} minutes old. The last verified snapshot remains visible.`,'warning'));
  if(data.continuous_status.status!=='running')$('notice').append(element('p','Continuous arena: '+data.continuous_status.status,'warning'));
  $('cards').replaceChildren();
  const daily=data.runs.filter(r=>r.kind==='daily');
  const tournaments=data.runs.filter(r=>['daily','cumulative'].includes(r.kind));
  const latest=daily.at(-1);
  const completed=[...daily].filter(r=>r.complete).sort((a,b)=>a.created.localeCompare(b.created)).at(-1);
  const team=completed?.components.length===1?Object.keys(completed.ratings).filter(id=>active.has(id)&&agent(id).agent_type==='team'&&Number.isFinite(completed.ratings[id])).sort((a,b)=>completed.ratings[b]-completed.ratings[a]):[];
  const leaders=new Set(elo.filter(r=>r.games>0).map(r=>r.contract)).size===1?elo.filter(r=>r.games>0).sort((a,b)=>b.elo-a.elo):[];
  const best=team[0], live=leaders[0];
  const cards=[['Active agents',data.roster.length,'Public and team versions'],['Rated games',fmt(data.continuous_elo.ratings.reduce((s,r)=>s+r.games,0)/2),'Completed games across all versions'],['Daily coverage',latest?`${fmt(latest.completed)} / ${fmt(latest.planned)}`:'Not started',latest?.complete?'Completed tournament':'Provisional tournament'],['Best team agent',best?agent(best).name:completed?'No comparable team result':daily.length?'First tournament in progress':'First tournament not started',best?completed.run+' · '+agent(best).version:'Selected from a completed tournament'],['Live Elo leader',live?agent(live.agent).name:'No comparable rated result',live?(agent(live.agent).agent_type==='public'?'PUBLIC':'Team')+' · '+fmt(live.elo)+' Elo · '+fmt(live.games)+' games':'Uses active agents under one evaluation contract']];
  for(const [label,value,note]of cards){const card=element('div',null,'card');card.append(element('small',label),element('strong',String(value),['Best team agent','Live Elo leader'].includes(label)?'champ':''),element('small',note));$('cards').append(card);}
  document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('selected',b.dataset.view===view));
  $('results-panel').hidden=view==='upload';$('upload-panel').hidden=view!=='upload';
  if(view==='upload'){loadUploads();return;}
  $('controls').replaceChildren();$('detail').replaceChildren();
  if(view==='elo'){
    $('title').textContent='Continuous Elo';$('description').textContent='Internal Elo, not Kaggle’s unpublished live formula. Fresh seeds, both seats. New versions start at 1500. K = 32 per completed seat pair. Early ratings can move sharply; check game counts.';
    $('timestamp').textContent='Last updated: '+date(data.continuous_elo.updated)+' · Last sync: '+date(data.continuous_status.at);
    const contracts=[...new Set(elo.map(r=>r.contract))];
    $('table').replaceChildren();
    for(const contract of contracts){
      if(contracts.length>1)$('table').append(element('h3','Contract '+contract.slice(0,10)));
      const rows=elo.filter(r=>r.contract===contract).sort((a,b)=>b.elo-a.elo).map((r,i)=>({r,i})).filter(({r})=>match(r.agent)).map(({r,i})=>[String(i+1),nameCell(r.agent),fmt(r.elo),fmt(r.games),pct(r.score)]);
      $('table').append(table(['Rank','Agent / version','Elo','Games','Win + ½ draw'],rows));
    }
  }else if(view==='bt'){
    $('title').textContent='Bradley–Terry tournaments';$('description').textContent='Draws count as half wins. No seat adjustment; weak regularization keeps estimates finite. Completed daily runs include 95% seed-bootstrap intervals. Cumulative views use resolved daily and continuous games between active versions within one contract; no intervals yet. Select an agent for opponent and seat results. Ratings across disconnected groups are not comparable.';
    if(!tournaments.length){$('table').textContent='No daily tournament yet.';$('timestamp').textContent='Last updated: Not yet';return;}
    if(!tournaments.some(r=>r.run===runId))runId=tournaments.at(-1).run;
    const select=element('select');select.setAttribute('aria-label','Tournament');
    for(const r of [...tournaments].reverse()){const o=element('option',r.run+(r.complete?' · complete':' · provisional'));o.value=r.run;select.append(o);}select.value=runId;select.onchange=()=>{runId=select.value;detailId=null;render();};$('controls').append(select);
    const r=tournaments.find(r=>r.run===runId);$('timestamp').textContent='Last updated: '+date(r.updated)+` · ${fmt(r.completed)} / ${fmt(r.planned)} games`;
    if(r.components.length>1)$('controls').append(element('p','Disconnected comparisons: compare ratings only inside each connected group.','warning'));
    const rows=Object.keys(r.ratings).sort((a,b)=>(r.ratings[b]??-Infinity)-(r.ratings[a]??-Infinity)).filter(match).map(id=>{const s=r.stats[id],ci=r.intervals[id];return[nameCell(id,true),(r.ratings[id]==null?'—':r.ratings[id].toFixed(2))+(ci?` [${ci[0].toFixed(2)}, ${ci[1].toFixed(2)}]`:''),`${s.wins} / ${s.losses} / ${s.draws}`,pct(s.win_rate),pct(s.score),fmt(s.cash_margin)];});
    $('table').replaceChildren(table(['Agent / version','BT [95% interval]','W / L / D','Strict wins','Win + ½ draw','Cash margin'],rows));renderDetail();
  }else{
    $('title').textContent='Agent roster';$('description').textContent='Exact versions keep separate results. Public notebooks are labeled. Pending versions must pass evaluation before joining the active pool.';
    $('timestamp').textContent='Last updated: '+date(data.updated);
    $('table').replaceChildren(table(['Agent / version','Source','State','In arena'],data.agents.filter(a=>match(a.id)).map(a=>[nameCell(a.id),a.agent_type==='public'?'Public notebook':'Team',a.status,active.has(a.id)?'Yes':'No'])));
  }
}
async function refresh(){try{const r=await fetch('/api/data',{cache:'no-store'});if(r.status===401){location.replace('/');return;}if(!r.ok)throw Error();data=await r.json();render();}catch{$('notice').replaceChildren(element('p','Could not refresh. Showing the last received results; check their timestamps.','warning'));}}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{view=b.dataset.view;render();});$('search').oninput=render;refresh();setInterval(refresh,60000);

async function loadUploads(){
  try{const r=await fetch('/api/uploads',{cache:'no-store'});if(r.status===401){location.replace('/');return;}if(!r.ok)throw Error();
  const {uploads}=await r.json();$('upload-list').replaceChildren(uploads.length?table(['Agent','Version','Status','Details'],uploads.map(u=>[element('td',u.name),u.version,u.status,element('td',u.message||'')])):element('p','No uploads yet.'));
  }catch{$('upload-list').textContent='Could not refresh submission status. Please try again.';}
}
$('upload-form').onsubmit=async e=>{
  e.preventDefault();const form=e.currentTarget,f=form.elements.file.files[0];
  if(!f||f.size>64*1024*1024){$('upload-message').textContent='Choose a file up to 64 MiB.';return;}
  $('upload-button').disabled=true;$('upload-message').textContent='Uploading… Keep this page open.';
  try{const r=await fetch('/api/uploads',{method:'POST',body:new FormData(form)});if(r.status===401){location.replace('/');return;}
    const result=await r.json();if(!r.ok)throw Error(result.error||'Upload failed. Please try again.');
    $('upload-message').textContent='Received. Your agent is queued for validation. Status will update here automatically.';
    form.elements.file.value='';await loadUploads();
  }catch(error){$('upload-message').textContent=error.message;}finally{$('upload-button').disabled=false;}
};
