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
  if(a.retired)td.append(element('span','RETIRED','badge'));
  if(a.aliases?.length)td.append(element('small','Also published as: '+a.aliases.join(', '),'version'));
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
    renderEloHistory();
    $('title').textContent='Continuous Elo';$('description').textContent='Internal Elo, not Kaggle’s unpublished live formula. Fresh seeds, both seats. New versions start at 1500. K = 32 per completed seat pair. Early ratings can move sharply; check game counts.';
    $('timestamp').textContent='Last updated: '+date(data.continuous_elo.updated)+' · Last sync: '+date(data.continuous_status.at);
    const contracts=[...new Set(elo.map(r=>r.contract))];
    $('table').replaceChildren();
    for(const contract of contracts){
      if(contracts.length>1)$('table').append(element('h3','Contract '+contract.slice(0,10)));
      const rows=elo.filter(r=>r.contract===contract).sort((a,b)=>b.elo-a.elo).map((r,i)=>({r,i})).filter(({r})=>match(r.agent)).map(({r,i})=>[String(i+1),nameCell(r.agent),fmt(r.elo),fmt(r.games),pct(r.score)]);
      $('table').append(table(['Rank','Agent / version','Elo','Games','Win + ½ draw'],rows));
    }
    const ratedIds=new Set(elo.map(r=>r.agent));
    const waiting=[...active].filter(id=>!ratedIds.has(id)&&match(id));
    if(waiting.length){
      $('table').append(element('h3','Awaiting first rated games'),table(['Agent / version','State'],waiting.map(id=>[nameCell(id),'Queued for the next round'])));
    }
    const retired=data.agents.filter(a=>a.retired&&match(a.id));
    if(retired.length){
      $('table').append(element('h3','Retired public agents'),element('p','No new matches or notebook updates. Ratings below are frozen at retirement. Previously scheduled games may still finish in their original tournament.'));
      const rows=retired.flatMap(a=>(a.frozen_elo?.length?a.frozen_elo:[{}]).map(r=>[nameCell(a.id),fmt(r.elo),fmt(r.games),pct(r.score),date(a.retired_at)]));
      $('table').append(table(['Agent / version','Frozen Elo','Games','Win + ½ draw','Retired'],rows));
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
    $('table').replaceChildren(table(['Agent / version','Source','State','In arena'],data.agents.filter(a=>match(a.id)).map(a=>[nameCell(a.id),a.agent_type==='public'?'Public notebook':'Team',a.retired?'Retired — no longer updating':a.status,active.has(a.id)?'Yes':'No'])));
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

// Exact rating history is separate from tournament results and agent versions.
let historyAgents=null, historyMode='both', historyAxis='time', historyContract=null;
function renderEloHistory(){
  const host=element('section',null,'elo-history');$('controls').append(host);
  const histories=data.continuous_elo.history||[];
  host.append(element('h3','Elo over time'));
  host.append(element('p',data.continuous_elo.history_note||'Rating history has not been recorded yet.'));
  const active=new Set(data.roster.map(r=>r.agent));
  const available=histories.filter(h=>active.has(h.agent)&&h.points.length);
  if(!available.length){host.append(element('p','Waiting for the first recorded rating update.'));return;}
  const contracts=[...new Set(available.map(h=>h.contract))];
  if(!contracts.includes(historyContract))historyContract=contracts[0];
  const redraw=()=>{host.remove();renderEloHistory();};
  if(contracts.length>1){const sel=element('select');sel.setAttribute('aria-label','Evaluation contract');for(const c of contracts){const o=element('option','Contract '+c.slice(0,10));o.value=c;sel.append(o);}sel.value=historyContract;sel.onchange=()=>{historyContract=sel.value;historyAgents=null;redraw();};host.append(sel);}
  const pool=available.filter(h=>h.contract===historyContract).sort((a,b)=>b.points.at(-1)[2]-a.points.at(-1)[2]);
  if(historyAgents===null)historyAgents=new Set(pool.slice(0,3).map(h=>h.agent));
  const choices=element('details');choices.append(element('summary','Choose agents (up to six)'));
  for(const h of pool){const label=element('label'),cb=element('input');cb.type='checkbox';cb.checked=historyAgents.has(h.agent);cb.onchange=()=>{if(cb.checked&&historyAgents.size>=6){cb.checked=false;return;}if(cb.checked)historyAgents.add(h.agent);else historyAgents.delete(h.agent);redraw();};label.style.display='block';label.append(cb,document.createTextNode(' '+agent(h.agent).name+' · '+agent(h.agent).version));choices.append(label);}host.append(choices);
  const buttons=element('div');buttons.style.margin='12px 0';
  for(const [key,title]of [['both','Both'],['raw','Actual'],['average','100-game average']]){const b=element('button',title);b.setAttribute('aria-pressed',historyMode===key);b.onclick=()=>{historyMode=key;redraw();};buttons.append(b);}
  const axis=element('select');axis.setAttribute('aria-label','Chart horizontal axis');for(const [key,title]of [['time','Time (UTC)'],['games','Games per agent']]){const o=element('option',title);o.value=key;axis.append(o);}axis.value=historyAxis;axis.onchange=()=>{historyAxis=axis.value;redraw();};buttons.append(axis);host.append(buttons);
  const selected=pool.filter(h=>historyAgents.has(h.agent));if(!selected.length){host.append(element('p','Select an agent to show its history.'));return;}
  const colors=['#2166ac','#b65b08','#8a3f8c','#287650','#ba3545','#655ac7'];
  const xval=p=>historyAxis==='time'?Date.parse(p[1]):p[0];
  const all=selected.flatMap(h=>h.points);let lo=Math.min(...all.map(xval)),hi=Math.max(...all.map(xval));if(hi===lo)hi=lo+(historyAxis==='time'?60000:2);
  const values=all.flatMap(p=>historyMode==='raw'?[p[2]]:historyMode==='average'?(p[3]===null?[]:[p[3]]):[p[2],...(p[3]===null?[]:[p[3]])]);
  if(!values.length){host.append(element('p','The 100-game average appears after 50 recorded seat-pair updates. Switch to Actual to see the current ratings.'));return;}
  let ymin=Math.min(...values),ymax=Math.max(...values);const pad=Math.max(10,(ymax-ymin)*.1);ymin-=pad;ymax+=pad;
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 900 360');svg.setAttribute('role','img');svg.setAttribute('aria-label','Actual Elo and 100-game rolling average for selected agent versions');svg.style.cssText='width:100%;min-width:580px;background:white';
  const add=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!=null)n.textContent=text;svg.append(n);return n;};
  const x=v=>65+(v-lo)/(hi-lo)*805,y=v=>305-(v-ymin)/(ymax-ymin)*275;
  for(let i=0;i<=5;i++){const v=ymin+(ymax-ymin)*i/5;add('line',{x1:65,x2:870,y1:y(v),y2:y(v),stroke:'#e4e8eb'});add('text',{x:55,y:y(v)+5,'text-anchor':'end','font-size':14,fill:'#374151'},Math.round(v));}
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;const label=historyAxis==='time'?new Date(v).toLocaleString('en-US',{timeZone:'UTC',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false}):Math.round(v).toLocaleString();add('text',{x:x(v),y:332,'text-anchor':i===0?'start':i===4?'end':'middle','font-size':13,fill:'#374151'},label);}
  add('text',{x:16,y:180,transform:'rotate(-90 16 180)','font-size':15,fill:'#374151'},'Elo');
  const legend=element('p');
  selected.forEach((h,i)=>{const color=colors[i];const label=element('span',agent(h.agent).name+'  ');label.style.cssText='color:'+color+';margin-right:14px';legend.append(label);
    for(const [field,width,opacity]of [[2,1,historyMode==='both'?.4:1],[3,2.5,1]]){if((field===2&&historyMode==='average')||(field===3&&historyMode==='raw'))continue;let path='',started=false;for(const p of h.points){if(p[field]===null){started=false;continue;}path+=(started?'L':'M')+x(xval(p)).toFixed(2)+','+y(p[field]).toFixed(2);started=true;}add('path',{d:path,fill:'none',stroke:color,'stroke-width':width,opacity});const last=h.points.at(-1);if(last[field]!==null)add('circle',{cx:x(xval(last)),cy:y(last[field]),r:3,fill:color});}
  });
  host.append(legend);const scroll=element('div',null,'scroll');scroll.append(svg);host.append(scroll);
  const inspect=element('input');inspect.type='range';inspect.min=0;inspect.max=1000;inspect.value=1000;inspect.setAttribute('aria-label','Inspect chart position');inspect.style.width='100%';const detail=element('p');detail.setAttribute('aria-live','polite');
  const show=f=>{const target=lo+f*(hi-lo);detail.textContent=selected.map(h=>{const p=h.points.reduce((best,p)=>Math.abs(xval(p)-target)<Math.abs(xval(best)-target)?p:best,h.points[0]);return agent(h.agent).name+': '+fmt(p[2])+' Elo; average '+fmt(p[3])+'; '+fmt(p[0])+' games; '+date(p[1]);}).join(' | ');};inspect.oninput=()=>show(Number(inspect.value)/1000);svg.onpointermove=e=>{const rect=svg.getBoundingClientRect();const f=Math.max(0,Math.min(1,((e.clientX-rect.left)/rect.width*900-65)/805));inspect.value=Math.round(f*1000);show(f);};host.append(inspect,detail);show(1);
  host.append(element('small','Faint lines: actual ratings. Bold lines: rolling average. A fixed K keeps ratings responsive. Changes in opponent mix can also move ratings; use tournament results and game counts alongside this chart.'));
}
