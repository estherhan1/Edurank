/* eslint-disable no-console */
function sigmoid(x){return 1/(1+Math.exp(-x));}
function pairKey(a,b){return [a,b].sort().join('_');}

function fitBradleyTerryScores(videos,matches){
  const n=videos.length;
  const zero={};
  videos.forEach(v=>{zero[v.id]=0;});
  if(n<2||!matches.length)return zero;
  const idx={};
  videos.forEach((v,i)=>{idx[v.id]=i;});
  const wins=Array.from({length:n},()=>Array(n).fill(0));
  const winTotals=Array(n).fill(0);
  matches.forEach(m=>{
    const wi=idx[m.winnerId],li=idx[m.loserId];
    if(wi===undefined||li===undefined||wi===li)return;
    wins[wi][li]+=1;
    winTotals[wi]+=1;
  });
  let theta=Array(n).fill(1);
  const maxIter=200,eps=1e-9;
  for(let it=0;it<maxIter;it++){
    const next=theta.slice();
    for(let i=0;i<n;i++){
      let denom=0;
      for(let j=0;j<n;j++){
        if(i===j)continue;
        const nij=wins[i][j]+wins[j][i];
        if(!nij)continue;
        denom+=nij/Math.max(theta[i]+theta[j],eps);
      }
      if(denom>eps)next[i]=(winTotals[i]+eps)/denom;
    }
    const g=Math.exp(next.reduce((s,v)=>s+Math.log(Math.max(v,eps)),0)/n);
    for(let i=0;i<n;i++)next[i]=Math.max(next[i]/g,eps);
    let delta=0;
    for(let i=0;i<n;i++)delta=Math.max(delta,Math.abs(next[i]-theta[i]));
    theta=next;
    if(delta<1e-7)break;
  }
  const scores={};
  videos.forEach((v,i)=>{scores[v.id]=Math.log(theta[i]);});
  const mean=Object.values(scores).reduce((s,x)=>s+x,0)/Math.max(Object.values(scores).length,1);
  Object.keys(scores).forEach(k=>{scores[k]-=mean;});
  return scores;
}

function getVideoRankingScore(v,base=1200,scale=200){
  return Number.isFinite(v.btScore)?v.btScore:((v.elo||1200)-base)/Math.max(scale,1);
}

function buildInitialSparsePairs(videos,videoCounts,pairCounts,batchSize,maxPairRepeats){
  const n=videos.length;
  if(n<2)return [];
  const ranked=[...videos].sort((a,b)=>(videoCounts[a.id]||0)-(videoCounts[b.id]||0));
  const out=[],seenBatch=new Set();
  for(let i=0;i<n&&out.length<batchSize;i++){
    const a=ranked[i],b=ranked[(i+1)%n];
    const k=pairKey(a.id,b.id);
    if(seenBatch.has(k)||(pairCounts[k]||0)>=maxPairRepeats)continue;
    out.push([a,b]);seenBatch.add(k);
  }
  for(let i=0;i<n&&out.length<batchSize;i++){
    const a=ranked[i],b=ranked[(i+2)%n];
    const k=pairKey(a.id,b.id);
    if(seenBatch.has(k)||(pairCounts[k]||0)>=maxPairRepeats)continue;
    out.push([a,b]);seenBatch.add(k);
  }
  return out;
}

function selectAdaptivePairsForTopic(videos,matches,batchSize,cfg){
  const minComp=cfg.adaptiveInitialMinComparisons;
  const maxPairRepeats=cfg.adaptiveMaxPairRepeats;
  const adjacentBonus=cfg.adaptiveAdjacentBonus;
  if(videos.length<2)return [];
  const videoCounts={},pairCounts={};
  videos.forEach(v=>{videoCounts[v.id]=0;});
  matches.forEach(m=>{
    videoCounts[m.winnerId]=(videoCounts[m.winnerId]||0)+1;
    videoCounts[m.loserId]=(videoCounts[m.loserId]||0)+1;
    const k=pairKey(m.winnerId,m.loserId);
    pairCounts[k]=(pairCounts[k]||0)+1;
  });
  const needsBootstrap=videos.some(v=>(videoCounts[v.id]||0)<minComp);
  if(needsBootstrap){
    const starter=buildInitialSparsePairs(videos,videoCounts,pairCounts,batchSize,maxPairRepeats);
    if(starter.length>=batchSize||starter.length>0)return starter.slice(0,batchSize);
  }
  const scores=fitBradleyTerryScores(videos,matches);
  const ranking=[...videos].sort((a,b)=>(scores[b.id]||0)-(scores[a.id]||0));
  const rankPos={};
  ranking.forEach((v,i)=>{rankPos[v.id]=i;});
  const maturity=Math.min(1,matches.length/Math.max(videos.length*minComp,1));
  const candidates=[];
  for(let i=0;i<videos.length;i++)for(let j=i+1;j<videos.length;j++){
    const a=videos[i],b=videos[j];
    const k=pairKey(a.id,b.id);
    const repeats=pairCounts[k]||0;
    if(repeats>=maxPairRepeats)continue;
    const p=sigmoid((scores[a.id]||0)-(scores[b.id]||0));
    const uncertainty=p*(1-p);
    const coverageBonus=1/((videoCounts[a.id]||0)+(videoCounts[b.id]||0)+1);
    const repetitionPenalty=1/(1+repeats);
    const rankGap=Math.abs((rankPos[a.id]||0)-(rankPos[b.id]||0));
    const rankBonus=1+((adjacentBonus*maturity)/(rankGap+1));
    candidates.push({pair:[a,b],priority:uncertainty*coverageBonus*repetitionPenalty*rankBonus});
  }
  candidates.sort((x,y)=>y.priority-x.priority);
  return candidates.slice(0,batchSize).map(c=>c.pair);
}

function recomputeRankingsAndStats(videos,matches,topicId){
  const vs=videos.filter(v=>v.topicId===topicId);
  const ms=matches.filter(m=>m.topicId===topicId);
  const scores=fitBradleyTerryScores(vs,ms);
  vs.forEach(v=>{v.btScore=Number.isFinite(scores[v.id])?scores[v.id]:0;});
}

function assert(cond,msg){
  if(!cond)throw new Error(msg);
}

function makeVideos(n,topicId){
  return Array.from({length:n},(_,i)=>({id:`${topicId}_v${i+1}`,topicId,elo:1200,wins:0,losses:0,btScore:0}));
}

function isConnected(videos,pairs){
  const g={};
  videos.forEach(v=>{g[v.id]=[];});
  pairs.forEach(([a,b])=>{g[a.id].push(b.id);g[b.id].push(a.id);});
  if(!videos.length)return true;
  const seen=new Set([videos[0].id]),q=[videos[0].id];
  while(q.length){
    const cur=q.shift();
    (g[cur]||[]).forEach(nxt=>{if(!seen.has(nxt)){seen.add(nxt);q.push(nxt);}});
  }
  return seen.size===videos.length;
}

function testColdStartConnectivity(){
  for(let n=2;n<=24;n++){
    const videos=makeVideos(n,'t');
    const counts={},pairCounts={};
    videos.forEach(v=>{counts[v.id]=0;});
    const pairs=buildInitialSparsePairs(videos,counts,pairCounts,n,3);
    assert(pairs.length>=n-1,`cold-start too sparse for n=${n}`);
    assert(isConnected(videos,pairs),`cold-start graph disconnected for n=${n}`);
  }
  console.log('ok: cold-start graph connectivity for n=2..24');
}

function testVerySmallMatches(){
  const cfg={adaptiveInitialMinComparisons:2,adaptiveMaxPairRepeats:3,adaptiveAdjacentBonus:0.35};
  const videos=makeVideos(6,'t');
  const matches=[{winnerId:videos[0].id,loserId:videos[1].id,topicId:'t'}];
  const pairs=selectAdaptivePairsForTopic(videos,matches,4,cfg);
  assert(pairs.length>0,'expected selectable pairs with very small match history');
  const idsInPairs=new Set(pairs.flatMap(p=>[p[0].id,p[1].id]));
  assert(idsInPairs.size>=4,'very-small-match selection should not collapse to one tiny subset');
  console.log('ok: very small matches');
}

function testRepeatedSamePairPenalty(){
  const cfg={adaptiveInitialMinComparisons:2,adaptiveMaxPairRepeats:3,adaptiveAdjacentBonus:0.35};
  const videos=makeVideos(8,'t');
  const heavyA=videos[0],heavyB=videos[1];
  const matches=[];
  for(let i=0;i<40;i++)matches.push({winnerId:heavyA.id,loserId:heavyB.id,topicId:'t'});
  for(let i=2;i<videos.length;i++)matches.push({winnerId:videos[i].id,loserId:videos[(i+1)%videos.length].id,topicId:'t'});
  const pairs=selectAdaptivePairsForTopic(videos,matches,6,cfg);
  const heavyPairAppears=pairs.some(([a,b])=>pairKey(a.id,b.id)===pairKey(heavyA.id,heavyB.id));
  assert(!heavyPairAppears,'oversampled pair should be filtered out by max repeats');
  console.log('ok: repeated same pair penalty');
}

function testUnevenExposurePrioritization(){
  const cfg={adaptiveInitialMinComparisons:2,adaptiveMaxPairRepeats:5,adaptiveAdjacentBonus:0.35};
  const videos=makeVideos(10,'t');
  const matches=[];
  // First 8 videos heavily compared, last 2 under-exposed.
  for(let r=0;r<6;r++){
    for(let i=0;i<7;i++){
      matches.push({winnerId:videos[i].id,loserId:videos[i+1].id,topicId:'t'});
    }
  }
  const under=[videos[8].id,videos[9].id];
  const pairs=selectAdaptivePairsForTopic(videos,matches,6,cfg);
  const underHits=pairs.filter(([a,b])=>under.includes(a.id)||under.includes(b.id)).length;
  assert(underHits>=2,'uneven exposure should prioritize under-compared items');
  console.log('ok: uneven exposure prioritization');
}

function testDifferentTopicSizes(){
  const videos=[...makeVideos(3,'t_small'),...makeVideos(16,'t_expected'),...makeVideos(21,'t_large')];
  const matches=[];
  for(let i=0;i<30;i++){
    matches.push({winnerId:'t_expected_v1',loserId:`t_expected_v${(i%15)+2}`,topicId:'t_expected'});
  }
  recomputeRankingsAndStats(videos,matches,'t_small');
  recomputeRankingsAndStats(videos,matches,'t_expected');
  recomputeRankingsAndStats(videos,matches,'t_large');
  const sSmall=videos.filter(v=>v.topicId==='t_small').map(v=>v.btScore);
  const sExp=videos.filter(v=>v.topicId==='t_expected').map(v=>v.btScore);
  const sLarge=videos.filter(v=>v.topicId==='t_large').map(v=>v.btScore);
  assert(sSmall.every(Number.isFinite),'small topic BT scores should be finite');
  assert(sExp.every(Number.isFinite),'expected-size topic BT scores should be finite');
  assert(sLarge.every(Number.isFinite),'large topic BT scores should be finite');
  console.log('ok: mixed topic sizes');
}

function testBtRankingIndependenceFromLegacyElo(){
  const videos=makeVideos(5,'t');
  videos.forEach((v,i)=>{v.btScore=2-i;v.elo=100000-(i*9999);}); // intentionally conflicting legacy display.
  const byBt=[...videos].sort((a,b)=>getVideoRankingScore(b)-getVideoRankingScore(a)).map(v=>v.id);
  assert(byBt[0]===videos[0].id&&byBt[4]===videos[4].id,'ranking should follow btScore, not legacy elo');
  console.log('ok: BT ranking independence from legacy display field');
}

function testAdjacentNotOverweightedTooEarly(){
  const cfg={adaptiveInitialMinComparisons:2,adaptiveMaxPairRepeats:3,adaptiveAdjacentBonus:0.35};
  const n=16;
  const videos=makeVideos(n,'t');
  // Construct just-enough bootstrap-like history with weak signal.
  const matches=[];
  for(let i=0;i<n;i++){
    const a=videos[i],b=videos[(i+1)%n];
    matches.push({winnerId:a.id,loserId:b.id,topicId:'t'});
  }
  const pairs=selectAdaptivePairsForTopic(videos,matches,8,cfg);
  // Adjacent by id index is just a rough proxy for "neighbor-heavy" behavior.
  const idx=id=>parseInt(id.split('_v')[1],10);
  const adjCount=pairs.filter(([a,b])=>Math.abs(idx(a.id)-idx(b.id))<=1).length;
  assert(adjCount<=6,'selector appears too concentrated on adjacent pairs at low maturity');
  console.log('ok: adjacent-rank bias damped in low-data phase');
}

function run(){
  testColdStartConnectivity();
  testVerySmallMatches();
  testRepeatedSamePairPenalty();
  testUnevenExposurePrioritization();
  testDifferentTopicSizes();
  testBtRankingIndependenceFromLegacyElo();
  testAdjacentNotOverweightedTooEarly();
  console.log('All adaptive BT edge-case tests passed.');
}

run();
